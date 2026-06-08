"""Run an exported DreamWaQ BPX policy in MuJoCo."""

from __future__ import annotations

import argparse
import json
import pathlib
import select
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import torch


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_XML = _PROJECT_ROOT / "assets" / "BPX" / "mujoco" / "bpx.xml"


def _load_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is not installed in this Python environment. "
            "Install mujoco or run inside the environment that has robot_rl/mjlab working."
        ) from exc
    return mujoco


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _format_command(command: list[float]) -> str:
    return f"vx={command[0]:.2f}, vy={command[1]:.2f}, wz={command[2]:.2f}"


class TerminalCommandController:
    """Non-blocking terminal command controller."""

    def __init__(self, enabled: bool, command_step: float, yaw_step: float):
        self.enabled = enabled and sys.stdin.isatty()
        self.command_step = command_step
        self.yaw_step = yaw_step
        self._fd = None
        self._old_settings = None

    def __enter__(self):
        if self.enabled:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            print(
                "[INFO] Interactive control: "
                "W/S vx, A/D vy, Q/E yaw, F fast, C crawl, Space stop, R reset, +/- step.",
                flush=True,
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll(self, command: list[float]) -> bool:
        if not self.enabled:
            return False
        changed = False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            lower = key.lower()
            if lower == "w":
                command[0] += self.command_step
            elif lower == "s":
                command[0] -= self.command_step
            elif lower == "a":
                command[1] += self.command_step
            elif lower == "d":
                command[1] -= self.command_step
            elif lower == "q":
                command[2] += self.yaw_step
            elif lower == "e":
                command[2] -= self.yaw_step
            elif lower == "f":
                command[0] = max(command[0], 1.0)
            elif lower == "c":
                command[0] = 0.3
            elif lower == "r":
                command[0], command[1], command[2] = 0.6, 0.0, 0.0
            elif key == " ":
                command[0], command[1], command[2] = 0.0, 0.0, 0.0
            elif key == "+":
                self.command_step = _clamp(self.command_step + 0.05, 0.05, 0.5)
                self.yaw_step = _clamp(self.yaw_step + 0.05, 0.05, 0.5)
            elif key == "-":
                self.command_step = _clamp(self.command_step - 0.05, 0.05, 0.5)
                self.yaw_step = _clamp(self.yaw_step - 0.05, 0.05, 0.5)
            else:
                continue
            command[0] = _clamp(command[0], -0.6, 1.8)
            command[1] = _clamp(command[1], -0.4, 0.4)
            command[2] = _clamp(command[2], -0.8, 0.8)
            changed = True

        if changed:
            print(f"[INFO] MuJoCo command: {_format_command(command)}", flush=True)
        return changed


def _load_metadata(policy_path: pathlib.Path, metadata_path: pathlib.Path | None) -> dict:
    if metadata_path is None:
        metadata_path = policy_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}. Run scripts/waq/export_policy.py first."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _add_stairs(root: ET.Element, step_height: float, step_width: float, num_steps: int):
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MuJoCo XML has no <worldbody>.")
    for index in range(num_steps):
        height = step_height * (index + 1)
        x_pos = 1.0 + step_width * index
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"sim2sim_step_{index}",
                "type": "box",
                "pos": f"{x_pos:.4f} 0 {height * 0.5:.4f}",
                "size": f"{step_width * 0.5:.4f} 1.0 {height * 0.5:.4f}",
                "material": "grid_mat",
                "friction": "0.8 0.01 0.001",
            },
        )


def _make_model(mujoco, xml_path: pathlib.Path, terrain: str, step_height: float, step_width: float, num_steps: int):
    xml_path = xml_path.expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(xml_path.parent / "meshes"))

    if terrain == "stairs":
        _add_stairs(root, step_height, step_width, num_steps)
    elif terrain != "flat":
        raise ValueError(f"Unsupported MuJoCo terrain: {terrain}")

    xml_text = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_text)


def _name_id(mujoco, model, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return obj_id


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    rot_w_b = _quat_to_matrix(quat_wxyz)
    return rot_w_b.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _sensor_slice(mujoco, model, data, name: str) -> np.ndarray:
    sensor_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return np.array(data.sensordata[start : start + dim], dtype=np.float64)


class BpxMujocoSim:
    def __init__(self, mujoco, model, metadata: dict, args):
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.metadata = metadata
        self.args = args

        self.joint_names = list(metadata["joint_names"])
        self.default_joint_pos = np.asarray(metadata["default_joint_pos"], dtype=np.float64)
        self.action_scale = np.asarray(metadata["action_scale"], dtype=np.float64) * args.action_scale_multiplier
        self.history_length = int(metadata["history_length"])
        self.num_actor_obs = int(metadata["num_actor_obs"])
        self.num_history_obs = int(metadata["num_history_obs"])
        self.num_actions = int(metadata["num_actions"])
        self.command = [args.command_x, args.command_y, args.command_yaw]
        self.last_action = np.zeros(self.num_actions, dtype=np.float64)
        self.history: deque[np.ndarray] = deque(maxlen=self.history_length)

        control = metadata["control"]
        self.kp = float(control["stiffness"]) * args.kp_multiplier
        self.kd = float(control["damping"]) * args.kd_multiplier
        self.effort_limit = float(control["effort_limit"])
        self.decimation = int(args.decimation or control.get("decimation", 4))
        self.sim_dt = float(args.sim_dt or control.get("sim_dt", 0.005))
        self.model.opt.timestep = self.sim_dt

        if len(self.joint_names) != self.num_actions:
            raise ValueError("Metadata joint count does not match action dimension.")

        self.joint_qpos_addr = np.array(
            [
                self.model.jnt_qposadr[_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in self.joint_names
            ],
            dtype=np.int64,
        )
        self.joint_qvel_addr = np.array(
            [
                self.model.jnt_dofadr[_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in self.joint_names
            ],
            dtype=np.int64,
        )
        self.actuator_ids = np.array(
            [
                _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
                for name in self.joint_names
            ],
            dtype=np.int64,
        )

        self.reset()

    def reset(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, float(self.metadata.get("base_height", 0.42))]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.joint_qpos_addr] = self.default_joint_pos
        self.last_action[:] = 0.0
        self.history.clear()
        self.mujoco.mj_forward(self.model, self.data)

    def joint_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[self.joint_qpos_addr], dtype=np.float64)

    def joint_vel(self) -> np.ndarray:
        return np.array(self.data.qvel[self.joint_qvel_addr], dtype=np.float64)

    def base_ang_vel(self) -> np.ndarray:
        return _sensor_slice(self.mujoco, self.model, self.data, "body_gyro")

    def base_quat(self) -> np.ndarray:
        quat = _sensor_slice(self.mujoco, self.model, self.data, "body_quat")
        if quat.shape[0] != 4:
            quat = np.array(self.data.qpos[3:7], dtype=np.float64)
        return quat

    def build_observation(self) -> np.ndarray:
        obs = np.concatenate(
            [
                self.base_ang_vel() * 0.2,
                _projected_gravity(self.base_quat()),
                np.asarray(self.command, dtype=np.float64),
                self.joint_pos() - self.default_joint_pos,
                self.joint_vel() * 0.05,
                self.last_action,
            ],
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self.num_actor_obs:
            raise RuntimeError(f"Observation dim mismatch: got {obs.shape[0]}, expected {self.num_actor_obs}.")
        return obs

    def update_history(self, obs: np.ndarray) -> np.ndarray:
        if not self.history:
            for _ in range(self.history_length):
                self.history.append(obs.copy())
        else:
            self.history.append(obs.copy())
        history = np.concatenate(list(self.history), axis=0).astype(np.float32)
        if history.shape[0] != self.num_history_obs:
            raise RuntimeError(f"History dim mismatch: got {history.shape[0]}, expected {self.num_history_obs}.")
        return history

    def apply_action(self, action: np.ndarray):
        target = self.default_joint_pos + action * self.action_scale
        for _ in range(self.decimation):
            joint_pos = self.joint_pos()
            joint_vel = self.joint_vel()
            torque = self.kp * (target - joint_pos) - self.kd * joint_vel
            torque = np.clip(torque, -self.effort_limit, self.effort_limit)
            self.data.ctrl[self.actuator_ids] = torque
            self.mujoco.mj_step(self.model, self.data)

    def measured_velocity(self) -> np.ndarray:
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        linear_world = np.asarray(self.data.qvel[0:3], dtype=np.float64)
        return _quat_to_matrix(quat).T @ linear_world


def _run_loop(policy, sim: BpxMujocoSim, args, viewer=None):
    device = torch.device(args.device)
    policy.to(device)
    policy.eval()
    control_dt = sim.sim_dt * sim.decimation
    start_wall = time.time()
    last_print = 0.0
    step = 0

    with TerminalCommandController(args.interactive, args.command_step, args.yaw_step) as controller:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if args.duration > 0.0 and sim.data.time >= args.duration:
                break

            step_start = time.time()
            controller.poll(sim.command)
            obs = sim.build_observation()
            history = sim.update_history(obs)
            with torch.inference_mode():
                action_tensor = policy(
                    torch.from_numpy(obs).unsqueeze(0).to(device),
                    torch.from_numpy(history).unsqueeze(0).to(device),
                )
            action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
            if args.clip_actions > 0.0:
                action = np.clip(action, -args.clip_actions, args.clip_actions)
            sim.last_action = action
            sim.apply_action(action)

            if viewer is not None:
                viewer.sync()

            if sim.data.time - last_print >= args.print_interval:
                vel = sim.measured_velocity()
                print(
                    "[INFO] "
                    f"t={sim.data.time:6.2f}s "
                    f"cmd=({_format_command(sim.command)}) "
                    f"vel_x={vel[0]: .3f} vel_y={vel[1]: .3f} "
                    f"action_mean_abs={np.mean(np.abs(action)): .3f} "
                    f"height={sim.data.qpos[2]: .3f}",
                    flush=True,
                )
                last_print = sim.data.time

            if args.real_time:
                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            step += 1

    print(f"[INFO] MuJoCo sim2sim finished at t={sim.data.time:.2f}s, wall={time.time() - start_wall:.2f}s.")


def main():
    parser = argparse.ArgumentParser(description="Play an exported DreamWaQ BPX policy in MuJoCo.")
    parser.add_argument("--policy", required=True, help="Exported TorchScript policy path.")
    parser.add_argument("--metadata", default=None, help="Policy metadata JSON. Defaults to <policy>.json.")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="BPX MuJoCo XML path.")
    parser.add_argument("--terrain", choices=("flat", "stairs"), default="flat", help="Simple MuJoCo terrain.")
    parser.add_argument("--step_height", "--step-height", dest="step_height", type=float, default=0.08)
    parser.add_argument("--step_width", "--step-width", dest="step_width", type=float, default=0.35)
    parser.add_argument("--num_steps", "--num-steps", dest="num_steps", type=int, default=5)
    parser.add_argument("--command_x", "--command-x", dest="command_x", type=float, default=0.6)
    parser.add_argument("--command_y", "--command-y", dest="command_y", type=float, default=0.0)
    parser.add_argument("--command_yaw", "--command-yaw", dest="command_yaw", type=float, default=0.0)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--command_step", "--command-step", dest="command_step", type=float, default=0.1)
    parser.add_argument("--yaw_step", "--yaw-step", dest="yaw_step", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds. Use 0 to run forever.")
    parser.add_argument("--headless", action="store_true", default=False, help="Do not open the MuJoCo viewer.")
    parser.add_argument("--real_time", "--real-time", dest="real_time", action="store_true", default=False)
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--sim_dt", "--sim-dt", dest="sim_dt", type=float, default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--clip_actions", "--clip-actions", dest="clip_actions", type=float, default=0.0)
    parser.add_argument("--kp_multiplier", "--kp-multiplier", dest="kp_multiplier", type=float, default=1.0)
    parser.add_argument("--kd_multiplier", "--kd-multiplier", dest="kd_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--action_scale_multiplier",
        "--action-scale-multiplier",
        dest="action_scale_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument("--print_interval", "--print-interval", dest="print_interval", type=float, default=1.0)
    args = parser.parse_args()

    mujoco = _load_mujoco()
    policy_path = pathlib.Path(args.policy).expanduser().resolve()
    metadata_path = pathlib.Path(args.metadata).expanduser().resolve() if args.metadata else None
    metadata = _load_metadata(policy_path, metadata_path)
    policy = torch.jit.load(str(policy_path), map_location=args.device)
    model = _make_model(
        mujoco,
        pathlib.Path(args.xml),
        args.terrain,
        args.step_height,
        args.step_width,
        args.num_steps,
    )
    sim = BpxMujocoSim(mujoco, model, metadata, args)

    print(f"[INFO] Loaded policy: {policy_path}")
    print(f"[INFO] Loaded MuJoCo XML: {pathlib.Path(args.xml).expanduser().resolve()}")
    print(
        "[INFO] Sim2sim control: "
        f"terrain={args.terrain}, dt={sim.sim_dt}, decimation={sim.decimation}, "
        f"kp={sim.kp:.4f}, kd={sim.kd:.4f}, effort={sim.effort_limit:.2f}, "
        f"action_scale_mean={float(np.mean(sim.action_scale)):.4f}"
    )

    if args.headless:
        _run_loop(policy, sim, args, viewer=None)
        return

    try:
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit("mujoco.viewer is unavailable. Re-run with --headless or install viewer dependencies.") from exc

    with mujoco.viewer.launch_passive(model, sim.data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 130
        _run_loop(policy, sim, args, viewer=viewer)


if __name__ == "__main__":
    main()
