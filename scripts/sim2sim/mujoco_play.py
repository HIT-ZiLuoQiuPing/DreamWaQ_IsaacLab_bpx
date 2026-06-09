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
LEG_PREFIXES = ("fl", "fr", "hl", "hr")
JOINT_SUFFIXES = ("hip_roll_joint", "hip_pitch_joint", "knee_joint")


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


def _joint_order(metadata_joint_names: list[str], order: str) -> list[str]:
    if order == "metadata":
        return list(metadata_joint_names)
    if order == "type_major":
        return [f"{leg}_{suffix}" for suffix in JOINT_SUFFIXES for leg in LEG_PREFIXES]
    if order == "alphabetical":
        return sorted(metadata_joint_names)
    raise ValueError(f"Unsupported joint order: {order}")


class BpxMujocoSim:
    def __init__(self, mujoco, model, metadata: dict, args):
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.metadata = metadata
        self.args = args

        metadata_joint_names = list(metadata["joint_names"])
        self.joint_names = _joint_order(metadata_joint_names, args.joint_order)
        default_by_name = dict(zip(metadata_joint_names, metadata["default_joint_pos"]))
        scale_by_name = dict(zip(metadata_joint_names, metadata["action_scale"]))
        self.default_joint_pos = np.asarray([default_by_name[name] for name in self.joint_names], dtype=np.float64)
        self.action_scale = (
            np.asarray([scale_by_name[name] for name in self.joint_names], dtype=np.float64)
            * args.action_scale_multiplier
        )
        self.action_sign = np.ones_like(self.action_scale)
        for index, name in enumerate(self.joint_names):
            if args.flip_hip_roll and name.endswith("_hip_roll_joint"):
                self.action_sign[index] *= -1.0
            if args.flip_hip_pitch and name.endswith("_hip_pitch_joint"):
                self.action_sign[index] *= -1.0
            if args.flip_knee and name.endswith("_knee_joint"):
                self.action_sign[index] *= -1.0
        self.history_length = int(metadata["history_length"])
        self.num_actor_obs = int(metadata["num_actor_obs"])
        self.num_history_obs = int(metadata["num_history_obs"])
        self.num_actions = int(metadata["num_actions"])
        self.command = [args.command_x, args.command_y, args.command_yaw]
        self.last_action = np.zeros(self.num_actions, dtype=np.float64)
        self.history_terms: list[deque[np.ndarray]] = []

        control = metadata["control"]
        self.kp = float(control["stiffness"]) * args.kp_multiplier
        self.kd = float(control["damping"]) * args.kd_multiplier
        self.effort_limit = float(control["effort_limit"])
        self.armature = float(args.armature if args.armature is not None else control.get("armature", 0.005))
        self.joint_friction = float(
            args.joint_friction if args.joint_friction is not None else control.get("joint_friction", 0.01)
        )
        self.decimation = int(args.decimation or control.get("decimation", 4))
        self.sim_dt = float(args.sim_dt or control.get("sim_dt", 0.005))
        self.model.opt.timestep = self.sim_dt

        if len(self.joint_names) != self.num_actions:
            raise ValueError("Metadata joint count does not match action dimension.")

        self.joint_ids = np.array(
            [
                _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ],
            dtype=np.int64,
        )
        self.joint_qpos_addr = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=np.int64)
        self.joint_qvel_addr = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int64,
        )
        self.actuator_ids = np.array(
            [
                _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
                for name in self.joint_names
            ],
            dtype=np.int64,
        )
        self.joint_range = np.asarray(self.model.jnt_range[self.joint_ids], dtype=np.float64)
        self.target_low = self.joint_range[:, 0] + args.joint_limit_margin
        self.target_high = self.joint_range[:, 1] - args.joint_limit_margin
        self.model.dof_armature[self.joint_qvel_addr] = self.armature
        self.model.dof_frictionloss[self.joint_qvel_addr] = self.joint_friction
        self.actuator_mode = args.actuator_mode
        if self.actuator_mode == "position":
            self._configure_position_actuators()

        self.reset()

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, float(self.metadata.get("base_height", 0.42))]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.joint_qpos_addr] = self.default_joint_pos
        self.last_action[:] = 0.0
        self.history_terms.clear()
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

    def build_observation_terms(self) -> list[np.ndarray]:
        return [
            (self.base_ang_vel() * 0.2).astype(np.float32),
            _projected_gravity(self.base_quat()).astype(np.float32),
            np.asarray(self.command, dtype=np.float32),
            (self.joint_pos() - self.default_joint_pos).astype(np.float32),
            (self.joint_vel() * 0.05).astype(np.float32),
            self.last_action.astype(np.float32),
        ]

    def build_observation(self, terms: list[np.ndarray] | None = None) -> np.ndarray:
        if terms is None:
            terms = self.build_observation_terms()
        obs = np.concatenate(terms, axis=0).astype(np.float32)
        if obs.shape[0] != self.num_actor_obs:
            raise RuntimeError(f"Observation dim mismatch: got {obs.shape[0]}, expected {self.num_actor_obs}.")
        return obs

    def update_history(self, terms: list[np.ndarray]) -> np.ndarray:
        if not self.history_terms:
            self.history_terms = [deque(maxlen=self.history_length) for _ in terms]
            for term_buffer, term in zip(self.history_terms, terms):
                for _ in range(self.history_length):
                    term_buffer.append(term.copy())
        else:
            for term_buffer, term in zip(self.history_terms, terms):
                term_buffer.append(term.copy())
        if self.args.history_layout == "term_major":
            # IsaacLab applies history to each observation term before concatenating the group.
            # CENet sees [term0_history, term1_history, ...], not [frame0_obs, frame1_obs, ...].
            history = np.concatenate(
                [np.concatenate(list(term_buffer), axis=0) for term_buffer in self.history_terms],
                axis=0,
            ).astype(np.float32)
        else:
            frames = []
            for index in range(self.history_length):
                frames.append(np.concatenate([term_buffer[index] for term_buffer in self.history_terms], axis=0))
            history = np.concatenate(frames, axis=0).astype(np.float32)
        if history.shape[0] != self.num_history_obs:
            raise RuntimeError(f"History dim mismatch: got {history.shape[0]}, expected {self.num_history_obs}.")
        return history

    def _configure_position_actuators(self):
        """Match mjlab's BuiltinPositionActuatorCfg on top of the BPX motor XML."""

        for actuator_id, joint_id in zip(self.actuator_ids, self.joint_ids):
            self.model.actuator_dyntype[actuator_id] = self.mujoco.mjtDyn.mjDYN_NONE
            self.model.actuator_gaintype[actuator_id] = self.mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[actuator_id] = self.mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[actuator_id, :] = 0.0
            self.model.actuator_biasprm[actuator_id, :] = 0.0
            self.model.actuator_gainprm[actuator_id, 0] = self.kp
            self.model.actuator_biasprm[actuator_id, 1] = -self.kp
            self.model.actuator_biasprm[actuator_id, 2] = -self.kd
            self.model.actuator_ctrllimited[actuator_id] = False
            self.model.actuator_forcelimited[actuator_id] = True
            self.model.actuator_forcerange[actuator_id, :] = [-self.effort_limit, self.effort_limit]
            joint_low, joint_high = self.model.jnt_range[joint_id]
            delta = self.effort_limit / max(self.kp, 1.0e-6)
            self.model.actuator_ctrlrange[actuator_id, :] = [joint_low - delta, joint_high + delta]

    def action_target(self, action: np.ndarray) -> np.ndarray:
        target = self.default_joint_pos + (action * self.action_sign) * self.action_scale
        if self.args.clip_targets:
            target = np.clip(target, self.target_low, self.target_high)
        return target

    def apply_action(self, action: np.ndarray) -> bool:
        if not np.all(np.isfinite(action)):
            return False
        target = self.action_target(action)
        for _ in range(self.decimation):
            if self.actuator_mode == "position":
                self.data.ctrl[self.actuator_ids] = target
            else:
                joint_pos = self.joint_pos()
                joint_vel = self.joint_vel()
                torque = self.kp * (target - joint_pos) - self.kd * joint_vel
                torque = np.clip(torque, -self.effort_limit, self.effort_limit)
                self.data.ctrl[self.actuator_ids] = torque
            self.mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                return False
            if self.data.qpos[2] < self.args.min_base_height or self.data.qpos[2] > self.args.max_base_height:
                return False
            if np.max(np.abs(self.data.qvel)) > self.args.max_qvel:
                return False
        return True

    def measured_velocity(self) -> np.ndarray:
        quat = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        linear_world = np.asarray(self.data.qvel[0:3], dtype=np.float64)
        return _quat_to_matrix(quat).T @ linear_world

    def debug_summary(self, obs: np.ndarray, action: np.ndarray | None = None) -> str:
        joint_pos = self.joint_pos()
        joint_vel = self.joint_vel()
        lines = [
            f"base_z={self.data.qpos[2]:.4f}",
            f"base_ang_vel={obs[0:3].tolist()}",
            f"projected_gravity={obs[3:6].tolist()}",
            f"command={obs[6:9].tolist()}",
            f"joint_pos_rel={obs[9:21].tolist()}",
            f"joint_vel_scaled={obs[21:33].tolist()}",
            f"last_action={obs[33:45].tolist()}",
            f"joint_pos={joint_pos.tolist()}",
            f"joint_vel={joint_vel.tolist()}",
        ]
        if action is not None:
            target = self.action_target(action)
            lines.extend(
                [
                    f"raw_or_clipped_action={action.tolist()}",
                    f"action_sign={self.action_sign.tolist()}",
                    f"target_joint_pos={target.tolist()}",
                    f"target_minus_joint={((target - joint_pos).tolist())}",
                ]
            )
        return "\n".join(f"[DEBUG] {line}" for line in lines)


def _run_loop(policy, sim: BpxMujocoSim, args, viewer=None):
    device = torch.device(args.device)
    policy.to(device)
    policy.eval()
    control_dt = sim.sim_dt * sim.decimation
    start_wall = time.time()
    last_print = 0.0
    step = 0
    reset_count = 0

    warmup_steps = int(max(args.stand_warmup, 0.0) / max(control_dt, 1.0e-6))
    if warmup_steps > 0:
        print(f"[INFO] Standing warmup for {warmup_steps} policy steps ({args.stand_warmup:.2f}s).", flush=True)
        zero_action = np.zeros(sim.num_actions, dtype=np.float64)
        for _ in range(warmup_steps):
            ok = sim.apply_action(zero_action)
            if viewer is not None:
                viewer.sync()
            if not ok:
                print("[WARN] MuJoCo became unstable during stand warmup; resetting.", flush=True)
                sim.reset()
                break
        if args.debug_obs:
            obs = sim.build_observation()
            print(sim.debug_summary(obs, zero_action), flush=True)

    if args.stand_only:
        print("[INFO] Running stand-only PD test. Policy will not be used.", flush=True)
        zero_action = np.zeros(sim.num_actions, dtype=np.float64)
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if args.duration > 0.0 and sim.data.time >= args.duration:
                break
            step_start = time.time()
            ok = sim.apply_action(zero_action)
            if viewer is not None:
                viewer.sync()
            if not ok:
                reset_count += 1
                print(f"[WARN] MuJoCo unstable during stand-only; resetting (reset_count={reset_count}).", flush=True)
                if args.disable_safety_reset:
                    break
                sim.reset()
            if sim.data.time - last_print >= args.print_interval:
                obs = sim.build_observation()
                print(
                    "[INFO] "
                    f"t={sim.data.time:6.2f}s stand-only height={sim.data.qpos[2]: .3f} "
                    f"joint_vel_abs={np.mean(np.abs(sim.joint_vel())): .4f}",
                    flush=True,
                )
                if args.debug_obs:
                    print(sim.debug_summary(obs, zero_action), flush=True)
                last_print = sim.data.time
            if args.real_time:
                sleep_time = control_dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
        print(
            f"[INFO] MuJoCo stand-only finished at t={sim.data.time:.2f}s, "
            f"wall={time.time() - start_wall:.2f}s, resets={reset_count}."
        )
        return

    with TerminalCommandController(args.interactive, args.command_step, args.yaw_step) as controller:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if args.duration > 0.0 and sim.data.time >= args.duration:
                break

            step_start = time.time()
            controller.poll(sim.command)
            obs_terms = sim.build_observation_terms()
            obs = sim.build_observation(obs_terms)
            history = sim.update_history(obs_terms)
            with torch.inference_mode():
                action_tensor = policy(
                    torch.from_numpy(obs).unsqueeze(0).to(device),
                    torch.from_numpy(history).unsqueeze(0).to(device),
                )
            action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
            if args.clip_actions > 0.0:
                action = np.clip(action, -args.clip_actions, args.clip_actions)
            sim.last_action = action
            ok = sim.apply_action(action)
            if not ok:
                reset_count += 1
                print(
                    "[WARN] MuJoCo unstable state detected; resetting "
                    f"(reset_count={reset_count}, t={sim.data.time:.3f}s).",
                    flush=True,
                )
                if args.disable_safety_reset:
                    break
                sim.reset()
                last_print = 0.0
                continue

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
                    f"action_max_abs={np.max(np.abs(action)): .3f} "
                    f"height={sim.data.qpos[2]: .3f}",
                    flush=True,
                )
                if args.debug_obs:
                    print(sim.debug_summary(obs, action), flush=True)
                last_print = sim.data.time

            if args.real_time:
                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            step += 1

    print(
        f"[INFO] MuJoCo sim2sim finished at t={sim.data.time:.2f}s, "
        f"wall={time.time() - start_wall:.2f}s, resets={reset_count}."
    )


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
    parser.add_argument("--stand_only", "--stand-only", dest="stand_only", action="store_true", default=False)
    parser.add_argument("--debug_obs", "--debug-obs", dest="debug_obs", action="store_true", default=False)
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--sim_dt", "--sim-dt", dest="sim_dt", type=float, default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument(
        "--clip_actions",
        "--clip-actions",
        dest="clip_actions",
        type=float,
        default=4.0,
        help="Raw policy action safety clip. Use 0 to disable.",
    )
    parser.add_argument("--kp_multiplier", "--kp-multiplier", dest="kp_multiplier", type=float, default=1.0)
    parser.add_argument("--kd_multiplier", "--kd-multiplier", dest="kd_multiplier", type=float, default=1.0)
    parser.add_argument("--armature", type=float, default=None, help="Override actuated joint armature.")
    parser.add_argument("--joint_friction", "--joint-friction", dest="joint_friction", type=float, default=None)
    parser.add_argument("--joint_limit_margin", "--joint-limit-margin", dest="joint_limit_margin", type=float, default=0.03)
    parser.add_argument("--stand_warmup", "--stand-warmup", dest="stand_warmup", type=float, default=0.5)
    parser.add_argument("--min_base_height", "--min-base-height", dest="min_base_height", type=float, default=0.06)
    parser.add_argument("--max_base_height", "--max-base-height", dest="max_base_height", type=float, default=1.5)
    parser.add_argument("--max_qvel", "--max-qvel", dest="max_qvel", type=float, default=80.0)
    parser.add_argument("--disable_safety_reset", "--disable-safety-reset", dest="disable_safety_reset", action="store_true", default=False)
    parser.add_argument(
        "--joint_order",
        "--joint-order",
        dest="joint_order",
        choices=("metadata", "type_major", "alphabetical"),
        default="metadata",
        help="Joint order used for policy observations/actions. metadata is the exported order; type_major tests roll/pitch/knee grouped by joint type.",
    )
    parser.add_argument(
        "--history_layout",
        "--history-layout",
        dest="history_layout",
        choices=("term_major", "frame_major"),
        default="term_major",
        help="CENet history layout. term_major matches IsaacLab observation history; frame_major is kept for ablation.",
    )
    parser.add_argument(
        "--actuator_mode",
        "--actuator-mode",
        dest="actuator_mode",
        choices=("position", "torque_pd"),
        default="position",
        help="MuJoCo actuator mode. position matches mjlab BuiltinPositionActuatorCfg; torque_pd keeps the older external PD path.",
    )
    parser.add_argument(
        "--clip_targets",
        "--clip-targets",
        dest="clip_targets",
        action="store_true",
        default=False,
        help="Clip position targets to joint limits. Disabled by default to match mjlab position actuators.",
    )
    parser.add_argument("--flip_hip_roll", "--flip-hip-roll", dest="flip_hip_roll", action="store_true", default=False)
    parser.add_argument("--flip_hip_pitch", "--flip-hip-pitch", dest="flip_hip_pitch", action="store_true", default=False)
    parser.add_argument("--flip_knee", "--flip-knee", dest="flip_knee", action="store_true", default=False)
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
        f"terrain={args.terrain}, actuator_mode={sim.actuator_mode}, joint_order={args.joint_order}, "
        f"history_layout={args.history_layout}, "
        f"dt={sim.sim_dt}, decimation={sim.decimation}, "
        f"kp={sim.kp:.4f}, kd={sim.kd:.4f}, effort={sim.effort_limit:.2f}, "
        f"armature={sim.armature:.4f}, joint_friction={sim.joint_friction:.4f}, "
        f"action_scale_mean={float(np.mean(sim.action_scale)):.4f}, "
        f"clip_actions={args.clip_actions:.2f}, clip_targets={args.clip_targets}"
    )
    print(f"[INFO] Joint order: {sim.joint_names}", flush=True)
    if args.flip_hip_roll or args.flip_hip_pitch or args.flip_knee:
        print(
            "[INFO] Action sign flips: "
            f"hip_roll={args.flip_hip_roll}, hip_pitch={args.flip_hip_pitch}, knee={args.flip_knee}",
            flush=True,
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
