# Copyright (c) 2026

"""Play a DreamWaQ-style BPX checkpoint."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "source" / "isaaclab_waq"))
os.environ.setdefault("WARP_CACHE_PATH", str(_PROJECT_ROOT / ".cache" / "warp"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))

from isaaclab.app import AppLauncher


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _has_nvidia_device() -> bool:
    if os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/nvidia0"):
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _arg_present(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


parser = argparse.ArgumentParser(description="Play a DreamWaQ-style BPX policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-BPX-WAQ-Rough-Play-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum play steps. Use 0 to run until the app stops.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--video", action="store_true", default=False, help="Record an mp4 video instead of opening a GUI.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video in sim steps.")
parser.add_argument("--video_dir", type=str, default=None, help="Optional directory for recorded videos.")
parser.add_argument("--gui", action="store_true", default=False, help="Allow GUI/RTX rendering for local visualization.")
parser.add_argument("--command_x", type=float, default=0.4, help="Fixed forward command used for play.")
parser.add_argument("--command_y", type=float, default=0.0, help="Fixed lateral command used for play.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw command used for play.")
parser.add_argument("--random_commands", action="store_true", default=False, help="Use environment-randomized commands.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

has_nvidia_device = _has_nvidia_device()
livestream_requested = _arg_present("--livestream")
if args_cli.video:
    args_cli.headless = True
    args_cli.livestream = 0
    args_cli.enable_cameras = True
    os.environ["HEADLESS"] = "1"
    os.environ["LIVESTREAM"] = "0"
    os.environ["ENABLE_CAMERAS"] = "1"
    print("[INFO] Recording WAQ play video with headless rendering.", flush=True)
elif not args_cli.gui and not livestream_requested:
    args_cli.headless = True
    args_cli.livestream = 0
    args_cli.enable_cameras = False
    os.environ["HEADLESS"] = "1"
    os.environ["LIVESTREAM"] = "0"
    os.environ["ENABLE_CAMERAS"] = "0"
    print(
        "[INFO] Running WAQ play in strict headless mode. "
        "Pass --gui or --livestream explicitly to enable rendering.",
        flush=True,
    )
elif not args_cli.headless and args_cli.livestream <= 0 and (not _has_display() or not has_nvidia_device):
    args_cli.headless = True
    args_cli.livestream = 0
    args_cli.enable_cameras = False
    os.environ["HEADLESS"] = "1"
    os.environ["LIVESTREAM"] = "0"
    os.environ["ENABLE_CAMERAS"] = "0"
    print("[INFO] No usable display/NVIDIA device detected. Running WAQ play in headless mode.", flush=True)
if not getattr(args_cli, "device_explicit", False) and not has_nvidia_device:
    args_cli.device = "cpu"
    print("[INFO] No NVIDIA device detected. Using --device cpu for WAQ play.", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_waq.tasks  # noqa: F401
from isaaclab_waq.algorithms.waq import DreamWaQConfig, DreamWaQRunner


def _parse_env_cfg(task_name: str, device: str, num_envs: int | None = None, entry_point_key: str = "env_cfg_entry_point"):
    cfg = load_cfg_from_registry(task_name, entry_point_key)
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for task '{task_name}' must be a config class, not a dict.")
    cfg.sim.device = device
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


def _load_waq_cfg(task_name: str) -> DreamWaQConfig:
    cfg = load_cfg_from_registry(task_name, "waq_cfg_entry_point")
    if isinstance(cfg, type):
        cfg = cfg()
    return cfg


def _unpack_observations(result):
    if isinstance(result, tuple):
        obs, extras = result
    else:
        obs, extras = result, {}
    return obs, extras


def _groups(extras: dict) -> dict:
    return extras.get("observations", {}) if isinstance(extras, dict) else {}


def _set_fixed_command(env, command: tuple[float, float, float]) -> bool:
    try:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        command_term.vel_command_b[:, 0] = command[0]
        command_term.vel_command_b[:, 1] = command[1]
        command_term.vel_command_b[:, 2] = command[2]
    except (AttributeError, KeyError):
        return False
    return True


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = _parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    agent_cfg = _load_waq_cfg(args_cli.task)
    log_root_path = os.path.abspath(os.path.join("logs", "waq", agent_cfg.experiment_name))
    checkpoint = args_cli.checkpoint or get_checkpoint_path(log_root_path, ".*", "model_.*.pt")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    try:
        env.unwrapped.sim.set_camera_view(eye=(2.6, -3.0, 1.4), target=(0.0, 0.0, 0.35))
    except AttributeError:
        pass

    if args_cli.video:
        video_folder = args_cli.video_dir or os.path.join(os.path.dirname(checkpoint), "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_folder}", flush=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = DreamWaQRunner(env, agent_cfg, log_dir=None, device=device)
    print(f"[INFO] Loading WAQ checkpoint from: {checkpoint}", flush=True)
    try:
        runner.load(checkpoint, load_optimizer=False)
    except RuntimeError as exc:
        env.close()
        print(
            "[ERROR] The checkpoint does not match the current WAQ model/environment configuration. "
            "Use a checkpoint trained after the latest observation/network changes, or restart training.",
            file=sys.stderr,
            flush=True,
        )
        print(f"[ERROR] Incompatible checkpoint: {checkpoint}", file=sys.stderr, flush=True)
        print(f"[ERROR] Loader detail: {exc}", file=sys.stderr, flush=True)
        simulation_app.close()
        raise SystemExit(2) from exc
    policy = runner.get_inference_policy()

    fixed_command = (args_cli.command_x, args_cli.command_y, args_cli.command_yaw)
    if not args_cli.random_commands:
        if _set_fixed_command(env, fixed_command):
            print(f"[INFO] Using fixed play command: vx={fixed_command[0]:.2f}, vy={fixed_command[1]:.2f}, wz={fixed_command[2]:.2f}")

    obs, extras = _unpack_observations(env.get_observations())
    history = _groups(extras)["cenet"]
    dt = env.unwrapped.step_dt

    step = 0
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or step < args_cli.max_steps):
        start_time = time.time()
        with torch.inference_mode():
            if not args_cli.random_commands:
                _set_fixed_command(env, fixed_command)
            actions = policy(obs.to(device), history.to(device))
            obs, _, _, infos = env.step(actions)
            if not args_cli.random_commands:
                _set_fixed_command(env, fixed_command)
                obs, extras = _unpack_observations(env.get_observations())
                history = _groups(extras)["cenet"]
            else:
                history = _groups(infos)["cenet"]
        step += 1
        if args_cli.video and step >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[INFO] WAQ play finished after {step} steps.", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
