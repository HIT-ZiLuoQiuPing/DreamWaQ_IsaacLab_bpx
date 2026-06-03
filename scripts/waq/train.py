# Copyright (c) 2026

"""Train a DreamWaQ-style BPX locomotion policy."""

from __future__ import annotations

import argparse
import atexit
import inspect
import os
import pathlib
import shutil
import sys
from datetime import datetime

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "source" / "isaaclab_waq"))
os.environ.setdefault("WARP_CACHE_PATH", str(_PROJECT_ROOT / ".cache" / "warp"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train a DreamWaQ-style BPX policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-BPX-WAQ-Rough-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and policy.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Optional suffix for the log directory.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a WAQ checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument(
    "--reset_curriculum_on_resume",
    action="store_true",
    default=False,
    help="Resume model weights but restart command/terrain curricula from the beginning.",
)
parser.add_argument(
    "--curriculum_offset_iterations",
    type=int,
    default=None,
    help="Override command/terrain curriculum progress when resuming, measured in learning iterations.",
)
parser.add_argument("--num_steps_per_env", type=int, default=None, help="Rollout steps per environment per iteration.")
parser.add_argument("--ppo_epochs", type=int, default=None, help="PPO learning epochs per rollout.")
parser.add_argument("--height_scan_resolution", type=float, default=None, help="Override height scanner grid resolution.")
parser.add_argument(
    "--height_scan_update_stride",
    type=int,
    default=None,
    help="Height scanner update period in control steps. Larger values train faster.",
)
parser.add_argument(
    "--no_random_ep_len",
    action="store_true",
    default=False,
    help="Disable random episode-length initialization at training start.",
)
parser.add_argument("--no_console_log", action="store_true", default=False, help="Do not tee stdout/stderr to console.log.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from isaaclab_waq.assets.robots.bpx import BPX_ACTION_SCALE, BPX_DAMPING, BPX_EFFORT_LIMIT, BPX_STIFFNESS
import isaaclab_waq.tasks  # noqa: F401
from isaaclab_waq.algorithms.waq import DreamWaQConfig, DreamWaQRunner


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


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


def _latest_checkpoint(log_root_path: str) -> str | None:
    log_root = pathlib.Path(log_root_path)
    if not log_root.exists():
        return None
    candidates = sorted(log_root.glob("*/model_*.pt"), key=lambda path: path.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = _parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    agent_cfg = _load_waq_cfg(args_cli.task)
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.ppo_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = args_cli.ppo_epochs
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    if args_cli.height_scan_resolution is not None:
        env_cfg.scene.height_scanner.pattern_cfg.resolution = args_cli.height_scan_resolution
    if args_cli.height_scan_update_stride is not None:
        env_cfg.scene.height_scanner.update_period = (
            args_cli.height_scan_update_stride * env_cfg.decimation * env_cfg.sim.dt
        )

    print(
        "[INFO] BPX IsaacLab bootstrap actuator profile: "
        f"effort={BPX_EFFORT_LIMIT:.2f}, stiffness={BPX_STIFFNESS:.2f}, "
        f"damping={BPX_DAMPING:.2f}, action_scale={next(iter(BPX_ACTION_SCALE.values())):.3f}"
    )

    env_cfg.seed = agent_cfg.seed
    torch.manual_seed(agent_cfg.seed)

    log_root_path = os.path.abspath(os.path.join("logs", "waq", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if not args_cli.no_console_log:
        console_log_path = os.path.join(log_dir, "console.log")
        console_log_file = open(console_log_path, "a", buffering=1)
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = _Tee(stdout, console_log_file)
        sys.stderr = _Tee(stderr, console_log_file)

        def _close_console_log():
            sys.stdout = stdout
            sys.stderr = stderr
            console_log_file.close()

        atexit.register(_close_console_log)
        print(f"[INFO] Console output is being saved to: {console_log_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = DreamWaQRunner(env, agent_cfg, log_dir=log_dir, device=device)
    if args_cli.resume:
        checkpoint = args_cli.checkpoint or _latest_checkpoint(log_root_path)
        if checkpoint is None:
            raise FileNotFoundError(f"No WAQ checkpoint found under {log_root_path}.")
        print(f"[INFO] Loading WAQ checkpoint from: {checkpoint}")
        curriculum_step_offset = None
        if args_cli.reset_curriculum_on_resume:
            curriculum_step_offset = 0
        if args_cli.curriculum_offset_iterations is not None:
            curriculum_step_offset = args_cli.curriculum_offset_iterations * agent_cfg.num_steps_per_env
        runner.load(checkpoint, curriculum_step_offset=curriculum_step_offset)
        env_unwrapped = getattr(env, "unwrapped", env)
        print(
            "[INFO] WAQ curriculum step offset: "
            f"{getattr(env_unwrapped, '_waq_curriculum_step_offset', 'default')}"
        )

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg.to_dict())
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=not args_cli.no_random_ep_len)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
