"""Export a DreamWaQ checkpoint to a single-input BPX sim2real policy."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import torch
import torch.nn as nn

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "source" / "isaaclab_waq"))

from export_policy import (  # noqa: E402
    DreamWaQJitPolicy,
    JOINT_NAMES,
    LEGACY_PROFILE_NAME,
    _control_profile,
    _infer_dimensions,
    _joint_action_scale,
    _joint_action_sign,
    _joint_default,
    _policy_cfg_from_checkpoint,
)
from isaaclab_waq.algorithms.waq.actor_critic import DreamWaQActorCritic  # noqa: E402


OBS_TERM_DIMS = (3, 3, 3, 12, 12, 12)
SIMREAL_ACTOR_OBS_DIM = 45
SIMREAL_HISTORY_LENGTH = 5
SIMREAL_HISTORY_ONLY_INPUT_DIM = SIMREAL_ACTOR_OBS_DIM * SIMREAL_HISTORY_LENGTH


class SimRealSingleInputPolicy(nn.Module):
    """Adapter for SDK/UI runners that can only call policy(x)."""

    def __init__(
        self,
        policy: DreamWaQJitPolicy,
        num_actor_obs: int,
        history_length: int,
        include_current_obs: bool,
    ):
        super().__init__()
        if sum(OBS_TERM_DIMS) != num_actor_obs:
            raise ValueError(f"Unsupported actor obs layout: {num_actor_obs}.")
        self.policy = policy
        self.num_actor_obs = num_actor_obs
        self.history_length = history_length
        self.include_current_obs = include_current_obs

    def _frame_major_to_term_major(self, frame_history: torch.Tensor) -> torch.Tensor:
        batch = frame_history.shape[0]
        frames = frame_history.reshape(batch, self.history_length, self.num_actor_obs)
        base_ang_vel = frames[:, :, 0:3].reshape(batch, -1)
        projected_gravity = frames[:, :, 3:6].reshape(batch, -1)
        command = frames[:, :, 6:9].reshape(batch, -1)
        joint_pos = frames[:, :, 9:21].reshape(batch, -1)
        joint_vel = frames[:, :, 21:33].reshape(batch, -1)
        last_action = frames[:, :, 33:45].reshape(batch, -1)
        return torch.cat((base_ang_vel, projected_gravity, command, joint_pos, joint_vel, last_action), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_current_obs:
            current_obs = x[:, : self.num_actor_obs]
            frame_history = x[:, self.num_actor_obs :]
        else:
            frame_history = x
            frames = frame_history.reshape(x.shape[0], self.history_length, self.num_actor_obs)
            current_obs = frames[:, self.history_length - 1, :]
        term_major_history = self._frame_major_to_term_major(frame_history)
        return self.policy(current_obs, term_major_history)


def _latest_checkpoint(log_root: pathlib.Path) -> pathlib.Path:
    candidates = sorted(log_root.glob("*/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under {log_root}.")
    return candidates[-1]


def _load_deployment_policy(checkpoint_path: pathlib.Path, device: str) -> tuple[DreamWaQJitPolicy, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    policy_cfg = _policy_cfg_from_checkpoint(checkpoint, state_dict)
    dims = _infer_dimensions(state_dict, policy_cfg.latent_dim)

    policy = DreamWaQActorCritic(
        dims["num_actor_obs"],
        dims["num_history_obs"],
        dims["num_critic_obs"],
        dims["num_estimator_target_obs"],
        dims["num_actions"],
        **asdict(policy_cfg),
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return DreamWaQJitPolicy(policy).to(device).eval(), dims, checkpoint


def _validate_simreal_dims(dims: dict):
    history_length = dims["history_length"]
    num_actor_obs = dims["num_actor_obs"]
    history_only_dim = dims["num_history_obs"]
    if (
        num_actor_obs != SIMREAL_ACTOR_OBS_DIM
        or history_length != SIMREAL_HISTORY_LENGTH
        or history_only_dim != SIMREAL_HISTORY_ONLY_INPUT_DIM
    ):
        raise ValueError(
            "Checkpoint is not compatible with the fixed BPX sim2real upper stack: "
            f"got actor_obs={num_actor_obs}, history_length={history_length}, "
            f"history_only_input={history_only_dim}; expected actor_obs={SIMREAL_ACTOR_OBS_DIM}, "
            f"history_length={SIMREAL_HISTORY_LENGTH}, history_only_input={SIMREAL_HISTORY_ONLY_INPUT_DIM}. "
            "Retrain with the updated rough_env_cfg.py before exporting for real deployment."
        )


def _write_config_patch(path: pathlib.Path, metadata: dict):
    default_joint_pos = metadata["default_joint_pos_by_name"]
    lines = [
        "# Copy these values into bpx_simreal_v6 configs/real_config_working.yaml",
        "# under isaaclab_policy_contract before loading this policy on the SDK UI.",
        "isaaclab_policy_contract:",
        "  action_formula: q_des_sim = default_joint_pos + clip(action,-clip,clip) * action_scale",
        f"  action_scale: {metadata['action_scale_scalar']:.12g}",
        "  obs_order:",
        "    - root_ang_vel_b * 0.25",
        "    - projected_gravity_b * 1.0",
        "    - command_xy_yaw * [2.0, 2.0, 0.25]",
        "    - joint_pos_minus_default * 1.0",
        "    - joint_vel_minus_default * 0.05",
        "    - last_action * 1.0",
        f"  dwaq_history_length: {metadata['history_length']}",
        f"  actor_obs_single_frame_dim: {metadata['num_actor_obs']}",
        f"  encoder_input_dim: {metadata['num_history_obs']}",
        "  policy_input_mode: auto",
        "  supported_policy_input_dims:",
        f"    history_only: {metadata['history_only_input_dim']}",
        f"    current_plus_history: {metadata['current_plus_history_input_dim']}",
        "  command_scale:",
        "    vx: 2.0",
        "    vy: 2.0",
        "    wz: 0.25",
        "  obs_scales:",
        "    ang_vel: 0.25",
        "    projected_gravity: 1.0",
        "    lin_vel_command_x: 2.0",
        "    lin_vel_command_y: 2.0",
        "    ang_vel_command_z: 0.25",
        "    joint_pos: 1.0",
        "    joint_vel: 0.05",
        "    last_action: 1.0",
        "  default_joint_pos:",
    ]
    for name in JOINT_NAMES:
        lines.append(f"    {name}: {default_joint_pos[name]:.12g}")
    lines.extend(
        [
            "  history_layout_note: SDK/UI input is frame_major_oldest_to_newest; exported wrapper converts it to term_major_oldest_to_newest internally.",
            "  deploy_action_guard:",
            "    enabled: true",
            "    clip: 0.12",
            "    meaning: EXTRA REAL ROBOT SAFETY GUARD. Keep small for first hanging/support tests.",
            "    last_action_source_when_enabled: sent_action",
            "    last_action_source_when_disabled: sim_action",
            "    target_rate_limit_enabled: true",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _script_and_check(
    wrapper: SimRealSingleInputPolicy,
    output_path: pathlib.Path,
    input_dim: int,
    num_actions: int,
    device: str,
):
    wrapper = wrapper.to(device).eval()
    with torch.inference_mode():
        scripted = torch.jit.script(wrapper)
        x = torch.zeros(1, input_dim, device=device)
        y = scripted(x)
        if y.shape[-1] != num_actions:
            raise RuntimeError(f"Unexpected action shape from {output_path}: {tuple(y.shape)}")
    scripted.save(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Export a BPX DreamWaQ policy for the sim2real SDK UI.")
    parser.add_argument("--checkpoint", default=None, help="Path to model_*.pt. Defaults to latest under logs/waq.")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="deploy/bpx_simreal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="bpx_dwaq_v2", help="Base name for the SDK default policy file.")
    parser.add_argument("--control-profile", choices=("current", LEGACY_PROFILE_NAME), default="current")
    args = parser.parse_args()

    checkpoint_path = (
        pathlib.Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else _latest_checkpoint(_PROJECT_ROOT / "logs" / "waq" / "bpx_waq_rough").resolve()
    )
    output_dir = pathlib.Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (_PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jit_policy, dims, checkpoint = _load_deployment_policy(checkpoint_path, args.device)
    _validate_simreal_dims(dims)
    history_length = dims["history_length"]
    num_actor_obs = dims["num_actor_obs"]
    history_only_dim = dims["num_history_obs"]
    current_plus_history_dim = num_actor_obs + dims["num_history_obs"]

    history_policy_path = output_dir / f"{args.name}.pt"
    current_plus_history_path = output_dir / f"{args.name}_current_plus_history.pt"

    _script_and_check(
        SimRealSingleInputPolicy(jit_policy, num_actor_obs, history_length, include_current_obs=False),
        history_policy_path,
        history_only_dim,
        dims["num_actions"],
        args.device,
    )
    _script_and_check(
        SimRealSingleInputPolicy(jit_policy, num_actor_obs, history_length, include_current_obs=True),
        current_plus_history_path,
        current_plus_history_dim,
        dims["num_actions"],
        args.device,
    )

    control_profile = _control_profile(args.control_profile)
    action_scale = [_joint_action_scale(name, control_profile["action_scale"]) for name in JOINT_NAMES]
    metadata = {
        "checkpoint": str(checkpoint_path),
        "iteration": int(checkpoint.get("iter", -1)),
        "history_only_policy": str(history_policy_path),
        "current_plus_history_policy": str(current_plus_history_path),
        "num_actor_obs": num_actor_obs,
        "num_history_obs": dims["num_history_obs"],
        "history_length": history_length,
        "num_actions": dims["num_actions"],
        "history_only_input_dim": history_only_dim,
        "current_plus_history_input_dim": current_plus_history_dim,
        "sdk_input_layout": "frame_major_oldest_to_newest",
        "internal_history_layout": "term_major_oldest_to_newest",
        "joint_order": "type_major",
        "joint_names": JOINT_NAMES,
        "default_joint_pos": [_joint_default(name) for name in JOINT_NAMES],
        "default_joint_pos_by_name": {name: _joint_default(name) for name in JOINT_NAMES},
        "action_scale": action_scale,
        "action_scale_scalar": float(action_scale[0]),
        "action_sign": [_joint_action_sign(name) for name in JOINT_NAMES],
        "control": {
            "profile": control_profile["profile"],
            "stiffness": control_profile["stiffness"],
            "damping": control_profile["damping"],
            "effort_limit": control_profile["effort_limit"],
            "armature": control_profile["armature"],
            "natural_frequency": control_profile["natural_frequency"],
            "damping_ratio": control_profile["damping_ratio"],
            "joint_friction": 0.01,
            "policy_dt": 0.02,
            "sim_dt": 0.005,
            "decimation": 4,
        },
        "observation_order": [
            "base_ang_vel*0.25",
            "projected_gravity",
            "velocity_command*[2.0,2.0,0.25]",
            "joint_pos-default_joint_pos",
            "joint_vel*0.05",
            "last_action",
        ],
        "simreal_notes": [
            f"Use bpx_dwaq_v2.pt with SDK/UI history_only mode after setting dwaq_history_length to {history_length}.",
            "The wrapper accepts SDK/UI frame-major history and converts it to IsaacLab term-major history internally.",
            "Keep command scales in the SDK UI/config at 2.0/2.0/0.25 for this IsaacLab export.",
            "Keep deploy_action_guard.clip small for first hanging/support tests.",
        ],
    }

    metadata_path = output_dir / f"{args.name}_simreal_metadata.json"
    patch_path = output_dir / "real_config_policy_patch.yaml"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_config_patch(patch_path, metadata)

    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] exported SDK history-only policy: {history_policy_path}")
    print(f"[INFO] exported SDK current+history policy: {current_plus_history_path}")
    print(f"[INFO] wrote metadata: {metadata_path}")
    print(f"[INFO] wrote SDK config patch: {patch_path}")
    print(
        "[INFO] dims: "
        f"obs={num_actor_obs}, history={dims['num_history_obs']} ({history_length} frames), "
        f"history_only_input={history_only_dim}, current_plus_history_input={current_plus_history_dim}, "
        f"actions={dims['num_actions']}"
    )


if __name__ == "__main__":
    main()
