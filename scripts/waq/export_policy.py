"""Export a DreamWaQ checkpoint to a TorchScript inference policy."""

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

from isaaclab_waq.algorithms.waq.actor_critic import DreamWaQActorCritic
from isaaclab_waq.algorithms.waq.config import DreamWaQPolicyCfg
from isaaclab_waq.assets.robots.bpx_constants import (
    BPX_ACTION_SCALE,
    BPX_ARMATURE,
    BPX_DAMPING,
    BPX_DAMPING_RATIO,
    BPX_DEFAULT_BASE_HEIGHT,
    BPX_EFFORT_LIMIT,
    BPX_NATURAL_FREQUENCY,
    BPX_STAND_JOINT_POS,
    BPX_STIFFNESS,
)


JOINT_NAMES = [
    "fl_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fl_knee_joint",
    "fr_hip_roll_joint",
    "fr_hip_pitch_joint",
    "fr_knee_joint",
    "hl_hip_roll_joint",
    "hl_hip_pitch_joint",
    "hl_knee_joint",
    "hr_hip_roll_joint",
    "hr_hip_pitch_joint",
    "hr_knee_joint",
]


class DreamWaQJitPolicy(nn.Module):
    """Deployment wrapper with deterministic DreamWaQ inference only."""

    def __init__(self, policy: DreamWaQActorCritic):
        super().__init__()
        self.actor_obs_normalizer = policy.actor_obs_normalizer
        self.history_obs_normalizer = policy.history_obs_normalizer
        self.encoder = policy.encoder
        self.velocity_head = policy.velocity_head
        self.latent_mean = policy.latent_mean
        self.actor = policy.actor

    def forward(self, observations: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        observations = self.actor_obs_normalizer(observations)
        history = self.history_obs_normalizer(history)
        encoded = self.encoder(history)
        velocity = self.velocity_head(encoded)
        latent = self.latent_mean(encoded)
        return self.actor(torch.cat((observations, velocity, latent), dim=-1))


def _linear_dims(state_dict: dict[str, torch.Tensor], prefix: str) -> tuple[int, list[int], int]:
    weights: list[tuple[int, torch.Tensor]] = []
    prefix_dot = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(prefix_dot) and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                weights.append((int(parts[1]), value))
    if not weights:
        raise KeyError(f"Could not infer linear layer dimensions for '{prefix}'.")
    weights.sort(key=lambda item: item[0])
    input_dim = int(weights[0][1].shape[1])
    output_dims = [int(weight.shape[0]) for _, weight in weights]
    return input_dim, output_dims[:-1], output_dims[-1]


def _policy_cfg_from_checkpoint(checkpoint: dict, state_dict: dict[str, torch.Tensor]) -> DreamWaQPolicyCfg:
    cfg = DreamWaQPolicyCfg()
    cfg_dict = checkpoint.get("cfg")
    if isinstance(cfg_dict, dict) and isinstance(cfg_dict.get("policy"), dict):
        for key, value in cfg_dict["policy"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    _, actor_hidden_dims, _ = _linear_dims(state_dict, "actor")
    _, critic_hidden_dims, _ = _linear_dims(state_dict, "critic")
    _, encoder_hidden_dims, encoder_output_dim = _linear_dims(state_dict, "encoder")
    _, decoder_hidden_dims, _ = _linear_dims(state_dict, "decoder")
    cfg.actor_hidden_dims = actor_hidden_dims
    cfg.critic_hidden_dims = critic_hidden_dims
    cfg.encoder_hidden_dims = [*encoder_hidden_dims, encoder_output_dim]
    cfg.decoder_hidden_dims = decoder_hidden_dims
    cfg.latent_dim = int(state_dict["latent_mean.weight"].shape[0])
    return cfg


def _infer_dimensions(state_dict: dict[str, torch.Tensor], latent_dim: int) -> dict[str, int]:
    actor_input_dim, _, num_actions = _linear_dims(state_dict, "actor")
    history_obs_dim, _, _ = _linear_dims(state_dict, "encoder")
    critic_obs_dim, _, _ = _linear_dims(state_dict, "critic")
    _, _, terrain_dim = _linear_dims(state_dict, "decoder")
    num_actor_obs = actor_input_dim - (3 + latent_dim)
    if num_actor_obs <= 0:
        raise ValueError(f"Invalid inferred actor observation dimension: {num_actor_obs}.")
    if history_obs_dim % num_actor_obs != 0:
        raise ValueError(
            f"History dimension {history_obs_dim} is not divisible by actor observation dimension {num_actor_obs}."
        )
    return {
        "num_actor_obs": num_actor_obs,
        "num_history_obs": history_obs_dim,
        "num_critic_obs": critic_obs_dim,
        "num_estimator_target_obs": terrain_dim + 3,
        "num_actions": num_actions,
        "history_length": history_obs_dim // num_actor_obs,
    }


def _joint_default(name: str) -> float:
    if name.endswith("_hip_roll_joint"):
        return float(BPX_STAND_JOINT_POS[".*_hip_roll_joint"])
    if name.endswith("_hip_pitch_joint"):
        return float(BPX_STAND_JOINT_POS[".*_hip_pitch_joint"])
    if name.endswith("_knee_joint"):
        return float(BPX_STAND_JOINT_POS[".*_knee_joint"])
    raise KeyError(name)


def _joint_action_scale(name: str) -> float:
    if name.endswith("_hip_roll_joint"):
        return float(BPX_ACTION_SCALE[".*_hip_roll_joint"])
    if name.endswith("_hip_pitch_joint"):
        return float(BPX_ACTION_SCALE[".*_hip_pitch_joint"])
    if name.endswith("_knee_joint"):
        return float(BPX_ACTION_SCALE[".*_knee_joint"])
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser(description="Export a DreamWaQ checkpoint for sim2sim deployment.")
    parser.add_argument("--checkpoint", required=True, help="Path to a DreamWaQ model_*.pt checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output TorchScript path. Defaults to <checkpoint_dir>/policy_jit.pt.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device used for export.")
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve() if args.output else checkpoint_path.parent / "policy_jit.pt"
    metadata_path = output_path.with_suffix(".json")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
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
    ).to(args.device)
    policy.load_state_dict(state_dict)
    policy.eval()

    jit_policy = DreamWaQJitPolicy(policy).to(args.device).eval()
    with torch.inference_mode():
        observations = torch.zeros(1, dims["num_actor_obs"], device=args.device)
        history = torch.zeros(1, dims["num_history_obs"], device=args.device)
        scripted = torch.jit.script(jit_policy)
        actions = scripted(observations, history)
        if actions.shape[-1] != dims["num_actions"]:
            raise RuntimeError(f"Unexpected action shape from exported policy: {tuple(actions.shape)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    metadata = {
        "checkpoint": str(checkpoint_path),
        "policy": str(output_path),
        "iteration": int(checkpoint.get("iter", -1)),
        "num_actor_obs": dims["num_actor_obs"],
        "num_history_obs": dims["num_history_obs"],
        "history_length": dims["history_length"],
        "num_actions": dims["num_actions"],
        "joint_names": JOINT_NAMES,
        "default_joint_pos": [_joint_default(name) for name in JOINT_NAMES],
        "action_scale": [_joint_action_scale(name) for name in JOINT_NAMES],
        "base_height": float(BPX_DEFAULT_BASE_HEIGHT),
        "control": {
            "stiffness": float(BPX_STIFFNESS),
            "damping": float(BPX_DAMPING),
            "effort_limit": float(BPX_EFFORT_LIMIT),
            "armature": float(BPX_ARMATURE),
            "natural_frequency": float(BPX_NATURAL_FREQUENCY),
            "damping_ratio": float(BPX_DAMPING_RATIO),
            "joint_friction": 0.01,
            "policy_dt": 0.02,
            "sim_dt": 0.005,
            "decimation": 4,
        },
        "observation_order": [
            "base_ang_vel*0.2",
            "projected_gravity",
            "velocity_command",
            "joint_pos-default_joint_pos",
            "joint_vel*0.05",
            "last_action",
        ],
        "history_layout": "term_major_oldest_to_newest",
        "notes": (
            "IsaacLab flattens history per observation term, then concatenates terms: "
            "[term0_oldest..newest, term1_oldest..newest, ...]."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[INFO] Exported TorchScript policy: {output_path}")
    print(f"[INFO] Wrote sim2sim metadata: {metadata_path}")
    print(
        "[INFO] Dims: "
        f"actor={dims['num_actor_obs']}, history={dims['num_history_obs']} "
        f"({dims['history_length']} frames), actions={dims['num_actions']}"
    )
    print(
        "[INFO] Control: "
        f"kp={BPX_STIFFNESS:.4f}, kd={BPX_DAMPING:.4f}, "
        f"effort={BPX_EFFORT_LIMIT:.2f}, armature={BPX_ARMATURE:.4f}, "
        f"joint_friction=0.0100, action_scale={next(iter(BPX_ACTION_SCALE.values())):.4f}"
    )


if __name__ == "__main__":
    main()
