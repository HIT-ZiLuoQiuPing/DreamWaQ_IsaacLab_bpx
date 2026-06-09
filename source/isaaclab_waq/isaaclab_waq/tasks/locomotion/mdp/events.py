"""BPX-specific event helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_like(
    tensor: torch.Tensor,
    distribution_params: tuple[float, float],
    distribution: Literal["uniform", "log_uniform"],
) -> torch.Tensor:
    low, high = distribution_params
    if distribution == "uniform":
        return torch.empty_like(tensor).uniform_(low, high)
    if distribution == "log_uniform":
        return torch.empty_like(tensor).uniform_(math.log(low), math.log(high)).exp()
    raise ValueError(f"Unsupported distribution: {distribution}")


def _apply_randomization(
    values: torch.Tensor,
    samples: torch.Tensor,
    operation: Literal["scale", "abs", "add"],
) -> torch.Tensor:
    if operation == "scale":
        return values * samples
    if operation == "abs":
        return samples
    if operation == "add":
        return values + samples
    raise ValueError(f"Unsupported operation: {operation}")


def randomize_actuator_effort_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    effort_limit_distribution_params: tuple[float, float],
    operation: Literal["scale", "abs", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform"] = "uniform",
):
    """Randomize explicit actuator torque clipping limits.

    IsaacLab provides built-in events for gains, armature, friction, and mass.
    The explicit BPX IdealPD actuator also clips the computed torque in its own
    actuator model; this event randomizes that clipping limit so policies cannot
    rely on a single simulator-side torque envelope.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_ids = slice(None)
        elif isinstance(actuator.joint_indices, slice):
            actuator_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)
        else:
            actuator_joint_ids = torch.as_tensor(actuator.joint_indices, device=asset.device, dtype=torch.long)
            asset_joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)
            actuator_ids = torch.nonzero(torch.isin(actuator_joint_ids, asset_joint_ids), as_tuple=False).view(-1)
            if actuator_ids.numel() == 0:
                continue

        effort_limit = actuator.effort_limit[env_ids].clone()
        selected = effort_limit[:, actuator_ids]
        samples = _sample_like(selected, effort_limit_distribution_params, distribution)
        effort_limit[:, actuator_ids] = _apply_randomization(selected, samples, operation)
        actuator.effort_limit[env_ids] = torch.clamp(effort_limit, min=1.0)
