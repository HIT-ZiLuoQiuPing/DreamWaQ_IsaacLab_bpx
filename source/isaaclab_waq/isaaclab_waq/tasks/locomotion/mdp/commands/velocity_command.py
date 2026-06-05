"""Velocity command configuration with explicit curriculum limits."""

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with ranges that curricula may expand toward."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class ForwardBiasedVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with a subset biased toward forward walking."""

    cfg: "ForwardBiasedVelocityCommandCfg"

    def __init__(self, cfg: "ForwardBiasedVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.is_forward_env = torch.zeros_like(self.is_standing_env)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tForward probability: {self.cfg.rel_forward_envs}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.is_forward_env[env_ids] = False

        forward_mask = torch.rand(len(env_ids), device=self.device) <= self.cfg.rel_forward_envs
        forward_mask &= ~self.is_standing_env[env_ids]
        self.is_forward_env[env_ids] = forward_mask

        forward_env_ids = env_ids[forward_mask]
        if forward_env_ids.numel() == 0:
            return

        num_forward = forward_env_ids.numel()
        self.vel_command_b[forward_env_ids, 0] = torch.empty(num_forward, device=self.device).uniform_(
            *self.cfg.forward_ranges.lin_vel_x
        )
        self.vel_command_b[forward_env_ids, 1] = torch.empty(num_forward, device=self.device).uniform_(
            *self.cfg.forward_ranges.lin_vel_y
        )
        self.vel_command_b[forward_env_ids, 2] = torch.empty(num_forward, device=self.device).uniform_(
            *self.cfg.forward_ranges.ang_vel_z
        )


@configclass
class ForwardBiasedVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Velocity command with explicit forward-walking samples."""

    class_type: type = ForwardBiasedVelocityCommand

    rel_forward_envs: float = 0.75
    """Probability of non-standing environments that sample from forward_ranges."""

    forward_ranges: UniformVelocityCommandCfg.Ranges = MISSING
