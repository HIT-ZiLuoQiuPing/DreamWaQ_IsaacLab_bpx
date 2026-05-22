"""Curriculum helpers for BPX rough-terrain walking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    promotion_distance_ratio: float = 0.75,
    demotion_command_ratio: float = 0.5,
) -> torch.Tensor:
    """Promote terrain level only for episodes that survive to timeout."""

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command(command_name)
    terrain_generator = terrain.cfg.terrain_generator
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    promotion_distance = terrain_generator.size[0] * promotion_distance_ratio

    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    move_up = (distance > promotion_distance) & timed_out.bool()
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * demotion_command_ratio
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def command_vel_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    velocity_stages: list[dict],
) -> torch.Tensor:
    """Set command ranges from global-step stages."""

    del env_ids
    command_term = env.command_manager.get_term(command_name)
    selected = velocity_stages[0]
    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            selected = stage
        else:
            break

    ranges = command_term.cfg.ranges
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if name in selected:
            setattr(ranges, name, selected[name])
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Expand linear velocity command ranges when tracking reward is high."""

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_term.weight * 0.8:
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)
        ranges.lin_vel_x = torch.clamp(
            torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
            limit_ranges.lin_vel_x[0],
            limit_ranges.lin_vel_x[1],
        ).tolist()
        ranges.lin_vel_y = torch.clamp(
            torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
            limit_ranges.lin_vel_y[0],
            limit_ranges.lin_vel_y[1],
        ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """Expand yaw-rate command ranges when tracking reward is high."""

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0 and reward > reward_term.weight * 0.8:
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)
        ranges.ang_vel_z = torch.clamp(
            torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
            limit_ranges.ang_vel_z[0],
            limit_ranges.ang_vel_z[1],
        ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
