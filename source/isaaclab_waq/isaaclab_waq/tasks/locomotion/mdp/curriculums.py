"""Curriculum helpers for BPX rough-terrain walking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _curriculum_step_counter(env: ManagerBasedRLEnv) -> int:
    """Return global curriculum steps, including resume offset from a checkpoint."""

    return int(getattr(env, "common_step_counter", 0)) + int(getattr(env, "_waq_curriculum_step_offset", 0))


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    promotion_distance_ratio: float = 0.85,
    promotion_command_ratio: float | None = None,
    demotion_command_ratio: float = 0.25,
    minimum_promotion_distance: float = 0.0,
    warmup_steps: int = 0,
    level_step_interval: int = 4096,
    consecutive_successes: int = 2,
    demote_only_early_termination: bool = True,
    min_level_hold_steps: int = 0,
) -> torch.Tensor:
    """Promote terrain level conservatively after repeated successful timeouts."""

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command(command_name)
    terrain_generator = terrain.cfg.terrain_generator
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    promotion_distance = terrain_generator.size[0] * promotion_distance_ratio

    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    command_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    distance_success = distance > promotion_distance
    if promotion_command_ratio is None:
        command_success = torch.zeros_like(distance_success)
    else:
        command_target = torch.clamp(command_distance * promotion_command_ratio, min=minimum_promotion_distance)
        command_success = distance > command_target
    successful = (distance_success | command_success) & timed_out.bool()

    if not hasattr(env, "_waq_terrain_success_streak"):
        env._waq_terrain_success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_waq_terrain_last_change_step"):
        env._waq_terrain_last_change_step = torch.full(
            (env.num_envs,), -min_level_hold_steps, dtype=torch.long, device=env.device
        )

    streak = env._waq_terrain_success_streak
    streak[env_ids] = torch.where(successful, streak[env_ids] + 1, torch.zeros_like(streak[env_ids]))

    step_counter = _curriculum_step_counter(env)
    if level_step_interval <= 0:
        allowed_max_level = max(getattr(terrain_generator, "num_rows", 1) - 1, 0)
    elif step_counter < warmup_steps:
        allowed_max_level = 0
    else:
        step_interval = max(level_step_interval, 1)
        allowed_max_level = 1 + (step_counter - warmup_steps) // step_interval
    max_terrain_level = max(getattr(terrain_generator, "num_rows", 1) - 1, 0)
    allowed_max_level = int(max(0, min(allowed_max_level, max_terrain_level)))

    current_levels = terrain.terrain_levels[env_ids]
    can_change = (
        step_counter - env._waq_terrain_last_change_step[env_ids]
    ) >= min_level_hold_steps
    move_up = (
        (streak[env_ids] >= consecutive_successes)
        & (current_levels < allowed_max_level)
        & can_change
    )

    move_down = distance < command_distance * demotion_command_ratio
    if demote_only_early_termination:
        move_down &= ~timed_out.bool()
    move_down &= current_levels > 0
    move_down &= ~move_up
    move_down &= can_change

    terrain.update_env_origins(env_ids, move_up, move_down)
    changed_ids = env_ids[move_up | move_down]
    if changed_ids.numel() > 0:
        env._waq_terrain_last_change_step[changed_ids] = step_counter
        streak[changed_ids] = 0

    env._waq_terrain_curriculum_stats = {
        "allowed_max_level": torch.tensor(float(allowed_max_level), device=env.device),
        "promotion_distance": torch.tensor(float(promotion_distance), device=env.device),
        "mean_distance": distance.detach().float().mean(),
        "mean_command_distance": command_distance.detach().float().mean(),
        "distance_success_rate": distance_success.detach().float().mean(),
        "command_success_rate": command_success.detach().float().mean(),
        "success_rate": successful.detach().float().mean(),
        "move_up_rate": move_up.detach().float().mean(),
        "move_down_rate": move_down.detach().float().mean(),
        "success_streak_mean": streak[env_ids].detach().float().mean(),
    }
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
    step_counter = _curriculum_step_counter(env)
    for stage in velocity_stages:
        if step_counter >= stage["step"]:
            selected = stage
        else:
            break

    ranges = command_term.cfg.ranges
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        if name in selected:
            setattr(ranges, name, selected[name])
    forward_ranges = getattr(command_term.cfg, "forward_ranges", None)
    if forward_ranges is not None:
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            stage_name = f"forward_{name}"
            if stage_name in selected:
                setattr(forward_ranges, name, selected[stage_name])
            elif name == "lin_vel_x" and name in selected:
                low, high = selected[name]
                setattr(forward_ranges, name, (max(0.0, low), high))
            elif name in selected:
                setattr(forward_ranges, name, selected[name])
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
