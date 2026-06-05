"""BPX observation helpers that match the robot_rl root-link convention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_link_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root link linear velocity in the root link frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def base_link_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root link angular velocity in the root link frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def foot_height_body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot heights relative to the root body height."""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    root_z = asset.data.root_pos_w[:, 2].unsqueeze(-1)
    return foot_z - root_z


def foot_height_scan(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot heights above the nearest scanned terrain point."""

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    ray_hits = sensor.data.ray_hits_w
    hits_z = ray_hits[..., 2]
    fallback_z = asset.data.root_pos_w[:, 2].unsqueeze(-1) - 0.42
    valid_hits = torch.isfinite(hits_z)
    hits_z = torch.where(valid_hits, hits_z, fallback_z)

    foot_xy = foot_pos[..., :2]
    ray_xy = ray_hits[..., :2]
    distances = torch.sum(torch.square(foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)), dim=-1)
    distances = torch.where(valid_hits.unsqueeze(1), distances, torch.full_like(distances, float("inf")))
    nearest_ids = torch.argmin(distances, dim=2)
    terrain_z = torch.gather(hits_z, 1, nearest_ids)
    return foot_pos[..., 2] - terrain_z


def foot_air_time(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current air time for each foot."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]


def foot_contact(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Binary foot contact state."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()


def foot_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact forces for each foot in world frame."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].flatten(start_dim=1)
