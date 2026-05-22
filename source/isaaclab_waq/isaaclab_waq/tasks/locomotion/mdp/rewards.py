"""BPX-specific reward helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab_waq.assets.robots.bpx import BPX_JOINT_ORDER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute joint power."""

    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the base z-axis aligned with gravity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt_error / std**2)


def root_height_below_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root-link height above scanned terrain is too low."""

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    fallback_height = asset.data.root_link_pos_w[:, 2].unsqueeze(1) - minimum_height
    hits_z = torch.where(torch.isfinite(hits_z), hits_z, fallback_height)
    terrain_height = torch.mean(hits_z, dim=1)
    relative_height = asset.data.root_link_pos_w[:, 2] - terrain_height
    return relative_height < minimum_height


def base_height_above_terrain_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root-link height error relative to scanned terrain."""

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    fallback_height = asset.data.root_link_pos_w[:, 2].unsqueeze(1) - target_height
    hits_z = torch.where(torch.isfinite(hits_z), hits_z, fallback_height)
    terrain_height = torch.mean(hits_z, dim=1)
    relative_height = asset.data.root_link_pos_w[:, 2] - terrain_height
    return torch.square(relative_height - target_height)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize vertical root-link velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize roll/pitch root-link angular velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking xy root-link velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking yaw root-link angular velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(error) / std**2)


def track_forward_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the commanded forward velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, 0] - asset.data.root_link_lin_vel_b[:, 0]
    return torch.exp(-torch.square(error) / std**2)


def forward_velocity_error_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute forward velocity tracking error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])


def no_forward_motion(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_command: float = 0.15,
    min_velocity_ratio: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when a forward command is active."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    min_velocity = torch.clamp(command[:, 0] * min_velocity_ratio, min=0.0)
    stalled = (command[:, 0] > min_command) & (asset.data.root_link_lin_vel_b[:, 0] < min_velocity)
    return stalled.float()


def track_lateral_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the commanded lateral velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, 1] - asset.data.root_link_lin_vel_b[:, 1]
    return torch.exp(-torch.square(error) / std**2)


def track_yaw_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the commanded yaw velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(error) / std**2)


def forward_lateral_drift(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_forward_command: float = 0.2,
    lateral_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize side drift when commanded to walk mostly forward."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    straight = (
        (command[:, 0] > min_forward_command)
        & (torch.abs(command[:, 1]) < lateral_command_threshold)
        & (torch.abs(command[:, 2]) < yaw_command_threshold)
    )
    return torch.square(asset.data.root_link_lin_vel_b[:, 1]) * straight.float()


def forward_yaw_drift(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_forward_command: float = 0.2,
    lateral_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize uncommanded yaw rate when commanded to walk mostly forward."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    straight = (
        (command[:, 0] > min_forward_command)
        & (torch.abs(command[:, 1]) < lateral_command_threshold)
        & (torch.abs(command[:, 2]) < yaw_command_threshold)
    )
    return torch.square(asset.data.root_link_ang_vel_b[:, 2]) * straight.float()


def leg_symmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_forward_command: float = 0.2,
    lateral_command_threshold: float = 0.08,
    yaw_command_threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage left/right leg symmetry while walking straight."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    straight = (
        (command[:, 0] > min_forward_command)
        & (torch.abs(command[:, 1]) < lateral_command_threshold)
        & (torch.abs(command[:, 2]) < yaw_command_threshold)
    )
    try:
        ids = [asset.joint_names.index(name) for name in BPX_JOINT_ORDER]
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device)

    q = asset.data.joint_pos[:, ids]
    front_roll = torch.square(q[:, 0] + q[:, 3])
    hind_roll = torch.square(q[:, 6] + q[:, 9])
    front_pitch = torch.square(q[:, 1] - q[:, 4])
    hind_pitch = torch.square(q[:, 7] - q[:, 10])
    front_knee = torch.square(q[:, 2] - q[:, 5])
    hind_knee = torch.square(q[:, 8] - q[:, 11])
    return (front_roll + hind_roll + front_pitch + hind_pitch + front_knee + hind_knee) * straight.float()


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint offset from the default pose, more strongly while standing."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    penalty = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), penalty, stand_still_scale * penalty)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > 4.0 * forces_z, dim=1).float()


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize uneven foot air/contact timing."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("ContactSensor.track_air_time must be enabled for air_time_variance_penalty.")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward swing feet for clearing the target height."""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_velocity_tanh * torch.exp(-foot_z_target_error / std)
    reward = torch.mean(reward, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Penalize swing-foot height error in the base frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    foot_vel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for index in range(len(asset_cfg.body_ids)):
        foot_pos_b[:, index, :] = quat_apply_inverse(asset.data.root_quat_w, foot_pos_translated[:, index, :])
        foot_vel_b[:, index, :] = quat_apply_inverse(asset.data.root_quat_w, foot_vel_translated[:, index, :])

    foot_z_target_error = torch.square(foot_pos_b[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_b[:, :, :2], dim=2))
    penalty = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return penalty
