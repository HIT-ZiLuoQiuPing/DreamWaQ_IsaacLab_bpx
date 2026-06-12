"""BPX-specific reward helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 惩罚关节速度和关节力矩共同带来的功率消耗，鼓励更省力的动作。
def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute joint power."""

    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


# 鼓励机身保持竖直，不要侧翻或前后翻。
def upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the base z-axis aligned with gravity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b 是机体坐标系下的重力方向；机器人越直立，x/y 分量越小。
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt_error / std**2)


# 惩罚机身高度相对于目标高度的误差。这个奖励鼓励机器人保持在一个合适的高度，既不太低（可能会碰到地面），也不太高（可能会失去稳定性）。
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


# 只惩罚机身低于目标高度的部分，避免机器人塌低贴地，但不过度限制抬高动作。
def base_height_below_target_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only root-link height below the target terrain-relative height."""

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    fallback_height = asset.data.root_link_pos_w[:, 2].unsqueeze(1) - target_height
    hits_z = torch.where(torch.isfinite(hits_z), hits_z, fallback_height)
    terrain_height = torch.mean(hits_z, dim=1)
    relative_height = asset.data.root_link_pos_w[:, 2] - terrain_height
    return torch.clamp(target_height - relative_height, min=0.0)


# 惩罚机身竖直方向速度，减少上下弹跳。
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize vertical root-link velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


# 惩罚机身 roll/pitch 角速度，减少左右晃动和前后俯仰。
def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize roll/pitch root-link angular velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


# 平面速度跟踪奖励：机身 xy 线速度越接近指令，奖励越高。
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


# 偏航角速度跟踪奖励：机身 yaw 角速度越接近指令，奖励越高。
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


# 前进方向 x 速度精细跟踪奖励：std 更小，因此比整体 xy 跟踪更严格。
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


# 侧向 y 速度精细跟踪奖励：侧向速度越接近指令，奖励越高。
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


# 偏航速度精细跟踪奖励：对 yaw 指令误差给更严格的指数奖励。
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


# 惩罚在前进命令下的侧向漂移。这个奖励鼓励机器人在有明显前进命令时，保持在前进方向上，而不是有过多的侧向移动。
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


# 惩罚前进命令下没有被指令要求的偏航角速度，避免边走边乱转。
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


# 惩罚关节偏离默认站姿；静止或低速时可通过 stand_still_scale 加强站姿约束。
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
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        penalty,
        stand_still_scale * penalty,
    )


# 惩罚足端横向撞到障碍或台阶侧壁，降低绊脚风险。
def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > 4.0 * forces_z, dim=1).float()


# 内部辅助：根据足端 xy 位置匹配最近的高度扫描点，估计足端相对地形的高度。
def _foot_heights_above_terrain(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
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


# 惩罚移动足端相对地形的离地高度偏离目标值，鼓励抬脚越障但不过度抬高。
def foot_clearance_terrain_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """Penalize moving feet that are not near the target height above terrain."""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_heights = _foot_heights_above_terrain(env, asset_cfg, sensor_cfg)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    penalty = torch.mean(torch.square(foot_heights - target_height) * foot_velocity_tanh, dim=1)
    command_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return penalty * (command_norm > command_threshold).float()


# 只对空中摆动足施加相对地形高度惩罚，让摆动腿贴近目标离地高度。
def feet_swing_height_terrain_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize airborne feet whose height above terrain misses the target."""

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, contact_sensor_cfg.body_ids] > 0.0
    foot_heights = _foot_heights_above_terrain(env, asset_cfg, sensor_cfg)
    penalty = torch.mean(torch.square(foot_heights - target_height) * in_air.float(), dim=1)
    command_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return penalty * (command_norm > command_threshold).float()
