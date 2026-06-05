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


def crawl_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_command: float = 0.25,
    min_velocity_ratio: float = 0.45,
    action_threshold: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize high-action crawling when a forward command is not being tracked."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    action = getattr(env.action_manager, "action", None)
    if action is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    command_x = command[:, 0]
    velocity_x = asset.data.root_link_lin_vel_b[:, 0]
    velocity_shortfall = torch.clamp(command_x * min_velocity_ratio - velocity_x, min=0.0)
    action_excess = torch.clamp(torch.mean(torch.abs(action), dim=1) - action_threshold, min=0.0)
    active = command_x > min_command
    return velocity_shortfall * action_excess * active.float()


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


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward feet matching a simple periodic contact schedule."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + offset_) % 1.0 for offset_ in offset], dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for index in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, index] < threshold
        reward += (~(is_stance ^ is_contact[:, index])).float()

    if command_name is not None:
        command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        reward *= command_norm > 0.1
    return reward


def all_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize bounding/hopping where every foot is airborne at once."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return (~torch.any(is_contact, dim=1)).float() * (command_norm > 0.1).float()


def feet_contact_count_error(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    moving_contact_count: float = 2.0,
    standing_contact_count: float = 4.0,
) -> torch.Tensor:
    """Penalize contact patterns far from two-leg trot support while moving."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    contact_count = torch.sum(is_contact.float(), dim=1)
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    target_count = torch.where(
        command_norm > 0.1,
        torch.full_like(contact_count, moving_contact_count),
        torch.full_like(contact_count, standing_contact_count),
    )
    return torch.square(contact_count - target_count)


def diagonal_trot_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward phase-free diagonal two-foot support while moving.

    The foot order is expected to be FL, FR, HL, HR. This avoids tying the policy
    to an unobserved episode-time clock while still preferring trot-like contacts.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if is_contact.shape[1] != 4:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    fl, fr, hl, hr = [is_contact[:, index] for index in range(4)]
    diagonal_support = torch.logical_or(fl & hr & ~fr & ~hl, fr & hl & ~fl & ~hr)
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return diagonal_support.float() * (command_norm > 0.1).float()


def bad_two_foot_contact_pattern(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize two-foot support patterns that look like bounding or pacing."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if is_contact.shape[1] != 4:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    fl, fr, hl, hr = [is_contact[:, index] for index in range(4)]
    contact_count = torch.sum(is_contact.float(), dim=1)
    two_feet = contact_count == 2.0
    front_pair = fl & fr
    hind_pair = hl & hr
    left_pair = fl & hl
    right_pair = fr & hr
    bad_pair = front_pair | hind_pair | left_pair | right_pair
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return (two_feet & bad_pair).float() * (command_norm > 0.1).float()


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
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    penalty = torch.mean(torch.square(foot_heights - target_height) * foot_velocity_tanh, dim=1)
    command_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return penalty * (command_norm > command_threshold).float()


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
