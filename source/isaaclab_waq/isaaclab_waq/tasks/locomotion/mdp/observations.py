"""BPX observation helpers that match the robot_rl root-link convention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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
