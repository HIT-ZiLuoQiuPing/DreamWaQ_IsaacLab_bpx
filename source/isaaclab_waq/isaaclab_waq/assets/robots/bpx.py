"""BPX robot configuration for IsaacLab."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .bpx_constants import (
    BASE_BODY_NAME,
    BPX_ACTION_SCALE,
    BPX_ARMATURE,
    BPX_DAMPING,
    BPX_DEFAULT_BASE_HEIGHT,
    BPX_EFFORT_LIMIT,
    BPX_STAND_JOINT_POS,
    BPX_STIFFNESS,
    CONTROLLED_JOINT_NAMES,
    FEET_BODY_NAMES,
    FEET_BODY_NAMES_ORDERED,
    UNDESIRED_BODY_NAMES,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
BPX_USD_PATH = _REPO_ROOT / "assets" / "BPX" / "usd" / "bpx.usd"
BPX_PLAY_USD_PATH = _REPO_ROOT / "assets" / "BPX" / "usd" / "bpx_play.usda"

def _make_bpx_cfg(usd_path: Path) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BPX_DEFAULT_BASE_HEIGHT),
            joint_pos=BPX_STAND_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=CONTROLLED_JOINT_NAMES,
                effort_limit=BPX_EFFORT_LIMIT,
                velocity_limit=40.0,
                stiffness=BPX_STIFFNESS,
                damping=BPX_DAMPING,
                friction=0.01,
                armature=BPX_ARMATURE,
            ),
        },
        soft_joint_pos_limit_factor=0.9,
    )


BPX_CFG = _make_bpx_cfg(BPX_USD_PATH)
BPX_PLAY_CFG = _make_bpx_cfg(BPX_PLAY_USD_PATH)
