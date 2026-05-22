"""BPX robot configuration for IsaacLab."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_REPO_ROOT = Path(__file__).resolve().parents[5]
BPX_USD_PATH = _REPO_ROOT / "assets" / "BPX" / "usd" / "bpx.usd"
BPX_PLAY_USD_PATH = _REPO_ROOT / "assets" / "BPX" / "usd" / "bpx_play.usda"

BASE_BODY_NAME = "torso"
FEET_BODY_NAMES = ".*_toe_link"
UNDESIRED_BODY_NAMES = [".*_hip_link", ".*_thigh_link", ".*_calf_link"]
CONTROLLED_JOINT_NAMES = [".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"]

BPX_EFFORT_LIMIT = 140.0
BPX_ARMATURE = 0.005
BPX_STIFFNESS = 48.0
BPX_DAMPING = 2.2
BPX_DEFAULT_BASE_HEIGHT = 0.45
BPX_ACTION_SCALE = {
    ".*_hip_roll_joint": 0.20,
    ".*_hip_pitch_joint": 0.20,
    ".*_knee_joint": 0.20,
}

BPX_STAND_JOINT_POS = {
    "fl_hip_roll_joint": 0.10,
    "fr_hip_roll_joint": -0.10,
    "hl_hip_roll_joint": 0.10,
    "hr_hip_roll_joint": -0.10,
    "fl_hip_pitch_joint": 0.80,
    "fr_hip_pitch_joint": 0.80,
    "hl_hip_pitch_joint": 0.80,
    "hr_hip_pitch_joint": 0.80,
    "fl_knee_joint": -1.50,
    "fr_knee_joint": -1.50,
    "hl_knee_joint": -1.50,
    "hr_knee_joint": -1.50,
}

BPX_JOINT_ORDER = [
    "fl_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fl_knee_joint",
    "fr_hip_roll_joint",
    "fr_hip_pitch_joint",
    "fr_knee_joint",
    "hl_hip_roll_joint",
    "hl_hip_pitch_joint",
    "hl_knee_joint",
    "hr_hip_roll_joint",
    "hr_hip_pitch_joint",
    "hr_knee_joint",
]


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
