"""BPX constants shared by IsaacLab assets and deployment export."""

BASE_BODY_NAME = "torso"
FEET_BODY_NAMES = ".*_toe_link"
FEET_BODY_NAMES_ORDERED = ["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"]
UNDESIRED_BODY_NAMES = [".*_hip_link", ".*_thigh_link", ".*_calf_link"]
CONTROLLED_JOINT_NAMES = [".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"]

BPX_EFFORT_LIMIT = 30.0
BPX_ARMATURE = 0.005
BPX_NATURAL_FREQUENCY = 28.0 * 2.0 * 3.1415926535
BPX_DAMPING_RATIO = 2.5
BPX_STIFFNESS = BPX_ARMATURE * BPX_NATURAL_FREQUENCY**2
BPX_DAMPING = 2.0 * BPX_DAMPING_RATIO * BPX_ARMATURE * BPX_NATURAL_FREQUENCY
BPX_DEFAULT_BASE_HEIGHT = 0.42
BPX_DEFAULT_ACTION_SCALE = 0.25 * BPX_EFFORT_LIMIT / BPX_STIFFNESS
BPX_ACTION_SCALE = {
    ".*_hip_roll_joint": BPX_DEFAULT_ACTION_SCALE,
    ".*_hip_pitch_joint": BPX_DEFAULT_ACTION_SCALE,
    ".*_knee_joint": BPX_DEFAULT_ACTION_SCALE,
}

BPX_STAND_JOINT_POS = {
    ".*_hip_roll_joint": 0.0,
    ".*_hip_pitch_joint": 0.60,
    ".*_knee_joint": -1.20,
}
