"""BPX rough-terrain velocity tracking environment for DreamWaQ."""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_waq.assets.robots.bpx import (
    BASE_BODY_NAME,
    BPX_ACTION_SCALE,
    BPX_CFG,
    BPX_PLAY_CFG,
    CONTROLLED_JOINT_NAMES,
    FEET_BODY_NAMES_ORDERED,
)
from isaaclab_waq.tasks.locomotion import mdp


HIND_FEET_BODY_NAMES = ["hl_toe_link", "hr_toe_link"]
HIND_CALF_BODY_NAMES = ["hl_calf_link", "hr_calf_link"]
HIND_HIP_ROLL_JOINT_NAMES = ["hl_hip_roll_joint", "hr_hip_roll_joint"]
HIND_LEG_JOINT_NAMES = [
    "hl_hip_roll_joint",
    "hl_hip_pitch_joint",
    "hl_knee_joint",
    "hr_hip_roll_joint",
    "hr_hip_pitch_joint",
    "hr_knee_joint",
]


BPX_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.12),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.08,
            noise_range=(0.005, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.0, 0.85),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.85),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.16),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.08,
            step_height_range=(0.04, 0.16),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.07,
            amplitude_range=(0.0, 0.12),
            num_waves=2,
            border_width=0.25,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Rough-terrain scene with BPX."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BPX_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = BPX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY_NAME}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.2, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.ForwardBiasedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.05,
        rel_forward_envs=0.80,
        rel_heading_envs=0.3,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.ForwardBiasedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.10, 0.65),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
            heading=(-math.pi, math.pi),
        ),
        forward_ranges=mdp.ForwardBiasedVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.65),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.ForwardBiasedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.80, 3.00),
            lin_vel_y=(-0.35, 0.35),
            ang_vel_z=(-0.75, 0.75),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale=BPX_ACTION_SCALE,
        use_default_offset=True,
        clip={
            ".*_hip_roll_joint": (-100.0, 100.0),
            ".*_hip_pitch_joint": (-100.0, 100.0),
            ".*_knee_joint": (-100.0, 100.0),
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_link_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CENetCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 15
            self.flatten_history_dim = True

    cenet: CENetCfg = CENetCfg()

    @configclass
    class EstimatorCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_link_lin_vel, clip=(-100, 100))
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
            scale=0.2,
        )
        foot_height = ObsTerm(
            func=mdp.foot_height_scan,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES_ORDERED),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-1.0, 1.0),
        )
        foot_air_time = ObsTerm(
            func=mdp.foot_air_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
            clip=(0.0, 1.0),
        )
        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
            clip=(0.0, 1.0),
        )
        foot_contact_forces = ObsTerm(
            func=mdp.foot_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
            clip=(-100.0, 100.0),
            scale=0.01,
        )

        def __post_init__(self):
            self.concatenate_terms = True

    estimator: EstimatorCfg = EstimatorCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_link_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_link_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
            scale=0.2,
        )
        foot_height = ObsTerm(
            func=mdp.foot_height_scan,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES_ORDERED),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-1.0, 1.0),
        )
        foot_air_time = ObsTerm(
            func=mdp.foot_air_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
            clip=(0.0, 1.0),
        )
        foot_contact = ObsTerm(
            func=mdp.foot_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
            clip=(0.0, 1.0),
        )
        foot_contact_forces = ObsTerm(
            func=mdp.foot_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
            clip=(-100.0, 100.0),
            scale=0.01,
        )

        def __post_init__(self):
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.4),
            "dynamic_friction_range": (0.3, 1.4),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "stiffness_distribution_params": (0.85, 1.20),
            "damping_distribution_params": (0.85, 1.30),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "friction_distribution_params": (0.75, 1.50),
            "armature_distribution_params": (0.85, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    actuator_effort_limit = EventTerm(
        func=mdp.randomize_actuator_effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "effort_limit_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    push_robot = None


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.05)
    upright = RewTerm(func=mdp.upright_exp, weight=0.60, params={"std": 0.45})

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.2,
        params={"command_name": "base_velocity", "std": 0.40},
    )
    track_forward_velocity_fine = RewTerm(
        func=mdp.track_forward_velocity_exp,
        weight=1.8,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    forward_velocity_error = RewTerm(
        func=mdp.forward_velocity_error_l1,
        weight=-0.8,
        params={"command_name": "base_velocity"},
    )
    no_forward_motion = RewTerm(
        func=mdp.no_forward_motion,
        weight=-2.0,
        params={"command_name": "base_velocity", "min_command": 0.20, "min_velocity_ratio": 0.35},
    )
    crawl_penalty = RewTerm(
        func=mdp.crawl_penalty,
        weight=-0.8,
        params={
            "command_name": "base_velocity",
            "min_command": 0.25,
            "min_velocity_ratio": 0.45,
            "action_threshold": 0.75,
        },
    )
    track_lateral_velocity_fine = RewTerm(
        func=mdp.track_lateral_velocity_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": 0.16},
    )
    track_yaw_velocity_fine = RewTerm(
        func=mdp.track_yaw_velocity_exp,
        weight=1.4,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    forward_lateral_drift = RewTerm(
        func=mdp.forward_lateral_drift,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )
    forward_yaw_drift = RewTerm(
        func=mdp.forward_yaw_drift,
        weight=-1.2,
        params={"command_name": "base_velocity"},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.12)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.90)
    base_height = RewTerm(
        func=mdp.base_height_above_terrain_l2,
        weight=-6.0,
        params={"target_height": 0.43, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    base_height_low = RewTerm(
        func=mdp.base_height_below_target_l2,
        weight=-18.0,
        params={"target_height": 0.43, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-4.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.12)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.015)
    torque_saturation = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-5.0e-6)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.3,
        },
    )
    hind_hip_roll_pose = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.85,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIND_HIP_ROLL_JOINT_NAMES),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.3,
        },
    )
    hind_leg_pose = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIND_LEG_JOINT_NAMES),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.3,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    air_time_variance = None
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES_ORDERED),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_terrain_l2,
        weight=-2.0,
        params={
            "target_height": 0.12,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES_ORDERED),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height_terrain_l2,
        weight=-0.25,
        params={
            "target_height": 0.12,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES_ORDERED),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED),
        },
    )
    hind_feet_swing_height = RewTerm(
        func=mdp.feet_swing_height_terrain_l2,
        weight=-0.32,
        params={
            "target_height": 0.15,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=HIND_FEET_BODY_NAMES),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_FEET_BODY_NAMES),
        },
    )
    diagonal_trot_contact = None
    bad_two_foot_contact = None
    all_feet_air = None
    feet_contact_count = None
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES_ORDERED)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.15,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf_link"),
        },
    )
    hind_calf_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.80,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=HIND_CALF_BODY_NAMES),
        },
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-25.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_NAME), "threshold": 5.0},
    )
    base_height = None
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2217304763960306})


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={
            "command_name": "base_velocity",
            "promotion_distance_ratio": 0.55,
            "promotion_command_ratio": 0.55,
            "demotion_command_ratio": 0.5,
            "minimum_promotion_distance": 2.0,
            "warmup_steps": 300 * 16,
            "level_step_interval": 800 * 16,
            "consecutive_successes": 1,
            "demote_only_early_termination": True,
            "min_level_hold_steps": 250 * 16,
        },
    )
    command_vel = CurrTerm(
        func=mdp.command_vel_stages,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.10, 0.65),
                    "lin_vel_y": (-0.10, 0.10),
                    "ang_vel_z": (-0.25, 0.25),
                },
                {
                    "step": 3000 * 16,
                    "lin_vel_x": (-0.15, 0.90),
                    "lin_vel_y": (-0.14, 0.14),
                    "ang_vel_z": (-0.32, 0.32),
                },
                {
                    "step": 7000 * 16,
                    "lin_vel_x": (-0.25, 1.20),
                    "lin_vel_y": (-0.18, 0.18),
                    "ang_vel_z": (-0.40, 0.40),
                },
                {
                    "step": 12000 * 16,
                    "lin_vel_x": (-0.35, 1.45),
                    "lin_vel_y": (-0.22, 0.22),
                    "ang_vel_z": (-0.48, 0.48),
                },
                {
                    "step": 20000 * 16,
                    "lin_vel_x": (-0.50, 1.80),
                    "lin_vel_y": (-0.30, 0.30),
                    "ang_vel_z": (-0.58, 0.58),
                },
                {
                    "step": 32000 * 16,
                    "lin_vel_x": (-0.65, 2.30),
                    "lin_vel_y": (-0.32, 0.32),
                    "ang_vel_z": (-0.65, 0.65),
                },
                {
                    "step": 45000 * 16,
                    "lin_vel_x": (-0.80, 2.65),
                    "lin_vel_y": (-0.35, 0.35),
                    "ang_vel_z": (-0.70, 0.70),
                },
                {
                    "step": 60000 * 16,
                    "lin_vel_x": (-0.80, 3.00),
                    "lin_vel_y": (-0.35, 0.35),
                    "ang_vel_z": (-0.75, 0.75),
                },
            ],
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = 2 * self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
                self.scene.terrain.max_init_terrain_level = 1
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = BPX_PLAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 4
            self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 5
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.observations.cenet.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.actuator_gains = None
        self.events.joint_parameters = None
        self.events.actuator_effort_limit = None
