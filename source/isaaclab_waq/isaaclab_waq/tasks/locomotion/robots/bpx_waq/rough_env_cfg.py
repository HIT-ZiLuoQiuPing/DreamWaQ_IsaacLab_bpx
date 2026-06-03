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
    FEET_BODY_NAMES,
    UNDESIRED_BODY_NAMES,
)
from isaaclab_waq.tasks.locomotion import mdp


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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.50),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.002, 0.03),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.12,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.08,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.04,
            grid_width=0.45,
            grid_height_range=(0.01, 0.06),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=(0.01, 0.06),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.02,
            step_height_range=(0.01, 0.06),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
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
        max_init_terrain_level=0,
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
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 0.45),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.60, 1.80),
            lin_vel_y=(-0.35, 0.35),
            ang_vel_z=(-0.70, 0.70),
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
    upright = RewTerm(func=mdp.upright_exp, weight=0.35, params={"std": 0.45})

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.4,
        params={"command_name": "base_velocity", "std": 0.32},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.6,
        params={"command_name": "base_velocity", "std": 0.40},
    )
    track_forward_velocity_fine = RewTerm(
        func=mdp.track_forward_velocity_exp,
        weight=2.2,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    forward_velocity_error = RewTerm(
        func=mdp.forward_velocity_error_l1,
        weight=-1.2,
        params={"command_name": "base_velocity"},
    )
    no_forward_motion = RewTerm(
        func=mdp.no_forward_motion,
        weight=-1.0,
        params={"command_name": "base_velocity", "min_command": 0.15, "min_velocity_ratio": 0.25},
    )
    track_lateral_velocity_fine = RewTerm(
        func=mdp.track_lateral_velocity_exp,
        weight=0.3,
        params={"command_name": "base_velocity", "std": 0.16},
    )
    track_yaw_velocity_fine = RewTerm(
        func=mdp.track_yaw_velocity_exp,
        weight=0.3,
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
    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry,
        weight=-0.25,
        params={"command_name": "base_velocity"},
    )

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    base_height = RewTerm(
        func=mdp.base_height_above_terrain_l2,
        weight=-3.0,
        params={"target_height": 0.43, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-4.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.06)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-5.0e-6)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.3,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.05,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
        },
    )
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.18,
        params={
            "period": 0.55,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
        },
    )
    all_feet_air = RewTerm(
        func=mdp.all_feet_air,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
        },
    )
    feet_contact_count = RewTerm(
        func=mdp.feet_contact_count_error,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "moving_contact_count": 2.0,
            "standing_contact_count": 4.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_BODY_NAMES),
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
    base_height = DoneTerm(
        func=mdp.root_height_below_terrain,
        params={"minimum_height": 0.32, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.8})


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={
            "command_name": "base_velocity",
            "promotion_distance_ratio": 0.85,
            "demotion_command_ratio": 0.25,
            "warmup_steps": 3000 * 12,
            "level_step_interval": 2500 * 12,
            "consecutive_successes": 3,
            "demote_only_early_termination": True,
            "min_level_hold_steps": 500 * 12,
        },
    )
    command_vel = CurrTerm(
        func=mdp.command_vel_stages,
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (0.10, 0.45), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
                {
                    "step": 4000 * 12,
                    "lin_vel_x": (0.10, 0.65),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-0.10, 0.10),
                },
                {
                    "step": 8000 * 12,
                    "lin_vel_x": (0.10, 0.85),
                    "lin_vel_y": (-0.05, 0.05),
                    "ang_vel_z": (-0.18, 0.18),
                },
                {
                    "step": 12000 * 12,
                    "lin_vel_x": (0.15, 1.05),
                    "lin_vel_y": (-0.10, 0.10),
                    "ang_vel_z": (-0.25, 0.25),
                },
                {
                    "step": 16000 * 12,
                    "lin_vel_x": (0.15, 1.20),
                    "lin_vel_y": (-0.12, 0.12),
                    "ang_vel_z": (-0.30, 0.30),
                },
                {
                    "step": 22000 * 12,
                    "lin_vel_x": (0.00, 1.35),
                    "lin_vel_y": (-0.18, 0.18),
                    "ang_vel_z": (-0.40, 0.40),
                },
                {
                    "step": 32000 * 12,
                    "lin_vel_x": (-0.20, 1.45),
                    "lin_vel_y": (-0.30, 0.30),
                    "ang_vel_z": (-0.60, 0.60),
                },
                {
                    "step": 43000 * 12,
                    "lin_vel_x": (-0.55, 1.60),
                    "lin_vel_y": (-0.32, 0.32),
                    "ang_vel_z": (-0.65, 0.65),
                },
                {
                    "step": 55000 * 12,
                    "lin_vel_x": (-0.60, 1.80),
                    "lin_vel_y": (-0.35, 0.35),
                    "ang_vel_z": (-0.70, 0.70),
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
                self.scene.terrain.max_init_terrain_level = 0
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
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.observations.cenet.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
