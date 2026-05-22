"""Gym registration for BPX DreamWaQ rough-terrain tasks."""

import gymnasium as gym


gym.register(
    id="Isaac-BPX-WAQ-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotPlayEnvCfg",
        "waq_cfg_entry_point": "isaaclab_waq.algorithms.waq.config:DreamWaQConfig",
    },
)

gym.register(
    id="Isaac-BPX-WAQ-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotPlayEnvCfg",
        "waq_cfg_entry_point": "isaaclab_waq.algorithms.waq.config:DreamWaQConfig",
    },
)

