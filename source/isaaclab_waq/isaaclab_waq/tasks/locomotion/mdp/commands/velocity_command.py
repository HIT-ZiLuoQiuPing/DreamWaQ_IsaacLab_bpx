"""Velocity command configuration with explicit curriculum limits."""

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with ranges that curricula may expand toward."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

