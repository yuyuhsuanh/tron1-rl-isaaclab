"""Brake task for WF-TRON1A: wheels-only action, decelerate to stop."""

import math
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import (
    WFBlindFlatEnvCfg,
    WFBaseEnvCfg_PLAY,
)
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg


@configclass
class WFBrakeEnvCfg(WFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Command: target velocity always zero ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # --- Reset: start with nonzero velocity so policy must learn to brake ---
        self.events.reset_robot_base.params["velocity_range"]["x"] = (-1.5, 1.5)
        self.events.reset_robot_base.params["velocity_range"]["y"] = (-0.3, 0.3)

        # Wheels must spin at v/r to match base velocity — must run after reset_robot_base
        self.events.reset_wheel_velocity = EventTerm(
            func=mdp.reset_wheel_velocity_from_base,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "wheel_radius": 0.128},
        )

        # --- Reward: rew_lin_vel_xy + rew_ang_vel_z already reward v->0 when cmd=0. ---
        self.rewards.stand_still = None


@configclass
class WFBrakeEnvCfg_PLAY(WFBrakeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
