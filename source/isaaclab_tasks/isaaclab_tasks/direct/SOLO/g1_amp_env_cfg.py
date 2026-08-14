"""Environment configurations for G1 AMP walk and dance."""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .g1_robot_cfg import G1_SOLO_CFG


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    observation_space = 101
    action_space = 29
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 101

    early_termination = True
    termination_height = 0.48
    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random"
    task_kind = "walk"

    target_velocity = 0.8
    velocity_tracking_sigma = 0.25
    nominal_height = 0.78
    height_sigma = 0.08
    upright_weight = 0.35
    height_weight = 0.20
    velocity_weight = 0.45
    action_rate_penalty = 0.002
    joint_velocity_penalty = 1.0e-5

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot = G1_SOLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")
    task_kind = "walk"


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_dance.npz")
    task_kind = "dance"
    velocity_weight = 0.0
    upright_weight = 0.65
    height_weight = 0.35

