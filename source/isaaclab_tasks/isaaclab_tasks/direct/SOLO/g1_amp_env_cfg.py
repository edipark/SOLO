"""Environment configurations for G1 AMP walk and dance."""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .g1_robot_cfg import G1_SOLO_CFG
from .task_math import AMP_HISTORY_STEPS, CONTROL_DECIMATION, EPISODE_LENGTH_S, PHYSICS_DT


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    episode_length_s = EPISODE_LENGTH_S
    decimation = CONTROL_DECIMATION
    observation_space = 101
    action_space = 29
    state_space = 0
    num_amp_observations = AMP_HISTORY_STEPS
    amp_observation_space = 101

    early_termination = True
    termination_height = 0.55
    termination_min_vel_x = 0.0
    vel_window_min_vx = 0.0
    vel_window_steps = 10
    motion_file: str = MISSING
    motion_speed_scale = 1.0
    reference_body = "pelvis"
    reset_strategy = "random"

    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=decimation,
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot = G1_SOLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_dance.npz")
    reset_strategy = "default"
