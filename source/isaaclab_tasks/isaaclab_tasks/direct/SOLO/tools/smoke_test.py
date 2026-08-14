"""One-environment smoke test for every registered SOLO G1 task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--task",
    choices=("amp_walk", "amp_dance", "ppo_walk"),
    default="amp_walk",
    help="Run one task per process because Isaac Sim owns a singleton simulation context",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


TASKS = {
    "amp_walk": ("Isaac-G1-AMP-Walk-SOLO-Direct-v0", 101),
    "amp_dance": ("Isaac-G1-AMP-Dance-SOLO-Direct-v0", 101),
    "ppo_walk": ("Isaac-G1-PPO-Walk-SOLO-Direct-v0", 99),
}


def main():
    task, observation_dim = TASKS[args.task]
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    observations, _ = env.reset()
    assert abs(env.unwrapped.step_dt - 1.0 / 30.0) < 1.0e-9
    assert env.unwrapped.max_episode_length == 600
    if task.startswith("Isaac-G1-AMP"):
        assert env.unwrapped.amp_observation_space.shape == (404,)
    assert env.unwrapped.sim.stage.GetPrimAtPath("/World/ground").IsValid()
    policy_observation = observations["policy"]
    assert policy_observation.shape == (args.num_envs, observation_dim), policy_observation.shape
    estimator_dim = 43 if task.startswith("Isaac-G1-AMP") else 9
    assert env.unwrapped.get_estimator_target().shape == (args.num_envs, estimator_dim)
    joint_pos, joint_vel, names = env.unwrapped.get_estimator_joint_state()
    assert joint_pos.shape == joint_vel.shape == (args.num_envs, 29)
    assert len(names) == 29
    limits = env.unwrapped.robot.data.joint_effort_limits[0]
    assert float(limits[names.index("left_hip_pitch_joint")]) == 88.0
    assert float(limits[names.index("left_knee_joint")]) == 139.0
    assert float(limits[names.index("left_ankle_pitch_joint")]) == 50.0
    for _ in range(args.steps):
        observations, rewards, terminated, truncated, extras = env.step(
            torch.zeros((args.num_envs, 29), device=env.unwrapped.device)
        )
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert "log" in extras
        targets = env.unwrapped.robot.data.joint_pos_target
        soft_limits = env.unwrapped.robot.data.soft_joint_pos_limits
        assert (targets >= soft_limits[..., 0] - 1.0e-5).all()
        assert (targets <= soft_limits[..., 1] + 1.0e-5).all()
    env.close()
    print(f"PASS {task}: obs={observation_dim}, action=29, estimator=58->{estimator_dim}")


if __name__ == "__main__":
    main()
    simulation_app.close()
