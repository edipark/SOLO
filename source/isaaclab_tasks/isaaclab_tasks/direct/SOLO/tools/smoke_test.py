"""One-environment smoke test for every registered SOLO G1 task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=4)
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
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env = gym.make(task, cfg=cfg)
    observations, _ = env.reset()
    policy_observation = observations["policy"]
    assert policy_observation.shape == (1, observation_dim), policy_observation.shape
    assert env.unwrapped.get_estimator_target().shape == (1, 9)
    joint_pos, joint_vel, names = env.unwrapped.get_estimator_joint_state()
    assert joint_pos.shape == joint_vel.shape == (1, 29)
    assert len(names) == 29
    for _ in range(args.steps):
        observations, rewards, terminated, truncated, extras = env.step(torch.zeros((1, 29), device=env.unwrapped.device))
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert "log" in extras
    env.close()
    print(f"PASS {task}: obs={observation_dim}, action=29, estimator=58->9")


if __name__ == "__main__":
    main()
    simulation_app.close()
