"""Evaluate the ground-truth privileged teacher baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--agent", required=True)
parser.add_argument("--adapter", choices=("amp", "ppo"), required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--collect-steps", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=0, help="Accepted for ablation command compatibility")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", default="logs/solo_g1/teacher")
parser.add_argument("--run-name", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.direct.SOLO.estimator.adapters import make_policy_adapter
from isaaclab_tasks.direct.SOLO.estimator.pipeline import collect_rollout
from isaaclab_tasks.direct.SOLO.skrl_compat import prepare_runner_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    prepare_runner_config(agent_cfg)
    agent_cfg["seed"] = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(Path(args_cli.teacher_checkpoint).resolve()))
    adapter = make_policy_adapter(args_cli.adapter, env, "all")
    _, metrics = collect_rollout(env, adapter, runner.agent, args_cli.collect_steps, window=1)
    default_name = (
        f"{args_cli.task}_TeacherGT_seed{args_cli.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output = Path(args_cli.output_dir) / (args_cli.run_name or default_name)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training.json").write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")
    print(f"TeacherGT return={metrics['return_mean']:.4f} success={metrics['success_rate']:.2f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
