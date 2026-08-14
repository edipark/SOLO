"""Play an SKRL 2.x G1 teacher with a SOLO state estimator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 policy with estimator-injected base state")
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--estimator-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-SOLO-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo"), default="amp")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--rollout-csv", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.direct.SOLO.estimator.adapters import make_policy_adapter
from isaaclab_tasks.direct.SOLO.estimator.pipeline import HistoryBuffer, load_estimator
from isaaclab_tasks.direct.SOLO.skrl_compat import prepare_runner_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    prepare_runner_config(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder="logs/solo_g1/videos", step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length, disable_logger=True,
        )
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(Path(args_cli.teacher_checkpoint).resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    estimator, payload = load_estimator(args_cli.estimator_checkpoint, args_cli.device)
    if payload["task"] != args_cli.task or payload["adapter"] != args_cli.adapter:
        raise ValueError(
            f"Checkpoint interface is {payload['task']}/{payload['adapter']}, requested {args_cli.task}/{args_cli.adapter}"
        )
    adapter = make_policy_adapter(args_cli.adapter, env, payload["joint_preset"])
    observations, _ = env.reset()
    history = HistoryBuffer(observations.shape[0], payload["window"], adapter.input_dim, observations.device)
    csv_file = None
    writer = None
    if args_cli.rollout_csv:
        output = Path(args_cli.rollout_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["step", *[f"action_{i}" for i in range(29)], *[f"estimate_{i}" for i in range(9)]])
    try:
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                frame = adapter.estimator_input()
                sequence = history.push(frame)
                model_input = frame if payload["model_config"]["type"].upper() == "MLP" else sequence
                estimate = estimator.predict(model_input)
                action = adapter.action(runner.agent, adapter.inject_estimate(observations, estimate))
                if writer:
                    writer.writerow([step, *action[0].cpu().tolist(), *estimate[0].cpu().tolist()])
                observations, _, terminated, truncated, _ = env.step(action)
                done = (terminated | truncated).flatten()
                if done.any():
                    history.reset(done)
    finally:
        if csv_file:
            csv_file.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
