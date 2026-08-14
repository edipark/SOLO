"""Train vanilla offline or DAgger action-distillation students for G1 policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SOLO G1 vanilla/DAgger distillation")
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-SOLO-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo"), default="amp")
parser.add_argument("--collect-steps", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--dagger-rounds", type=int, default=0, help="0 is vanilla offline distillation")
parser.add_argument("--max-dataset-size", type=int, default=500000)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", default="logs/solo_g1/distillation")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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
from isaaclab_tasks.direct.SOLO.estimator.models import VanillaStudent
from isaaclab_tasks.direct.SOLO.estimator.pipeline import collect_rollout, collect_student_rollout, train_student
from isaaclab_tasks.direct.SOLO.skrl_compat import prepare_runner_config, require_skrl_2


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    prepare_runner_config(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(Path(args_cli.teacher_checkpoint).resolve()))
    adapter = make_policy_adapter(args_cli.adapter, env, "all")
    dataset, collection = collect_rollout(env, adapter, runner.agent, args_cli.collect_steps, window=1)
    student = VanillaStudent(adapter.input_dim, adapter.schema.action_dim)
    training = train_student(student, dataset, args_cli.epochs, args_cli.batch_size, args_cli.lr, args_cli.device)
    rounds = [{"round": 0, "collection": collection, "training": training}]
    for round_index in range(1, args_cli.dagger_rounds + 1):
        teacher_probability = max(0.0, 1.0 - round_index / args_cli.dagger_rounds)
        new_data, collection = collect_student_rollout(
            env, adapter, runner.agent, student, args_cli.collect_steps, teacher_probability
        )
        dataset = dataset.append(new_data, args_cli.max_dataset_size)
        training = train_student(student, dataset, args_cli.epochs, args_cli.batch_size, args_cli.lr * 0.5, args_cli.device)
        rounds.append({"round": round_index, "teacher_probability": teacher_probability, "collection": collection, "training": training})
        print(f"  student_dagger={round_index}/{args_cli.dagger_rounds} teacher_p={teacher_probability:.2f}")
    _, evaluation = collect_student_rollout(env, adapter, runner.agent, student, args_cli.collect_steps, 0.0)
    kind = "student_dagger" if args_cli.dagger_rounds else "vanilla_student"
    output = Path(args_cli.output_dir) / f"{args_cli.task}_{kind}_seed{args_cli.seed}"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "solo_schema_version": 1, "kind": kind, "skrl_version": require_skrl_2(), "task": args_cli.task,
        "adapter": args_cli.adapter, "joint_preset": "all", "velocity_source": "sim_joint_velocity",
        "model_config": {"type": "VanillaStudent", "input_dim": 58, "action_dim": 29},
        "model_state_dict": student.state_dict(), "rounds": rounds, "metrics": evaluation,
    }
    torch.save(payload, output / "best_student.pt")
    (output / "training.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "model_state_dict"}, indent=2), encoding="utf-8")
    print(f"  saved={output / 'best_student.pt'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
