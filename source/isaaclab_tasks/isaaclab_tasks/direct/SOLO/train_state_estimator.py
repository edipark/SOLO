"""Train a G1 state estimator with initial teacher data and optional DAgger rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="SOLO G1 state-estimator training")
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-SOLO-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo"), default="amp")
parser.add_argument("--joint-preset", choices=("all", "legs", "upper"), default="all")
parser.add_argument("--estimator", choices=("LSTM", "TCN", "MLP"), default="LSTM")
parser.add_argument("--window", type=int, default=50)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--tcn-channels", type=int, nargs="+", default=[64, 128, 128])
parser.add_argument("--collect-steps", type=int, default=2000)
parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.02])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--dagger-rounds", type=int, default=10)
parser.add_argument("--dagger-est-ratio", type=float, default=0.8)
parser.add_argument("--dagger-est-ratio-final", type=float, default=1.0)
parser.add_argument("--dagger-est-ratio-schedule", choices=("linear", "constant"), default="linear")
parser.add_argument("--dagger-extra-rounds", type=int, default=0)
parser.add_argument("--max-dataset-size", type=int, default=500000)
parser.add_argument("--eval-episodes", type=int, default=200)
parser.add_argument("--max-episode-steps", type=int, default=1000)
parser.add_argument("--eval-seed-offset", type=int, default=10000)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", default="logs/solo_g1/estimators")
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
from isaaclab_tasks.direct.SOLO.estimator.models import build_estimator
from isaaclab_tasks.direct.SOLO.estimator.pipeline import (
    collect_rollout,
    evaluate_estimator_closed_loop,
    evaluate_predictions,
    save_solo_checkpoint,
    train_estimator,
)
from isaaclab_tasks.direct.SOLO.skrl_compat import prepare_runner_config


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
    adapter = make_policy_adapter(args_cli.adapter, env, args_cli.joint_preset)
    window = 1 if args_cli.estimator == "MLP" else args_cli.window
    estimator = build_estimator(
        args_cli.estimator, adapter.input_dim, adapter.schema.estimator_target_dim,
        args_cli.hidden_size, args_cli.num_layers, tuple(args_cli.tcn_channels),
    )
    run_name = f"{args_cli.task}_{args_cli.estimator}_w{window}_{args_cli.joint_preset}_seed{args_cli.seed}"
    output = Path(args_cli.output_dir) / run_name
    output.mkdir(parents=True, exist_ok=True)

    print("\nSOLO G1 estimator")
    print(f"  task={args_cli.task} adapter={args_cli.adapter} preset={args_cli.joint_preset}")
    print(
        f"  model={args_cli.estimator} window={window} input={adapter.input_dim} "
        f"target={adapter.schema.estimator_target_dim}"
    )
    print("  velocity_source=sim_joint_velocity")
    dataset = None
    initial_collections = []
    initial_sample_cap = max(1, args_cli.max_dataset_size // max(len(args_cli.noise_levels), 1))
    for noise in args_cli.noise_levels:
        batch, collection = collect_rollout(
            env,
            adapter,
            runner.agent,
            args_cli.collect_steps,
            window,
            action_noise=noise,
            max_samples=initial_sample_cap,
        )
        dataset = batch if dataset is None else dataset.append(batch, args_cli.max_dataset_size)
        initial_collections.append({"action_noise": noise, **collection})
    assert dataset is not None
    training = train_estimator(
        estimator, dataset, args_cli.estimator, args_cli.epochs, args_cli.batch_size, args_cli.lr, args_cli.device
    )
    closed_loop = evaluate_estimator_closed_loop(
        env,
        adapter,
        runner.agent,
        estimator,
        args_cli.estimator,
        window,
        args_cli.eval_episodes,
        args_cli.max_episode_steps,
        args_cli.seed + args_cli.eval_seed_offset,
    )
    rounds = [{"round": 0, "collection": initial_collections, "training": training, "evaluation": closed_loop}]
    best_score = (
        closed_loop["episode_length_mean"],
        -closed_loop["death_rate"],
        -closed_loop["rmse"],
    )
    best_round = 0
    best_state = {name: value.detach().cpu().clone() for name, value in estimator.state_dict().items()}
    save_solo_checkpoint(output / "estimator_round_0.pt", estimator, adapter, args_cli.task, window, closed_loop)

    total_rounds = args_cli.dagger_rounds + args_cli.dagger_extra_rounds
    for round_index in range(1, total_rounds + 1):
        if round_index > args_cli.dagger_rounds:
            ratio = 1.0
        elif args_cli.dagger_est_ratio_schedule == "constant" or args_cli.dagger_rounds <= 1:
            ratio = args_cli.dagger_est_ratio
        else:
            progress = (round_index - 1) / (args_cli.dagger_rounds - 1)
            ratio = args_cli.dagger_est_ratio + progress * (
                args_cli.dagger_est_ratio_final - args_cli.dagger_est_ratio
            )
        new_data, collection = collect_rollout(
            env,
            adapter,
            runner.agent,
            args_cli.collect_steps,
            window,
            estimator,
            ratio,
            action_noise=0.01,
            max_samples=args_cli.max_dataset_size,
        )
        dataset = dataset.append(new_data, args_cli.max_dataset_size)
        training = train_estimator(
            estimator,
            dataset,
            args_cli.estimator,
            args_cli.epochs,
            args_cli.batch_size,
            args_cli.lr * 0.5,
            args_cli.device,
        )
        closed_loop = evaluate_estimator_closed_loop(
            env,
            adapter,
            runner.agent,
            estimator,
            args_cli.estimator,
            window,
            args_cli.eval_episodes,
            args_cli.max_episode_steps,
            args_cli.seed + args_cli.eval_seed_offset,
        )
        rounds.append(
            {
                "round": round_index,
                "estimator_ratio": ratio,
                "collection": collection,
                "training": training,
                "evaluation": closed_loop,
            }
        )
        score = (closed_loop["episode_length_mean"], -closed_loop["death_rate"], -closed_loop["rmse"])
        if score > best_score:
            best_score = score
            best_round = round_index
            best_state = {name: value.detach().cpu().clone() for name, value in estimator.state_dict().items()}
        save_solo_checkpoint(
            output / f"estimator_round_{round_index}.pt", estimator, adapter, args_cli.task, window, closed_loop
        )
        print(
            f"  round={round_index}/{total_rounds} ratio={ratio:.2f} samples={len(dataset.targets):,} "
            f"episode={closed_loop['episode_length_mean']:.1f} death={closed_loop['death_rate']:.1f}%"
        )

    estimator.load_state_dict(best_state)
    evaluation_data, _ = collect_rollout(
        env,
        adapter,
        runner.agent,
        min(args_cli.collect_steps, 200),
        window,
        estimator,
        estimator_ratio=1.0,
        max_samples=10000,
    )
    metrics = evaluate_predictions(
        estimator,
        evaluation_data,
        args_cli.estimator,
        args_cli.device,
        adapter.schema.estimator_target_names,
    )
    metrics.update(rounds[best_round]["evaluation"])
    metrics["best_round"] = best_round
    metrics["rounds"] = rounds
    save_solo_checkpoint(output / "best_estimator.pt", estimator, adapter, args_cli.task, window, metrics)
    (output / "training.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"  saved={output / 'best_estimator.pt'} best_round={best_round} "
        f"rmse={metrics['rmse']:.5f} r2={metrics['r2']:.4f}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
