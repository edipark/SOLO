"""Iterative DAgger distillation from a privileged G1 teacher to a 58-D student."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="SOLO G1 DAgger policy distillation")
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-AMP-Walk-SOLO-Direct-v0")
parser.add_argument("--agent", default="skrl_amp_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo"), default="amp")
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--num-iterations", type=int, default=300)
parser.add_argument("--rollout-steps", type=int, default=500)
parser.add_argument("--train-steps", type=int, default=0, help="0 uses twice the newly collected batch count")
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1.0e-4)
parser.add_argument("--weight-decay", type=float, default=1.0e-4)
parser.add_argument("--beta-init", type=float, default=1.0)
parser.add_argument("--beta-decay", type=float, default=0.998)
parser.add_argument("--beta-min", type=float, default=0.02)
parser.add_argument("--buffer-capacity", type=int, default=2_000_000)
parser.add_argument("--eval-interval", type=int, default=20)
parser.add_argument("--eval-steps", type=int, default=300)
parser.add_argument("--save-interval", type=int, default=50)
parser.add_argument("--log-dir", default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime
import json
from pathlib import Path
import time

import gymnasium as gym
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.direct.SOLO.estimator.adapters import make_policy_adapter
from isaaclab_tasks.direct.SOLO.estimator.models import (
    DaggerStudent,
    ReplayBuffer,
    RunningNormalizer,
    dagger_beta,
)
from isaaclab_tasks.direct.SOLO.schema import JOINT_PRESETS, SCHEMA_VERSION
from isaaclab_tasks.direct.SOLO.skrl_compat import prepare_runner_config, require_skrl_2


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    prepare_runner_config(agent_cfg)
    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(Path(args_cli.teacher_checkpoint).resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    adapter = make_policy_adapter(args_cli.adapter, env, "all")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args_cli.log_dir or f"logs/solo_g1/dagger/{timestamp}").resolve()
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    device = adapter.core_env.device
    student = DaggerStudent(adapter.input_dim, adapter.schema.action_dim).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args_cli.lr, weight_decay=args_cli.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_cli.num_iterations, eta_min=args_cli.lr * 0.1
    )
    observation_normalizer = RunningNormalizer(adapter.input_dim, device)
    action_normalizer = RunningNormalizer(adapter.schema.action_dim, device, clip=10.0)
    replay = ReplayBuffer(
        args_cli.buffer_capacity, adapter.input_dim, adapter.schema.action_dim, device
    )
    policy_dt = float(adapter.core_env.step_dt)
    beta = dagger_beta(args_cli.beta_init, args_cli.beta_decay, args_cli.beta_min, 0)
    best_episode_length = float("-inf")
    best_iteration = 0
    best_metrics: dict = {}
    evaluation_history: list[dict] = []
    observations, _ = env.reset()
    started = time.monotonic()

    def checkpoint_payload(iteration: int) -> dict:
        return {
            "solo_schema_version": SCHEMA_VERSION,
            "kind": "dagger_student",
            "skrl_version": require_skrl_2(),
            "task": args_cli.task,
            "adapter": args_cli.adapter,
            "joint_preset": "all",
            "joint_names": JOINT_PRESETS["all"],
            "velocity_source": "sim_joint_velocity",
            "policy_dt": policy_dt,
            "iteration": iteration,
            "beta": beta,
            "buffer_size": replay.size,
            "model_config": student.config(),
            "model_state_dict": student.state_dict(),
            "observation_normalizer": observation_normalizer.state_dict(),
            "action_normalizer": action_normalizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

    def save_checkpoint(
        iteration: int, *, best_metrics: dict | None = None, numbered: bool = False
    ) -> None:
        payload = checkpoint_payload(iteration)
        if best_metrics is not None:
            payload["best_metrics"] = best_metrics
            torch.save(payload, checkpoint_dir / "student_best_eval.pt")
            return
        torch.save(payload, checkpoint_dir / "student_latest.pt")
        if numbered:
            torch.save(payload, checkpoint_dir / f"student_iter_{iteration:05d}.pt")

    @torch.no_grad()
    def evaluate() -> tuple[dict, torch.Tensor]:
        student.eval()
        eval_observations, _ = env.reset()
        lengths = torch.zeros(eval_observations.shape[0], device=device)
        returns = torch.zeros_like(lengths)
        completed_lengths: list[float] = []
        completed_returns: list[float] = []
        mse_total = action_norm_total = 0.0
        deaths = timeouts = 0
        for _ in range(args_cli.eval_steps):
            frame = adapter.estimator_input()
            teacher_action = adapter.action(runner.agent, eval_observations)
            student_action = action_normalizer.denormalize(student(observation_normalizer.normalize(frame)))
            mse_total += float(nn.functional.mse_loss(student_action, teacher_action))
            action_norm_total += float(student_action.norm(dim=-1).mean())
            eval_observations, rewards, terminated, truncated, _ = env.step(student_action)
            lengths += 1
            returns += rewards.flatten()
            done = (terminated | truncated).flatten()
            if done.any():
                completed_lengths.extend(lengths[done].cpu().tolist())
                completed_returns.extend(returns[done].cpu().tolist())
                deaths += int(terminated.sum())
                timeouts += int((truncated & ~terminated).sum())
                lengths[done] = 0.0
                returns[done] = 0.0
        mean_length = (
            sum(completed_lengths) / len(completed_lengths)
            if completed_lengths
            else float(lengths.mean())
        )
        mean_return = (
            sum(completed_returns) / len(completed_returns)
            if completed_returns
            else float(returns.mean())
        )
        completed = deaths + timeouts
        metrics = {
            "action_mse": mse_total / args_cli.eval_steps,
            "episode_length_mean": mean_length,
            "return_mean": mean_return,
            "student_action_norm": action_norm_total / args_cli.eval_steps,
            "deaths": deaths,
            "timeouts": timeouts,
            "death_rate": 100.0 * deaths / completed if completed else 0.0,
            "timeout_rate": 100.0 * timeouts / completed if completed else 0.0,
        }
        return metrics, env.reset()[0]

    try:
        for iteration in range(1, args_cli.num_iterations + 1):
            student.eval()
            rollout_frames: list[torch.Tensor] = []
            rollout_labels: list[torch.Tensor] = []
            for _ in range(args_cli.rollout_steps):
                frame = adapter.estimator_input()
                observation_normalizer.update(frame)
                with torch.no_grad():
                    teacher_action = adapter.action(runner.agent, observations)
                    student_action = action_normalizer.denormalize(
                        student(observation_normalizer.normalize(frame))
                    )
                action_normalizer.update(teacher_action)
                action = beta * teacher_action + (1.0 - beta) * student_action
                rollout_frames.append(frame.detach())
                rollout_labels.append(teacher_action.detach())
                observations, _, _, _, _ = env.step(action)
            replay.add(torch.cat(rollout_frames), torch.cat(rollout_labels))

            student.train()
            new_samples = args_cli.num_envs * args_cli.rollout_steps
            train_steps = args_cli.train_steps or max(100, 2 * new_samples // args_cli.batch_size)
            loss_total = grad_total = 0.0
            for _ in range(train_steps):
                frame, teacher_action = replay.sample(args_cli.batch_size)
                prediction = student(observation_normalizer.normalize(frame))
                target = action_normalizer.normalize(teacher_action)
                loss = nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                loss_total += float(loss)
                grad_total += float(grad)
            beta = dagger_beta(args_cli.beta_init, args_cli.beta_decay, args_cli.beta_min, iteration)
            scheduler.step()
            writer.add_scalar("Loss/mse", loss_total / train_steps, iteration)
            writer.add_scalar("Metric/beta", beta, iteration)
            writer.add_scalar("Metric/buffer_size", replay.size, iteration)
            writer.add_scalar("Metric/grad_norm", grad_total / train_steps, iteration)
            writer.add_scalar("Metric/lr", scheduler.get_last_lr()[0], iteration)
            save_checkpoint(
                iteration,
                numbered=iteration % args_cli.save_interval == 0 or iteration == args_cli.num_iterations,
            )
            print(
                f"[iter {iteration:04d}/{args_cli.num_iterations}] loss={loss_total / train_steps:.6f} "
                f"beta={beta:.4f} buffer={replay.size:,} elapsed={time.monotonic() - started:.0f}s"
            )

            if iteration % args_cli.eval_interval == 0 or iteration == args_cli.num_iterations:
                metrics, observations = evaluate()
                evaluation_history.append({"iteration": iteration, **metrics})
                for name, value in metrics.items():
                    writer.add_scalar(f"Eval/{name}", value, iteration)
                print(
                    f"  eval episode={metrics['episode_length_mean']:.1f} "
                    f"mse={metrics['action_mse']:.6f} deaths={metrics['deaths']}"
                )
                if metrics["episode_length_mean"] > best_episode_length:
                    best_episode_length = metrics["episode_length_mean"]
                    best_iteration = iteration
                    best_metrics = dict(metrics)
                    save_checkpoint(iteration, best_metrics=metrics)
    finally:
        writer.close()
        env.close()
    print(
        f"DAgger complete: best mean episode length={best_episode_length:.1f} "
        f"at iteration {best_iteration}; checkpoints={checkpoint_dir}"
    )
    (log_dir / "training.json").write_text(
        json.dumps(
            {
                "metrics": {
                    **best_metrics,
                    "best_iteration": best_iteration,
                },
                "evaluations": evaluation_history,
                "policy_dt": policy_dt,
                "velocity_source": "sim_joint_velocity",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
