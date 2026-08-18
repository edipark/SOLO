"""Collection, supervised training, DAgger, evaluation, and checkpoint I/O."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import numpy as np
from torch import nn

from ..schema import JOINT_PRESETS, SCHEMA_VERSION
from ..skrl_compat import amp_reward_components, force_skrl_isaaclab_reset, require_skrl_2
from .adapters import PolicyAdapter
from .models import NormalizedEstimator, build_estimator


@dataclass
class RolloutDataset:
    histories: torch.Tensor
    targets: torch.Tensor
    frames: torch.Tensor
    teacher_actions: torch.Tensor

    def append(self, other: "RolloutDataset", max_size: int | None = None) -> "RolloutDataset":
        self_size = len(self.histories)
        other_size = len(other.histories)
        total_size = self_size + other_size
        if max_size is None or total_size <= max_size:
            return RolloutDataset(
                *(torch.cat((getattr(self, field), getattr(other, field))) for field in self.__dataclass_fields__)
            )

        # Do not concatenate the complete datasets before truncating. For a
        # 500k-sample w100 history, the old batch, new batch, concatenation and
        # indexed result would otherwise coexist and exceed 64 GB of RAM.
        device = self.histories.device
        selected = torch.randperm(total_size, device=device)[:max_size]
        values = []
        copy_chunk = 4096
        for field in self.__dataclass_fields__:
            first = getattr(self, field)
            second = getattr(other, field)
            if first.shape[1:] != second.shape[1:]:
                raise ValueError(f"Cannot append incompatible dataset field {field}")
            output = torch.empty((max_size, *first.shape[1:]), dtype=first.dtype, device=first.device)
            for start in range(0, max_size, copy_chunk):
                stop = min(start + copy_chunk, max_size)
                source = selected[start:stop]
                from_first = source < self_size
                if from_first.any():
                    destination = torch.arange(start, stop, device=device)[from_first]
                    output.index_copy_(0, destination, first.index_select(0, source[from_first]))
                if (~from_first).any():
                    destination = torch.arange(start, stop, device=device)[~from_first]
                    output.index_copy_(0, destination, second.index_select(0, source[~from_first] - self_size))
            values.append(output)
        return RolloutDataset(*values)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.pid{os.getpid()}.tmp")
        try:
            torch.save({"dataset": self.__dict__, "metadata": metadata or {}}, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "RolloutDataset":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**payload["dataset"])

    @classmethod
    def load_with_metadata(cls, path: str | Path) -> tuple["RolloutDataset", dict]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**payload["dataset"]), payload.get("metadata", {})

    def project_joint_history(
        self,
        joint_ids: tuple[int, ...] | list[int],
        source_window: int,
        target_window: int,
        full_joint_count: int,
    ) -> "RolloutDataset":
        """Derive a shorter/subset estimator dataset without recollection.

        The source is expected to contain all joint positions followed by all
        joint velocities. Taking the newest history suffix is exactly
        equivalent to collecting with a shorter history buffer.
        """
        if target_window > source_window:
            raise ValueError(f"Cannot derive window {target_window} from cached window {source_window}")
        joint_ids = tuple(joint_ids)
        all_joints = joint_ids == tuple(range(full_joint_count))
        if target_window == source_window and all_joints:
            return self
        ids = list(joint_ids) + [full_joint_count + index for index in joint_ids]
        histories = self.histories[:, -target_window:]
        if not all_joints:
            histories = histories[:, :, ids]
            frames = self.frames[:, ids]
        else:
            # Clone a shorter suffix so deleting the source releases the much
            # larger backing storage of the longest-window cache.
            histories = histories.clone()
            frames = self.frames
        return RolloutDataset(
            histories=histories,
            targets=self.targets,
            frames=frames,
            teacher_actions=self.teacher_actions,
        )


class HistoryBuffer:
    def __init__(self, num_envs: int, window: int, input_dim: int, device: torch.device | str):
        self.values = torch.zeros((num_envs, window, input_dim), device=device)

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        self.values = torch.roll(self.values, -1, dims=1)
        self.values[:, -1] = frame
        return self.values

    def reset(self, done: torch.Tensor) -> None:
        self.values[done] = 0.0


@torch.no_grad()
def collect_rollout(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    steps: int,
    window: int = 50,
    estimator: NormalizedEstimator | None = None,
    estimator_ratio: float = 0.0,
    action_noise: float = 0.0,
    max_samples: int | None = None,
) -> tuple[RolloutDataset, dict]:
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    history = HistoryBuffer(observations.shape[0], window, adapter.input_dim, observations.device)
    histories, targets, frames, teacher_actions = [], [], [], []
    deaths = timeouts = 0
    returns = torch.zeros(observations.shape[0], device=observations.device)
    lengths = torch.zeros(observations.shape[0], device=observations.device)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    metric_totals: dict[str, float] = {}
    metric_steps = 0
    previous_action = torch.zeros((observations.shape[0], adapter.schema.action_dim), device=observations.device)

    if estimator is not None:
        estimator.eval()
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    samples_per_step = observations.shape[0]
    if max_samples is not None:
        samples_per_step = max(1, min(observations.shape[0], max_samples // max(steps, 1)))
    for _ in range(steps):
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = history.push(frame)
        teacher_action = adapter.action(teacher_agent, observations)
        action = teacher_action
        estimator_action = None
        if estimator is not None and estimator_ratio > 0.0:
            estimate = estimator.predict(frame if estimator.__class__.__name__.startswith("MLP") else sequence)
            use_estimator = torch.rand(observations.shape[0], device=observations.device) < estimator_ratio
            estimated_obs = adapter.inject_estimate(observations, estimate)
            estimator_action = adapter.action(teacher_agent, estimated_obs)
            action = torch.where(use_estimator[:, None], estimator_action, teacher_action)
        if action_noise:
            action = action + action_noise * torch.randn_like(action)
        if samples_per_step < observations.shape[0]:
            sample_ids = torch.randperm(observations.shape[0], device=observations.device)[:samples_per_step]
        else:
            sample_ids = slice(None)
        histories.append(sequence[sample_ids].cpu().clone())
        targets.append(target[sample_ids].cpu().clone())
        frames.append(frame[sample_ids].cpu().clone())
        teacher_actions.append(teacher_action[sample_ids].cpu().clone())
        observations, rewards, terminated, truncated, _ = env.step(action)
        returns += rewards.flatten()
        lengths += 1
        core = adapter.core_env
        torque = core.robot.data.applied_torque
        joint_velocity = core.robot.data.joint_vel
        effort_limit = core.robot.data.joint_effort_limits.clamp_min(1.0e-6)
        metric_target = adapter.estimator_target()
        target_names = adapter.schema.estimator_target_names
        linear_ids = [target_names.index(f"base_lin_vel_{axis}") for axis in "xyz"]
        angular_ids = [target_names.index(f"base_ang_vel_{axis}") for axis in "xyz"]
        step_metrics = {
            "action_smoothness": float((action - previous_action).square().mean()),
            "torque_rms": float(torque.square().mean().sqrt()),
            "energy": float((torque * joint_velocity).abs().sum(dim=-1).mean() * core.step_dt),
            "action_saturation": float((action.abs() >= 0.999).float().mean()),
            "torque_saturation": float((torque.abs() >= effort_limit).float().mean()),
            "base_linear_speed": float(metric_target[:, linear_ids].norm(dim=-1).mean()),
            "base_angular_speed": float(metric_target[:, angular_ids].norm(dim=-1).mean()),
            "raw_task_reward": float(rewards.mean()),
        }
        if estimator_action is not None:
            step_metrics["teacher_action_mse"] = float((estimator_action - teacher_action).square().mean())
        extras = getattr(core, "extras", {})
        amp_components = (
            amp_reward_components(teacher_agent, extras.get("amp_obs"), rewards)
            if extras.get("amp_obs") is not None
            else None
        )
        if amp_components is not None:
            for name in ("raw_style", "scaled_task", "scaled_style", "effective_reward"):
                step_metrics[f"amp_{name}"] = float(amp_components[name].mean())
            step_metrics["task_reward_scale"] = amp_components["task_reward_scale"]
            step_metrics["style_reward_scale"] = amp_components["style_reward_scale"]
        for name, value in step_metrics.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + value
        metric_steps += 1
        previous_action = action
        done = (terminated | truncated).flatten()
        deaths += int(terminated.sum())
        timeouts += int((truncated & ~terminated).sum())
        if done.any():
            completed_returns.extend(returns[done].cpu().tolist())
            completed_lengths.extend(lengths[done].cpu().tolist())
            returns[done] = 0.0
            lengths[done] = 0.0
            history.reset(done)
            previous_action[done] = 0.0

    dataset = RolloutDataset(*(torch.cat(items) for items in (histories, targets, frames, teacher_actions)))
    stats = {
        "samples": len(dataset.targets),
        "deaths": deaths,
        "timeouts": timeouts,
        "death_rate": 100.0 * deaths / (deaths + timeouts) if deaths + timeouts else 0.0,
        "timeout_rate": 100.0 * timeouts / (deaths + timeouts) if deaths + timeouts else 0.0,
        "return_mean": sum(completed_returns) / len(completed_returns) if completed_returns else 0.0,
        "episode_length_mean": sum(completed_lengths) / len(completed_lengths) if completed_lengths else 0.0,
        "episode_length_std": float(np.std(completed_lengths)) if completed_lengths else 0.0,
        "success_rate": 100.0 * timeouts / (deaths + timeouts) if deaths + timeouts else 0.0,
        "velocity_source": "sim_joint_velocity",
        **{name: value / max(metric_steps, 1) for name, value in metric_totals.items()},
    }
    return dataset, stats


@torch.no_grad()
def evaluate_estimator_closed_loop(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    estimator: NormalizedEstimator,
    estimator_type: str,
    window: int,
    episodes: int = 200,
    max_episode_steps: int = 1000,
    seed: int | None = None,
) -> dict:
    """Evaluate estimator-injected teacher observations over completed episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    history = HistoryBuffer(observations.shape[0], window, adapter.input_dim, observations.device)
    returns = torch.zeros(observations.shape[0], device=observations.device)
    lengths = torch.zeros_like(returns)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    deaths = timeouts = 0
    squared_error = torch.zeros(adapter.schema.estimator_target_dim, dtype=torch.float64)
    sample_count = 0
    estimator.eval()
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    max_steps = max_episode_steps * max(1, (episodes + observations.shape[0] - 1) // observations.shape[0] + 1)
    for _ in range(max_steps):
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = history.push(frame)
        model_input = frame if estimator_type.upper() == "MLP" else sequence
        estimate = estimator.predict(model_input)
        squared_error += (estimate - target).double().square().sum(dim=0).cpu()
        sample_count += target.shape[0]
        action = adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))
        observations, rewards, terminated, truncated, _ = env.step(action)
        returns += rewards.flatten()
        lengths += 1
        done = (terminated | truncated).flatten()
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            remaining = episodes - len(completed_lengths)
            selected = done_ids[:remaining]
            completed_returns.extend(returns[selected].cpu().tolist())
            completed_lengths.extend(lengths[selected].cpu().tolist())
            deaths += int(terminated[selected].sum())
            timeouts += int((truncated[selected] & ~terminated[selected]).sum())
            returns[done_ids] = 0.0
            lengths[done_ids] = 0.0
            history.reset(done)
            if len(completed_lengths) >= episodes:
                break
    if not completed_lengths:
        raise RuntimeError("Closed-loop evaluation completed no episodes")
    target_rmse = (squared_error / max(sample_count, 1)).sqrt().float()
    return {
        "episodes": len(completed_lengths),
        "episode_length_mean": sum(completed_lengths) / len(completed_lengths),
        "episode_length_std": float(np.std(completed_lengths)),
        "return_mean": sum(completed_returns) / len(completed_returns),
        "deaths": deaths,
        "timeouts": timeouts,
        "death_rate": 100.0 * deaths / len(completed_lengths),
        "timeout_rate": 100.0 * timeouts / len(completed_lengths),
        "success_rate": 100.0 * timeouts / len(completed_lengths),
        "rmse": float(target_rmse.square().mean().sqrt()),
        "target_rmse": target_rmse.tolist(),
    }


def _fit_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    epoch_callback: Callable[[dict], None] | None = None,
) -> dict:
    model.to(device)
    size = len(targets)
    validation_size = max(1, min(size // 10, 10000))
    order = torch.randperm(size)
    validation_ids, training_ids = order[:validation_size], order[validation_size:]
    if len(training_ids) == 0:
        raise ValueError("At least two samples are required for training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    best_loss, best_state = float("inf"), None
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = training_ids[torch.randperm(len(training_ids))]
        train_total = 0.0
        for start in range(0, len(permutation), batch_size):
            ids = permutation[start : start + batch_size]
            prediction = model(inputs[ids].to(device))
            loss = nn.functional.mse_loss(prediction, targets[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss) * len(ids)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.mse_loss(model(inputs[validation_ids].to(device)), targets[validation_ids].to(device))
            )
        row = {"epoch": epoch + 1, "train_mse": train_total / len(training_ids), "validation_mse": validation_loss}
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"best_validation_mse": best_loss, "epochs": history}


def train_estimator(
    estimator: NormalizedEstimator,
    dataset: RolloutDataset,
    estimator_type: str,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    device: str = "cuda:0",
    epoch_callback: Callable[[dict], None] | None = None,
) -> dict:
    inputs = dataset.frames if estimator_type.upper() == "MLP" else dataset.histories
    estimator.to(inputs.device)
    estimator.set_normalization(inputs, dataset.targets)
    normalized_targets = estimator.normalized_targets(dataset.targets)
    return _fit_model(
        estimator, inputs, normalized_targets, epochs, batch_size, learning_rate, device, epoch_callback
    )


def evaluate_predictions(
    estimator: NormalizedEstimator,
    dataset: RolloutDataset,
    estimator_type: str,
    device: str,
    target_names: tuple[str, ...] | list[str] | None = None,
) -> dict:
    inputs = dataset.frames if estimator_type.upper() == "MLP" else dataset.histories
    estimator.eval().to(device)
    start = time.perf_counter()
    with torch.no_grad():
        prediction = estimator.predict(inputs.to(device)).cpu()
    elapsed = time.perf_counter() - start
    error = prediction - dataset.targets
    mse = error.square().mean(dim=0)
    variance = dataset.targets.var(dim=0).clamp_min(1.0e-8)
    metrics = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "r2": float((1.0 - mse / variance).mean()),
        "target_mae": error.abs().mean(dim=0).tolist(),
        "target_rmse": mse.sqrt().tolist(),
        "inference_ms_per_sample": elapsed * 1000.0 / len(inputs),
        "parameters": sum(parameter.numel() for parameter in estimator.parameters()),
        "trace_target": dataset.targets[:200].tolist(),
        "trace_prediction": prediction[:200].tolist(),
    }
    if target_names is not None:
        metrics["target_names"] = list(target_names)
    return metrics


def save_solo_checkpoint(
    path: str | Path,
    model: nn.Module,
    adapter: PolicyAdapter,
    task: str,
    window: int,
    metrics: dict,
    kind: str = "estimator",
) -> None:
    skrl_version = require_skrl_2()
    payload = {
        "solo_schema_version": SCHEMA_VERSION,
        "kind": kind,
        "skrl_version": skrl_version,
        "task": task,
        "adapter": adapter.name(),
        "observation_schema": adapter.schema.to_dict(),
        "joint_preset": adapter.joint_preset,
        "joint_names": JOINT_PRESETS[adapter.joint_preset],
        "velocity_source": "sim_joint_velocity",
        "window": window,
        "model_config": model.config() if hasattr(model, "config") else {"type": model.__class__.__name__},
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "module_manifest": sorted(model.state_dict()),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    json_payload = {
        key: value for key, value in payload.items() if key not in ("model_state_dict", "module_manifest")
    }
    path.with_suffix(".json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )


def load_estimator(path: str | Path, device: str = "cpu") -> tuple[NormalizedEstimator, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("solo_schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported SOLO checkpoint schema")
    if payload.get("kind") != "estimator":
        raise ValueError("Checkpoint is not a SOLO state estimator")
    if payload.get("velocity_source") != "sim_joint_velocity":
        raise ValueError("Checkpoint was not trained with simulator joint velocities")
    config = payload["model_config"]
    estimator = build_estimator(
        config["type"], config["input_dim"], config["output_dim"],
        config.get("hidden_size", 256), config.get("num_layers", 2), tuple(config.get("channels", (64, 128, 128))),
    ).to(device)
    estimator.load_state_dict(payload["model_state_dict"])
    estimator.eval()
    return estimator, payload
