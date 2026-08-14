"""Pure timing and reward helpers shared by SOLO G1 environments and tests."""

from __future__ import annotations

import numpy as np
import torch


PHYSICS_DT = 1.0 / 120.0
CONTROL_DECIMATION = 4
POLICY_DT = PHYSICS_DT * CONTROL_DECIMATION
EPISODE_LENGTH_S = 20.0
AMP_HISTORY_STEPS = 4
WALK_TARGET_VELOCITY = 0.6
NORMALIZED_ACTION_LIMIT = 1.0


def clip_normalized_actions(
    actions: torch.Tensor, limit: float = NORMALIZED_ACTION_LIMIT
) -> torch.Tensor:
    """Clamp normalized policy actions before mapping them to joint targets."""
    if limit <= 0.0:
        raise ValueError(f"Action clip limit must be positive, got {limit}")
    return actions.clamp(min=-limit, max=limit)


def soft_limit_action_parameters(soft_joint_limits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the midpoint offset and half-range scale of ``[..., joint, lower/upper]`` limits."""
    if soft_joint_limits.shape[-1] != 2:
        raise ValueError("Soft joint limits must end with lower/upper bounds")
    lower, upper = soft_joint_limits.unbind(dim=-1)
    if torch.any(upper < lower):
        raise ValueError("Soft joint upper limits must be greater than or equal to lower limits")
    return 0.5 * (upper + lower), 0.5 * (upper - lower)


def normalized_action_to_position(
    actions: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Clip normalized actions and map them into soft joint-position limits."""
    return offset + scale * clip_normalized_actions(actions)


def inject_observation_estimate(
    observations: torch.Tensor, estimate: torch.Tensor, indices: tuple[int, ...]
) -> torch.Tensor:
    """Replace selected policy-observation columns without mutating the source tensor."""
    if estimate.shape[:-1] != observations.shape[:-1] or estimate.shape[-1] != len(indices):
        raise ValueError("Estimator output does not match the selected observation columns")
    result = observations.clone()
    result[..., list(indices)] = estimate
    return result


def reference_history_times(
    current_times: np.ndarray,
    num_steps: int = AMP_HISTORY_STEPS,
    policy_dt: float = POLICY_DT,
) -> np.ndarray:
    """Return newest-to-oldest AMP sample times at the policy control interval."""
    return (
        np.expand_dims(current_times, axis=-1)
        - policy_dt * np.arange(num_steps, dtype=np.float64)
    ).reshape(-1)


def normalized_saturation_huber(computed_torque: torch.Tensor, effort_limits: torch.Tensor) -> torch.Tensor:
    """Mean Huber penalty for torque demand above each joint's effort limit."""
    excess_ratio = (computed_torque.abs() / effort_limits.clamp_min(1.0e-6) - 1.0).clamp_min(0.0)
    huber = torch.where(excess_ratio <= 1.0, 0.5 * excess_ratio.square(), excess_ratio - 0.5)
    return huber.mean(dim=-1)


def compose_task_reward(
    velocity: torch.Tensor,
    upright: torch.Tensor,
    height: torch.Tensor,
    action_rate: torch.Tensor,
    saturation: torch.Tensor,
    *,
    velocity_weight: float,
    upright_weight: float,
    height_weight: float,
    action_rate_weight: float,
    saturation_weight: float,
) -> torch.Tensor:
    """Compose the Dextra-aligned raw task reward."""
    return (
        velocity_weight * velocity
        + upright_weight * upright
        + height_weight * height
        - action_rate_weight * action_rate
        - saturation_weight * saturation
    )
