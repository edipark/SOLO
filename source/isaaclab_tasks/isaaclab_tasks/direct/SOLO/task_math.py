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
