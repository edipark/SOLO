"""Policy adapters that make AMP and PPO share the estimator pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..schema import AMP_OBSERVATION_SCHEMA, PPO_OBSERVATION_SCHEMA, ObservationSchema, joint_indices
from ..skrl_compat import deterministic_action
from ..task_math import inject_observation_estimate


def unwrap_direct_env(env):
    current = env
    for _ in range(32):
        candidate = getattr(current, "unwrapped", current)
        if hasattr(candidate, "get_estimator_joint_state") and hasattr(candidate, "get_estimator_target"):
            return candidate
        next_env = getattr(current, "_env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Environment does not implement the SOLO estimator interface")


class PolicyAdapter(ABC):
    schema: ObservationSchema

    def __init__(self, env, joint_preset: str = "all"):
        self.env = env
        self.core_env = unwrap_direct_env(env)
        _, _, names = self.core_env.get_estimator_joint_state()
        self.joint_preset = joint_preset
        self.joint_ids = joint_indices(names, joint_preset)

    @property
    def input_dim(self) -> int:
        return 2 * len(self.joint_ids)

    def estimator_input(self) -> torch.Tensor:
        # Deliberately use the simulator-provided joint velocity. There is no
        # finite-difference, encoder quantization, EMA, or hardware-noise path.
        joint_pos, joint_vel, _ = self.core_env.get_estimator_joint_state()
        ids = torch.as_tensor(self.joint_ids, device=joint_pos.device)
        return torch.cat((joint_pos.index_select(1, ids), joint_vel.index_select(1, ids)), dim=-1)

    def estimator_target(self) -> torch.Tensor:
        target = self.core_env.get_estimator_target()
        if target.shape[-1] != self.schema.estimator_target_dim:
            raise RuntimeError(
                f"Adapter expected target dim {self.schema.estimator_target_dim}, got {target.shape[-1]}"
            )
        return target

    def inject_estimate(self, observations: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        if estimate.shape[-1] != self.schema.estimator_target_dim:
            raise ValueError("Estimator output does not match the policy schema")
        return inject_observation_estimate(observations, estimate, self.schema.estimator_target_indices)

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def action(self, agent: Any, observations: torch.Tensor) -> torch.Tensor:
        states = self.env.state() if hasattr(self.env, "state") else None
        return deterministic_action(agent, observations, states)


class AmpPolicyAdapter(PolicyAdapter):
    schema = AMP_OBSERVATION_SCHEMA

    def name(self) -> str:
        return "amp"


class PpoPolicyAdapter(PolicyAdapter):
    schema = PPO_OBSERVATION_SCHEMA

    def name(self) -> str:
        return "ppo"


def make_policy_adapter(kind: str, env, joint_preset: str = "all") -> PolicyAdapter:
    kind = kind.lower()
    if kind == "amp":
        return AmpPolicyAdapter(env, joint_preset)
    if kind == "ppo":
        return PpoPolicyAdapter(env, joint_preset)
    raise ValueError(f"Unknown policy adapter {kind!r}; choose amp or ppo")
