"""State estimators and the iterative DAgger student model."""

from __future__ import annotations

import torch
from torch import nn


def dagger_beta(initial: float, decay: float, minimum: float, iteration: int) -> float:
    """Return the teacher-mixing coefficient after ``iteration`` decays."""
    if not 0.0 <= minimum <= initial <= 1.0:
        raise ValueError("DAgger beta values must satisfy 0 <= minimum <= initial <= 1")
    if not 0.0 < decay <= 1.0:
        raise ValueError("DAgger beta decay must be in (0, 1]")
    if iteration < 0:
        raise ValueError("DAgger iteration must be non-negative")
    return max(minimum, initial * decay**iteration)


class RunningNormalizer:
    """Numerically stable online normalizer used by the DAgger student."""

    def __init__(self, dim: int, device: torch.device | str, clip: float = 5.0, eps: float = 1.0e-6):
        self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
        self.var = torch.ones(dim, device=device, dtype=torch.float64)
        self.count = torch.tensor(1.0e-4, device=device, dtype=torch.float64)
        self.clip = float(clip)
        self.eps = float(eps)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        batch = values.double()
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        combined = self.var * self.count + batch_var * batch_count
        combined += delta.square() * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.var = combined / total
        self.count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.float()
        std = self.var.float().sqrt().clamp_min(self.eps)
        return ((values - mean) / std).clamp(-self.clip, self.clip)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.var.float().sqrt().clamp_min(self.eps) + self.mean.float()

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count.detach().cpu(),
            "clip": self.clip,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count.copy_(state["count"].to(self.count.device))
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))


class ReplayBuffer:
    """Fixed-capacity device-local ring buffer for DAgger pairs."""

    def __init__(self, capacity: int, observation_dim: int, action_dim: int, device: torch.device | str):
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self.observations = torch.empty((capacity, observation_dim), device=device)
        self.actions = torch.empty((capacity, action_dim), device=device)
        self.write_index = 0
        self.size = 0

    def add(self, observations: torch.Tensor, actions: torch.Tensor) -> None:
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("Replay observations and actions must have the same batch size")
        if observations.shape[0] >= self.capacity:
            observations = observations[-self.capacity :]
            actions = actions[-self.capacity :]
        count = observations.shape[0]
        first = min(count, self.capacity - self.write_index)
        self.observations[self.write_index : self.write_index + first] = observations[:first]
        self.actions[self.write_index : self.write_index + first] = actions[:first]
        remaining = count - first
        if remaining:
            self.observations[:remaining] = observations[first:]
            self.actions[:remaining] = actions[first:]
        self.write_index = (self.write_index + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        indices = torch.randint(self.size, (batch_size,), device=self.observations.device)
        return self.observations[indices], self.actions[indices]


class NormalizedEstimator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("output_mean", torch.zeros(output_dim))
        self.register_buffer("output_std", torch.ones(output_dim))

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.input_mean) / self.input_std.clamp_min(1.0e-6)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self(inputs) * self.output_std + self.output_mean

    def set_normalization(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        input_axes = tuple(range(inputs.ndim - 1))
        self.input_mean.copy_(inputs.mean(dim=input_axes))
        self.input_std.copy_(inputs.std(dim=input_axes).clamp_min(1.0e-6))
        self.output_mean.copy_(targets.mean(dim=0))
        self.output_std.copy_(targets.std(dim=0).clamp_min(1.0e-6))

    def normalized_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.output_mean) / self.output_std.clamp_min(1.0e-6)

    def config(self) -> dict:
        raise NotImplementedError


class LSTMStateEstimator(NormalizedEstimator):
    def __init__(self, input_dim: int = 58, output_dim: int = 43, hidden_size: int = 256, num_layers: int = 2):
        super().__init__(input_dim, output_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.normalize(inputs))
        return self.head(hidden[-1])

    def config(self) -> dict:
        return {"type": "LSTM", "input_dim": self.input_dim, "output_dim": self.output_dim, "hidden_size": self.hidden_size, "num_layers": self.num_layers}


class _CausalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.conv = nn.Conv1d(input_channels, output_channels, 3, dilation=dilation, padding=padding)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv(inputs)
        if self.padding:
            output = output[:, :, :-self.padding]
        return self.activation(self.dropout(output) + self.skip(inputs))


class TCNStateEstimator(NormalizedEstimator):
    def __init__(
        self,
        input_dim: int = 58,
        output_dim: int = 43,
        channels: tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, output_dim)
        layers: list[nn.Module] = []
        current = input_dim
        for index, channel in enumerate(channels):
            layers.append(_CausalBlock(current, channel, 2**index, dropout))
            current = channel
        self.channels = tuple(channels)
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(current, current // 2), nn.ReLU(), nn.Linear(current // 2, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.network(self.normalize(inputs).transpose(1, 2))
        return self.head(features[:, :, -1])

    def config(self) -> dict:
        return {"type": "TCN", "input_dim": self.input_dim, "output_dim": self.output_dim, "channels": self.channels}


class MLPStateEstimator(NormalizedEstimator):
    def __init__(self, input_dim: int = 58, output_dim: int = 43, hidden_size: int = 256):
        super().__init__(input_dim, output_dim)
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs[:, -1]
        return self.network(self.normalize(inputs))

    def config(self) -> dict:
        return {"type": "MLP", "input_dim": self.input_dim, "output_dim": self.output_dim, "hidden_size": self.hidden_size}


class DaggerStudent(nn.Module):
    """Deterministic Dextra-aligned student; normalization is external."""

    def __init__(
        self,
        input_dim: int = 58,
        action_dim: int = 29,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, hidden), nn.ELU()))
            current = hidden
        layers.append(nn.Linear(current, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def config(self) -> dict:
        return {
            "type": "DaggerStudent",
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
        }


def build_estimator(
    estimator_type: str,
    input_dim: int = 58,
    output_dim: int = 43,
    hidden_size: int = 256,
    num_layers: int = 2,
    tcn_channels: tuple[int, ...] = (64, 128, 128),
) -> NormalizedEstimator:
    estimator_type = estimator_type.upper()
    if estimator_type == "LSTM":
        return LSTMStateEstimator(input_dim, output_dim, hidden_size, num_layers)
    if estimator_type == "TCN":
        return TCNStateEstimator(input_dim, output_dim, tcn_channels)
    if estimator_type == "MLP":
        return MLPStateEstimator(input_dim, output_dim, hidden_size)
    raise ValueError(f"Unknown estimator type {estimator_type!r}; choose LSTM, TCN, or MLP")
