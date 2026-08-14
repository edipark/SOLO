"""State estimators and policy-distillation student models."""

from __future__ import annotations

import torch
from torch import nn


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
    def __init__(self, input_dim: int = 58, output_dim: int = 9, hidden_size: int = 256, num_layers: int = 2):
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
    def __init__(self, input_dim: int = 58, output_dim: int = 9, channels: tuple[int, ...] = (64, 128, 128), dropout: float = 0.1):
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
    def __init__(self, input_dim: int = 58, output_dim: int = 9, hidden_size: int = 256):
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


class VanillaStudent(nn.Module):
    """Default 58-D all-joint to 29-D action distillation student."""

    def __init__(self, input_dim: int = 58, action_dim: int = 29):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def set_normalization(self, inputs: torch.Tensor) -> None:
        self.input_mean.copy_(inputs.mean(dim=0))
        self.input_std.copy_(inputs.std(dim=0).clamp_min(1.0e-6))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.input_mean) / self.input_std.clamp_min(1.0e-6)
        return self.network(normalized)


def build_estimator(
    estimator_type: str,
    input_dim: int = 58,
    output_dim: int = 9,
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
