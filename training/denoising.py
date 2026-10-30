"""SB3-compatible feature denoising layers."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualDenoisingFeatureExtractor(BaseFeaturesExtractor):
    """Residual MLP denoiser that preserves the original observation dimension.

    The extractor is intentionally shaped as ``obs -> obs`` so the downstream
    SB3 actor/critic MLP can be copied from an existing policy and optionally
    fine-tuned later with standard PPO.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        hidden_dim: int = 256,
        depth: int = 2,
        residual_scale: float = 1.0,
    ):
        if observation_space.shape is None or len(observation_space.shape) != 1:
            raise ValueError("ResidualDenoisingFeatureExtractor expects a flat 1D observation space")
        features_dim = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=features_dim)
        self.residual_scale = float(residual_scale)
        hidden_dim = int(hidden_dim)
        depth = max(1, int(depth))

        layers: list[torch.nn.Module] = []
        in_dim = features_dim
        for _ in range(depth):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, features_dim))
        self.denoiser = torch.nn.Sequential(*layers)
        self.reset_identity()

    def reset_identity(self) -> None:
        """Start as an exact identity mapping."""

        last = self.denoiser[-1]
        if isinstance(last, torch.nn.Linear):
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations + self.residual_scale * self.denoiser(observations)
