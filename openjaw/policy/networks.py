"""MLP Actor-Critic policy networks for articulatory control.

Architecture (from PRD-C):
  Actor:  obs_dim → 256 → 256 → 13 (tanh output)
  Critic: obs_dim → 256 → 256 → 1

Total parameters < 1M. Orthogonal initialization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _init_orthogonal(module: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Apply orthogonal initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class ActorNetwork(nn.Module):
    """MLP Actor: maps observations to 13-DOF action means.

    Architecture: obs_dim → 256 → 256 → action_dim (tanh)
    Output range: [-1, 1], scaled to action bounds externally.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 13,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.apply(lambda m: _init_orthogonal(m, gain=np.sqrt(2)))
        # Output layer uses smaller gain for tanh
        _init_orthogonal(self.net[-2], gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: obs → action means in [-1, 1]."""
        return self.net(obs)


class CriticNetwork(nn.Module):
    """MLP Critic: maps observations to state value estimate.

    Architecture: obs_dim → 256 → 256 → 1
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim

        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self.apply(lambda m: _init_orthogonal(m, gain=np.sqrt(2)))
        # Output layer uses gain=1
        _init_orthogonal(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: obs → scalar value estimate."""
        return self.net(obs)


class ActorCritic(nn.Module):
    """Combined Actor-Critic for PPO.

    Separate networks (no shared backbone) following SB3 convention.
    Includes learnable log_std for Gaussian action distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 13,
        hidden_sizes: tuple[int, ...] = (256, 256),
        initial_log_std: float = -0.36,  # ~0.7 std
    ) -> None:
        super().__init__()
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_sizes)
        self.critic = CriticNetwork(obs_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: obs → (action_mean, value)."""
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value

    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Get the action distribution for sampling."""
        action_mean = self.actor(obs)
        std = self.log_std.exp()
        return torch.distributions.Normal(action_mean, std)

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
