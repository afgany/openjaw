"""Rollout buffer for on-policy PPO data collection.

Stores transitions (obs, action, log_prob, reward, value, done) for one
rollout, then computes GAE advantages and discounted returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Fixed-size buffer for one rollout of on-policy experience.

    Stores numpy arrays during collection, converts to torch tensors
    for PPO update. Computes GAE-lambda advantages in-place.
    """

    buffer_size: int
    obs_dim: int
    action_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"

    # Internal storage (initialized in __post_init__)
    observations: np.ndarray = field(init=False, repr=False)
    actions: np.ndarray = field(init=False, repr=False)
    log_probs: np.ndarray = field(init=False, repr=False)
    rewards: np.ndarray = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    dones: np.ndarray = field(init=False, repr=False)
    advantages: np.ndarray = field(init=False, repr=False)
    returns: np.ndarray = field(init=False, repr=False)
    pos: int = field(init=False, default=0)
    full: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = float(done)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute GAE advantages and discounted returns.

        GAE(gamma, lambda):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_l
            G_t = A_t + V(s_t)
        """
        last_gae = 0.0
        for t in reversed(range(self.pos)):
            if t == self.pos - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[: self.pos] = self.advantages[: self.pos] + self.values[: self.pos]

    def get_batches(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches as torch tensors."""
        size = self.pos

        # Normalize advantages
        adv = self.advantages[:size].copy()
        adv_std = adv.std()
        if adv_std > 1e-8:
            adv = (adv - adv.mean()) / adv_std

        indices = np.random.permutation(size)
        batches = []
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            idx = indices[start:end]
            batches.append({
                "obs": torch.as_tensor(self.observations[idx], device=self.device),
                "actions": torch.as_tensor(self.actions[idx], device=self.device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx], device=self.device),
                "advantages": torch.as_tensor(adv[idx], device=self.device),
                "returns": torch.as_tensor(self.returns[idx], device=self.device),
                "old_values": torch.as_tensor(self.values[idx], device=self.device),
            })
        return batches

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self.pos = 0
        self.full = False
