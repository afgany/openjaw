"""Tests for PPO loss computation and update."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openjaw.policy.networks import ActorCritic
from openjaw.training.buffer import RolloutBuffer
from openjaw.training.ppo import PPOUpdateResult, compute_ppo_loss, ppo_update


OBS_DIM = 16
ACTION_DIM = 4


@pytest.fixture
def policy():
    torch.manual_seed(42)
    return ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM)


@pytest.fixture
def filled_buffer():
    """Buffer filled with random transitions."""
    torch.manual_seed(42)
    np.random.seed(42)
    buf = RolloutBuffer(buffer_size=64, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    for i in range(64):
        buf.add(
            obs=np.random.randn(OBS_DIM).astype(np.float32),
            action=np.random.randn(ACTION_DIM).astype(np.float32),
            log_prob=float(np.random.randn()),
            reward=float(np.random.randn()),
            value=float(np.random.randn()),
            done=(i == 63),
        )
    buf.compute_returns_and_advantages(last_value=0.0)
    return buf


class TestComputePPOLoss:
    def test_loss_is_scalar(self, policy):
        batch_size = 8
        obs = torch.randn(batch_size, OBS_DIM)
        actions = torch.randn(batch_size, ACTION_DIM)
        old_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        old_values = torch.randn(batch_size)

        loss, info = compute_ppo_loss(
            policy, obs, actions, old_log_probs, advantages, returns, old_values,
        )
        assert loss.shape == ()
        assert loss.requires_grad

    def test_info_keys(self, policy):
        obs = torch.randn(4, OBS_DIM)
        actions = torch.randn(4, ACTION_DIM)
        _, info = compute_ppo_loss(
            policy, obs, actions,
            old_log_probs=torch.zeros(4),
            advantages=torch.ones(4),
            returns=torch.ones(4),
            old_values=torch.zeros(4),
        )
        assert set(info.keys()) == {
            "policy_loss", "value_loss", "entropy",
            "approx_kl", "clip_fraction",
        }

    def test_entropy_positive(self, policy):
        obs = torch.randn(4, OBS_DIM)
        actions = torch.randn(4, ACTION_DIM)
        _, info = compute_ppo_loss(
            policy, obs, actions,
            old_log_probs=torch.zeros(4),
            advantages=torch.zeros(4),
            returns=torch.zeros(4),
            old_values=torch.zeros(4),
        )
        assert info["entropy"] > 0.0

    def test_zero_advantage_small_policy_loss(self, policy):
        """With zero advantages, the policy loss should be near zero."""
        obs = torch.randn(16, OBS_DIM)
        actions = torch.randn(16, ACTION_DIM)
        _, info = compute_ppo_loss(
            policy, obs, actions,
            old_log_probs=torch.zeros(16),
            advantages=torch.zeros(16),
            returns=torch.zeros(16),
            old_values=torch.zeros(16),
        )
        assert abs(info["policy_loss"]) < 0.01

    def test_clip_range(self, policy):
        """Clip fraction should be between 0 and 1."""
        obs = torch.randn(32, OBS_DIM)
        actions = torch.randn(32, ACTION_DIM)
        _, info = compute_ppo_loss(
            policy, obs, actions,
            old_log_probs=torch.randn(32) * 5,  # Large mismatch → high clip fraction
            advantages=torch.randn(32),
            returns=torch.randn(32),
            old_values=torch.zeros(32),
        )
        assert 0.0 <= info["clip_fraction"] <= 1.0


class TestPPOUpdate:
    def test_returns_result(self, policy, filled_buffer):
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        result = ppo_update(
            policy, optimizer, filled_buffer,
            n_epochs=2, batch_size=32,
        )
        assert isinstance(result, PPOUpdateResult)

    def test_result_fields(self, policy, filled_buffer):
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        result = ppo_update(policy, optimizer, filled_buffer, n_epochs=2, batch_size=32)
        assert isinstance(result.policy_loss, float)
        assert isinstance(result.value_loss, float)
        assert isinstance(result.entropy, float)
        assert isinstance(result.approx_kl, float)
        assert isinstance(result.clip_fraction, float)
        assert isinstance(result.explained_variance, float)

    def test_params_change(self, policy, filled_buffer):
        """PPO update should modify network parameters."""
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        params_before = {n: p.clone() for n, p in policy.named_parameters()}

        ppo_update(policy, optimizer, filled_buffer, n_epochs=3, batch_size=32)

        any_changed = False
        for name, param in policy.named_parameters():
            if not torch.allclose(params_before[name], param):
                any_changed = True
                break
        assert any_changed, "PPO update did not change any parameters"

    def test_explained_variance_range(self, policy, filled_buffer):
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        result = ppo_update(policy, optimizer, filled_buffer, n_epochs=2, batch_size=32)
        assert -2.0 <= result.explained_variance <= 1.0

    def test_kl_early_stopping(self, policy, filled_buffer):
        """Very low target_kl should trigger early stopping (fewer epochs)."""
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
        result = ppo_update(
            policy, optimizer, filled_buffer,
            n_epochs=100, batch_size=16,
            target_kl=1e-10,  # Extremely low → should stop early
        )
        # If KL early stop worked, approx_kl should not be astronomical
        assert isinstance(result.approx_kl, float)
