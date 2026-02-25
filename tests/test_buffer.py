"""Tests for RolloutBuffer (GAE, batching, reset)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openjaw.training.buffer import RolloutBuffer


class TestRolloutBufferBasics:
    def test_create(self):
        buf = RolloutBuffer(buffer_size=10, obs_dim=4, action_dim=2)
        assert buf.pos == 0
        assert not buf.full
        assert buf.observations.shape == (10, 4)
        assert buf.actions.shape == (10, 2)

    def test_add_single(self):
        buf = RolloutBuffer(buffer_size=5, obs_dim=3, action_dim=1)
        buf.add(
            obs=np.array([1.0, 2.0, 3.0]),
            action=np.array([0.5]),
            log_prob=-0.1,
            reward=1.0,
            value=0.8,
            done=False,
        )
        assert buf.pos == 1
        assert not buf.full

    def test_fill_buffer(self):
        buf = RolloutBuffer(buffer_size=3, obs_dim=2, action_dim=1)
        for i in range(3):
            buf.add(
                obs=np.array([float(i), float(i)]),
                action=np.array([0.1]),
                log_prob=-0.5,
                reward=1.0,
                value=0.5,
                done=(i == 2),
            )
        assert buf.pos == 3
        assert buf.full

    def test_reset(self):
        buf = RolloutBuffer(buffer_size=3, obs_dim=2, action_dim=1)
        for i in range(3):
            buf.add(
                obs=np.zeros(2),
                action=np.zeros(1),
                log_prob=0.0,
                reward=0.0,
                value=0.0,
                done=False,
            )
        assert buf.full
        buf.reset()
        assert buf.pos == 0
        assert not buf.full


class TestGAE:
    def _fill_buffer(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> RolloutBuffer:
        """Fill a small buffer with known values for GAE testing."""
        buf = RolloutBuffer(
            buffer_size=4, obs_dim=2, action_dim=1,
            gamma=gamma, gae_lambda=gae_lambda,
        )
        # 4 transitions: rewards=[1, 2, 3, 4], values=[0.5, 0.6, 0.7, 0.8], no dones
        for i in range(4):
            buf.add(
                obs=np.array([float(i), 0.0]),
                action=np.array([0.0]),
                log_prob=0.0,
                reward=float(i + 1),
                value=0.5 + 0.1 * i,
                done=False,
            )
        return buf

    def test_gae_shapes(self):
        buf = self._fill_buffer()
        buf.compute_returns_and_advantages(last_value=0.9)
        assert buf.advantages[:4].shape == (4,)
        assert buf.returns[:4].shape == (4,)

    def test_gae_returns_equal_advantages_plus_values(self):
        buf = self._fill_buffer()
        buf.compute_returns_and_advantages(last_value=0.9)
        np.testing.assert_allclose(
            buf.returns[:4],
            buf.advantages[:4] + buf.values[:4],
            atol=1e-6,
        )

    def test_gae_with_done(self):
        """Done=True should cut off bootstrapping."""
        buf = RolloutBuffer(buffer_size=3, obs_dim=2, action_dim=1, gamma=0.99, gae_lambda=0.95)
        buf.add(np.zeros(2), np.zeros(1), 0.0, reward=1.0, value=0.5, done=False)
        buf.add(np.zeros(2), np.zeros(1), 0.0, reward=2.0, value=0.6, done=True)
        buf.add(np.zeros(2), np.zeros(1), 0.0, reward=3.0, value=0.7, done=False)
        buf.compute_returns_and_advantages(last_value=0.8)

        # At t=1, done=True, so next_non_terminal=0 → GAE at t=1 = r_1 - V(s_1) = 2.0 - 0.6 = 1.4
        assert buf.advantages[1] == pytest.approx(1.4, abs=1e-5)

    def test_gae_gamma_zero(self):
        """With gamma=0, advantages = reward - value (TD(0))."""
        buf = self._fill_buffer(gamma=0.0, gae_lambda=0.95)
        buf.compute_returns_and_advantages(last_value=0.9)
        for t in range(4):
            expected = buf.rewards[t] - buf.values[t]
            assert buf.advantages[t] == pytest.approx(expected, abs=1e-5)


class TestBatching:
    def _make_full_buffer(self, size: int = 20) -> RolloutBuffer:
        buf = RolloutBuffer(buffer_size=size, obs_dim=4, action_dim=2)
        for i in range(size):
            buf.add(
                obs=np.random.randn(4).astype(np.float32),
                action=np.random.randn(2).astype(np.float32),
                log_prob=float(np.random.randn()),
                reward=float(np.random.randn()),
                value=float(np.random.randn()),
                done=(i == size - 1),
            )
        buf.compute_returns_and_advantages(last_value=0.0)
        return buf

    def test_batch_count(self):
        buf = self._make_full_buffer(20)
        batches = buf.get_batches(batch_size=8)
        # 20 / 8 = 3 batches (8, 8, 4)
        assert len(batches) == 3

    def test_batch_keys(self):
        buf = self._make_full_buffer(10)
        batches = buf.get_batches(batch_size=5)
        for batch in batches:
            assert set(batch.keys()) == {
                "obs", "actions", "old_log_probs",
                "advantages", "returns", "old_values",
            }

    def test_batch_tensors(self):
        buf = self._make_full_buffer(10)
        batches = buf.get_batches(batch_size=5)
        for batch in batches:
            assert isinstance(batch["obs"], torch.Tensor)
            assert batch["obs"].shape[1] == 4
            assert batch["actions"].shape[1] == 2

    def test_advantages_normalized(self):
        """Advantages in batches should be approximately normalized."""
        buf = self._make_full_buffer(100)
        batches = buf.get_batches(batch_size=100)
        adv = batches[0]["advantages"]
        assert adv.mean().abs() < 0.2  # roughly zero-mean
        assert adv.std() < 2.0  # roughly unit std

    def test_all_indices_covered(self):
        """All buffer entries should appear exactly once across batches."""
        buf = self._make_full_buffer(12)
        batches = buf.get_batches(batch_size=5)
        all_obs = torch.cat([b["obs"] for b in batches], dim=0)
        assert all_obs.shape[0] == 12
