"""Step 9: Policy network tests."""

import torch

from openjaw.core.mdp import ObservationSpace
from openjaw.policy.networks import ActorCritic, ActorNetwork, CriticNetwork


OBS_DIM = ObservationSpace().dim  # 841
ACTION_DIM = 13
BATCH_SIZE = 32


class TestActorNetwork:
    def test_output_shape(self):
        actor = ActorNetwork(OBS_DIM, ACTION_DIM)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        out = actor(obs)
        assert out.shape == (BATCH_SIZE, ACTION_DIM)

    def test_output_range(self):
        """Tanh output should be in [-1, 1]."""
        actor = ActorNetwork(OBS_DIM, ACTION_DIM)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        out = actor(obs)
        assert (out >= -1.0).all()
        assert (out <= 1.0).all()

    def test_single_observation(self):
        actor = ActorNetwork(OBS_DIM, ACTION_DIM)
        obs = torch.randn(1, OBS_DIM)
        out = actor(obs)
        assert out.shape == (1, ACTION_DIM)


class TestCriticNetwork:
    def test_output_shape(self):
        critic = CriticNetwork(OBS_DIM)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        out = critic(obs)
        assert out.shape == (BATCH_SIZE, 1)

    def test_scalar_output(self):
        critic = CriticNetwork(OBS_DIM)
        obs = torch.randn(1, OBS_DIM)
        out = critic(obs)
        assert out.shape == (1, 1)


class TestActorCritic:
    def test_forward(self):
        ac = ActorCritic(OBS_DIM, ACTION_DIM)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        action_mean, value = ac(obs)
        assert action_mean.shape == (BATCH_SIZE, ACTION_DIM)
        assert value.shape == (BATCH_SIZE, 1)

    def test_param_count_under_1m(self):
        """Total parameters must be < 1M."""
        ac = ActorCritic(OBS_DIM, ACTION_DIM)
        count = ac.param_count()
        assert count < 1_000_000, f"Param count {count} exceeds 1M limit"

    def test_action_distribution(self):
        ac = ActorCritic(OBS_DIM, ACTION_DIM)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        dist = ac.get_action_distribution(obs)
        sample = dist.sample()
        assert sample.shape == (BATCH_SIZE, ACTION_DIM)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (BATCH_SIZE, ACTION_DIM)

    def test_initial_std(self):
        """Initial std should be approximately 0.7."""
        ac = ActorCritic(OBS_DIM, ACTION_DIM, initial_log_std=-0.36)
        std = ac.log_std.exp()
        assert torch.allclose(std, torch.tensor(0.7), atol=0.05)

    def test_gradient_flow(self):
        """Gradients should flow through actor, critic, and log_std."""
        ac = ActorCritic(OBS_DIM, ACTION_DIM)
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        # Use distribution path so log_std gets gradients
        dist = ac.get_action_distribution(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = ac.critic(obs)
        loss = -log_prob.mean() + value.mean()
        loss.backward()
        for name, param in ac.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
