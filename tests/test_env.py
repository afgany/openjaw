"""Steps 3-4: MuJoCo environment tests."""

import gymnasium as gym
import numpy as np
import pytest

from openjaw.core.types import FRAME_STACK_K, NUM_DOF
from openjaw.env.mouth_env import MouthEnv
from openjaw.env.oral_cavity import OralCavityModel


class TestMuJoCoLoads:
    def test_model_loads(self):
        """MuJoCo XML loads without error."""
        model = OralCavityModel()
        assert model.model is not None
        assert model.data is not None

    def test_actuator_count(self):
        """Model has 12 physical actuators (vocal loudness is virtual)."""
        model = OralCavityModel()
        assert model.model.nu == 12  # MuJoCo actuators
        assert model.num_actuators == 13  # Total DOF including virtual
        assert model.num_physical_actuators == 12

    def test_joint_count(self):
        """Model has 12 joints."""
        model = OralCavityModel()
        assert model.model.njnt == 12

    def test_sensor_count(self):
        """Model has 24 sensors (12 pos + 12 vel)."""
        model = OralCavityModel()
        assert model.model.nsensor == 24


class TestSimulationStability:
    def test_random_actions_no_nan(self):
        """1000 random-action steps without NaN or divergence."""
        model = OralCavityModel()
        model.reset()
        rng = np.random.default_rng(42)

        for _ in range(1000):
            action = rng.uniform(-0.5, 0.5, size=NUM_DOF).astype(np.float32)
            model.step(action, n_substeps=20)
            assert not model.has_nan(), "Simulation diverged (NaN detected)"

    def test_zero_actions_stable(self):
        """Zero actions produce stable simulation."""
        model = OralCavityModel()
        model.reset()
        action = np.zeros(NUM_DOF, dtype=np.float32)

        for _ in range(100):
            model.step(action)
            assert not model.has_nan()


class TestStateExtraction:
    def test_positions_shape(self):
        model = OralCavityModel()
        model.reset()
        pos = model.get_positions()
        assert pos.shape == (13,)
        assert pos.dtype == np.float32

    def test_velocities_shape(self):
        model = OralCavityModel()
        model.reset()
        vel = model.get_velocities()
        assert vel.shape == (13,)
        assert vel.dtype == np.float32

    def test_state_shape(self):
        model = OralCavityModel()
        model.reset()
        state = model.get_state()
        assert state.shape == (26,)  # 13 pos + 13 vel (prev_action added by env)

    def test_reset_returns_zeros(self):
        """After reset, positions and velocities should be near zero."""
        model = OralCavityModel()
        model.reset()
        pos = model.get_positions()
        vel = model.get_velocities()
        assert np.allclose(pos, 0, atol=1e-6)
        assert np.allclose(vel, 0, atol=1e-6)

    def test_action_changes_state(self):
        """Non-zero actions should change state."""
        model = OralCavityModel()
        model.reset()
        state_before = model.get_state().copy()

        action = np.full(NUM_DOF, 0.3, dtype=np.float32)
        model.step(action)
        state_after = model.get_state()

        assert not np.allclose(state_before, state_after), "State should change after action"

    def test_vocal_loudness(self):
        """Vocal loudness is stored and retrievable."""
        model = OralCavityModel()
        model.reset()
        assert model.vocal_loudness == 0.0

        action = np.zeros(NUM_DOF, dtype=np.float32)
        action[12] = 0.4
        model.step(action)
        assert model.vocal_loudness == pytest.approx(0.4)


class TestGymnasiumInterface:
    def test_make_env(self):
        """Environment can be created via Gymnasium registry."""
        import openjaw.env  # noqa: F401 — triggers registration
        env = gym.make("OpenJaw-Mouth-v0")
        assert env is not None
        env.close()

    def test_direct_creation(self):
        env = MouthEnv()
        assert env.action_space.shape == (13,)
        assert env.observation_space.shape[0] == FRAME_STACK_K * 3 * NUM_DOF + 768
        env.close()

    def test_reset(self):
        env = MouthEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert "positions" in info
        env.close()

    def test_step(self):
        env = MouthEnv()
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "step" in info2
        assert info2["step"] == 1
        env.close()

    def test_episode_truncation(self):
        """Episode truncates after episode_length steps."""
        env = MouthEnv(episode_length=5)
        env.reset(seed=42)
        for i in range(5):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            if i < 4:
                assert not truncated
            else:
                assert truncated
        env.close()

    def test_action_clipping(self):
        """Out-of-bounds actions are clipped, not rejected."""
        env = MouthEnv()
        env.reset(seed=42)
        action = np.full(NUM_DOF, 5.0, dtype=np.float32)  # Way out of bounds
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape  # No error
        env.close()

    def test_vectorized(self):
        """16 parallel environments work correctly."""
        import openjaw.env  # noqa: F401
        envs = gym.make_vec("OpenJaw-Mouth-v0", num_envs=16)
        obs, info = envs.reset(seed=42)
        assert obs.shape == (16, FRAME_STACK_K * 3 * NUM_DOF + 768)
        actions = np.stack([envs.single_action_space.sample() for _ in range(16)])
        obs2, rewards, terminated, truncated, info2 = envs.step(actions)
        assert obs2.shape == (16, FRAME_STACK_K * 3 * NUM_DOF + 768)
        assert rewards.shape == (16,)
        envs.close()
