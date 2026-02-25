"""Step 2: MDP formulation tests — verify all mathematical specs."""

import numpy as np

from openjaw.core.mdp import (
    ARTICULATOR_NAMES,
    ARTICULATORS,
    ActionSpace,
    MDPSpec,
    ObservationSpace,
    RewardSpec,
    StateSpace,
    TransitionSpec,
)
from openjaw.core.types import NUM_DOF
from openjaw.env.articulators import get_articulator_groups


class TestStateSpace:
    def test_dimensions(self):
        s = StateSpace()
        assert s.num_dof == 13
        assert s.positions_dim == 13
        assert s.velocities_dim == 13
        assert s.prev_action_dim == 13
        assert s.dim == 39  # 3 * 13

    def test_custom_dof(self):
        s = StateSpace(num_dof=18)
        assert s.dim == 54  # 3 * 18


class TestActionSpace:
    def test_dimensions(self):
        a = ActionSpace()
        assert a.dim == 13
        assert a.low == -0.5
        assert a.high == 0.5

    def test_sample_shape(self):
        a = ActionSpace()
        rng = np.random.default_rng(42)
        action = a.sample(rng)
        assert action.shape == (13,)
        assert action.dtype == np.float32

    def test_sample_bounds(self):
        a = ActionSpace()
        rng = np.random.default_rng(42)
        for _ in range(100):
            action = a.sample(rng)
            assert np.all(action >= a.low)
            assert np.all(action <= a.high)

    def test_clip(self):
        a = ActionSpace()
        too_high = np.ones(13, dtype=np.float32) * 2.0
        clipped = a.clip(too_high)
        assert np.allclose(clipped, 0.5)

        too_low = np.ones(13, dtype=np.float32) * -2.0
        clipped = a.clip(too_low)
        assert np.allclose(clipped, -0.5)


class TestObservationSpace:
    def test_dimensions(self):
        o = ObservationSpace()
        # frame_stack_k=15, state_dim=39, target_embed=256
        assert o.stacked_state_dim == 15 * 39  # 585
        assert o.dim == 15 * 39 + 768  # 1353

    def test_custom_target_dim(self):
        o = ObservationSpace(target_embed_dim=128)
        assert o.dim == 15 * 39 + 128  # custom override


class TestRewardSpec:
    def test_default_weights(self):
        r = RewardSpec()
        assert r.w_audio == 0.7
        assert r.w_visual == 0.3
        assert r.weights_sum == 1.0

    def test_penalties(self):
        r = RewardSpec()
        assert r.lambda_silence == 1.0
        assert r.lambda_smooth == 0.01
        assert r.lambda_energy == 0.001


class TestTransitionSpec:
    def test_timing(self):
        t = TransitionSpec()
        assert t.control_freq_hz == 25
        assert t.control_dt == 0.04  # 40ms
        assert t.physics_dt == 0.002  # 2ms
        assert t.episode_length == 50
        assert t.episode_duration_seconds == 2.0  # 50 * 40ms

    def test_discount(self):
        t = TransitionSpec()
        assert t.discount == 0.99


class TestArticulators:
    def test_count(self):
        assert len(ARTICULATORS) == 13
        assert len(ARTICULATOR_NAMES) == 13

    def test_indices_sequential(self):
        for i, art in enumerate(ARTICULATORS):
            assert art.index == i

    def test_names_unique(self):
        assert len(set(ARTICULATOR_NAMES)) == 13

    def test_voicing_is_last(self):
        assert ARTICULATORS[-1].name == "vocal_loudness"
        assert ARTICULATORS[-1].index == 12

    def test_groups_cover_all_dof(self):
        groups = get_articulator_groups()
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(NUM_DOF))


class TestMDPSpec:
    def test_complete_spec(self):
        mdp = MDPSpec()
        assert mdp.state.dim == 39
        assert mdp.action.dim == 13
        assert mdp.observation.dim == 15 * 39 + 768
        assert mdp.reward.weights_sum == 1.0
        assert mdp.transition.episode_duration_seconds == 2.0

    def test_frozen(self):
        """MDP spec should be immutable."""
        mdp = MDPSpec()
        try:
            mdp.state = StateSpace(num_dof=18)
            assert False, "Should not be able to mutate frozen dataclass"
        except AttributeError:
            pass
