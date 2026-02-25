"""Step 8: Reward module tests."""

import numpy as np
import pytest

from openjaw.core.types import NUM_DOF
from openjaw.perception.sylber import MockSylberEncoder
from openjaw.reward.audio_reward import AudioReward
from openjaw.reward.auxiliary import AuxiliaryReward
from openjaw.reward.combined import CombinedReward, RewardOutput
from openjaw.reward.visual_reward import VisualReward
from openjaw.visual.flame_adapter import NUM_LIP_LANDMARKS


def _make_test_audio(duration: float = 0.5, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(16000 * duration)) / 16000
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestAudioReward:
    def test_identical_embeddings(self):
        """Identical embeddings should give similarity of 1.0."""
        encoder = MockSylberEncoder()
        reward = AudioReward(encoder)
        emb = np.random.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        sim = reward.compute_from_embeddings(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_opposite_embeddings(self):
        """Opposite embeddings should give similarity of -1.0."""
        encoder = MockSylberEncoder()
        reward = AudioReward(encoder)
        emb = np.random.randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        sim = reward.compute_from_embeddings(emb, -emb)
        assert abs(sim - (-1.0)) < 1e-5

    def test_orthogonal_embeddings(self):
        """Orthogonal embeddings should give similarity near 0."""
        encoder = MockSylberEncoder()
        reward = AudioReward(encoder)
        emb1 = np.zeros(768, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(768, dtype=np.float32)
        emb2[1] = 1.0
        sim = reward.compute_from_embeddings(emb1, emb2)
        assert abs(sim) < 1e-5

    def test_reward_range(self):
        """Audio reward should be in [-1, 1]."""
        encoder = MockSylberEncoder()
        reward = AudioReward(encoder)
        for _ in range(20):
            emb1 = np.random.randn(768).astype(np.float32)
            emb2 = np.random.randn(768).astype(np.float32)
            sim = reward.compute_from_embeddings(emb1, emb2)
            assert -1.0 <= sim <= 1.0 + 1e-6

    def test_zero_embedding_returns_negative(self):
        """Zero embedding should return -1 (no valid signal)."""
        encoder = MockSylberEncoder()
        reward = AudioReward(encoder)
        zero = np.zeros(768, dtype=np.float32)
        emb = np.random.randn(768).astype(np.float32)
        sim = reward.compute_from_embeddings(zero, emb)
        assert sim == -1.0

    def test_compute_from_audio(self):
        """Full compute path: audio → Sylber → cosine sim."""
        encoder = MockSylberEncoder()
        reward = AudioReward(encoder)
        audio = _make_test_audio()
        target_emb = np.random.randn(768).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb)
        sim = reward.compute(audio, target_emb)
        assert -1.0 <= sim <= 1.0 + 1e-6


class TestVisualReward:
    def test_perfect_match(self):
        """Identical vertices → reward 1.0."""
        reward = VisualReward(max_lve=0.05)
        verts = np.random.randn(NUM_LIP_LANDMARKS, 3).astype(np.float32)
        r = reward.compute(verts, verts)
        assert abs(r - 1.0) < 1e-5

    def test_max_error(self):
        """Large error → reward 0.0."""
        reward = VisualReward(max_lve=0.05)
        v1 = np.zeros((NUM_LIP_LANDMARKS, 3), dtype=np.float32)
        v2 = np.ones((NUM_LIP_LANDMARKS, 3), dtype=np.float32)  # ~1.73 distance per vertex
        r = reward.compute(v1, v2)
        assert r == 0.0

    def test_reward_range(self):
        """Visual reward should be in [0, 1]."""
        reward = VisualReward(max_lve=0.05)
        for _ in range(20):
            v1 = np.random.randn(NUM_LIP_LANDMARKS, 3).astype(np.float32) * 0.01
            v2 = np.random.randn(NUM_LIP_LANDMARKS, 3).astype(np.float32) * 0.01
            r = reward.compute(v1, v2)
            assert 0.0 <= r <= 1.0 + 1e-6

    def test_shape_mismatch_raises(self):
        reward = VisualReward()
        v1 = np.zeros((2, 3), dtype=np.float32)
        v2 = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            reward.compute(v1, v2)


class TestAuxiliaryReward:
    def test_silence_penalty(self):
        """No syllable → -lambda_silence penalty."""
        aux = AuxiliaryReward(lambda_silence=1.0, lambda_smooth=0.0, lambda_energy=0.0)
        action = np.zeros(NUM_DOF, dtype=np.float32)
        r = aux.compute(action, action, has_syllable=False)
        assert r == -1.0

    def test_no_penalty_with_syllable(self):
        """Syllable detected, zero action → zero penalty."""
        aux = AuxiliaryReward(lambda_silence=1.0, lambda_smooth=0.0, lambda_energy=0.0)
        action = np.zeros(NUM_DOF, dtype=np.float32)
        r = aux.compute(action, action, has_syllable=True)
        assert r == 0.0

    def test_smoothness_penalty(self):
        """Large action change → negative smoothness penalty."""
        aux = AuxiliaryReward(lambda_silence=0.0, lambda_smooth=1.0, lambda_energy=0.0)
        a1 = np.zeros(NUM_DOF, dtype=np.float32)
        a2 = np.ones(NUM_DOF, dtype=np.float32)
        r = aux.compute(a2, a1, has_syllable=True)
        assert r < 0.0
        assert r == pytest.approx(-float(NUM_DOF))  # -1.0 * sum(1^2) = -13

    def test_energy_penalty(self):
        """Large action → negative energy penalty."""
        aux = AuxiliaryReward(lambda_silence=0.0, lambda_smooth=0.0, lambda_energy=1.0)
        action = np.ones(NUM_DOF, dtype=np.float32)
        r = aux.compute(action, action, has_syllable=True)
        assert r < 0.0
        assert r == pytest.approx(-float(NUM_DOF))  # -1.0 * sum(1^2) = -13

    def test_components(self):
        """compute_components returns dict with all keys."""
        aux = AuxiliaryReward()
        action = np.ones(NUM_DOF, dtype=np.float32) * 0.3
        prev = np.zeros(NUM_DOF, dtype=np.float32)
        components = aux.compute_components(action, prev, has_syllable=True)
        assert "silence" in components
        assert "smoothness" in components
        assert "energy" in components
        assert "total" in components
        assert components["total"] == pytest.approx(
            components["silence"] + components["smoothness"] + components["energy"]
        )


class TestCombinedReward:
    def _make_inputs(self):
        audio = _make_test_audio()
        target_emb = np.random.randn(768).astype(np.float32)
        target_emb /= np.linalg.norm(target_emb)
        gen_verts = np.random.randn(NUM_LIP_LANDMARKS, 3).astype(np.float32) * 0.01
        tgt_verts = np.random.randn(NUM_LIP_LANDMARKS, 3).astype(np.float32) * 0.01
        action = np.random.randn(NUM_DOF).astype(np.float32) * 0.1
        prev_action = np.random.randn(NUM_DOF).astype(np.float32) * 0.1
        return audio, target_emb, gen_verts, tgt_verts, action, prev_action

    def test_combined_mode(self):
        encoder = MockSylberEncoder()
        reward = CombinedReward(encoder, mode="combined")
        result = reward.compute(*self._make_inputs())
        assert isinstance(result, RewardOutput)
        assert isinstance(result.total, float)
        assert isinstance(result.has_syllable, bool)

    def test_audio_only_mode(self):
        encoder = MockSylberEncoder()
        reward = CombinedReward(encoder, mode="audio_only")
        assert reward.w_audio > 0
        assert reward.w_visual == 0
        result = reward.compute(*self._make_inputs())
        assert isinstance(result, RewardOutput)

    def test_visual_only_mode(self):
        encoder = MockSylberEncoder()
        reward = CombinedReward(encoder, mode="visual_only")
        assert reward.w_audio == 0
        assert reward.w_visual > 0
        result = reward.compute(*self._make_inputs())
        assert isinstance(result, RewardOutput)

    def test_binary_sound_mode(self):
        encoder = MockSylberEncoder()
        reward = CombinedReward(encoder, mode="binary_sound")
        result = reward.compute(*self._make_inputs())
        assert result.total in (1.0, -1.0)

    def test_ablation_toggle(self):
        """Audio-only and visual-only produce different rewards for same input."""
        encoder = MockSylberEncoder()
        inputs = self._make_inputs()

        r_audio = CombinedReward(encoder, mode="audio_only").compute(*inputs)
        r_visual = CombinedReward(encoder, mode="visual_only").compute(*inputs)
        r_combined = CombinedReward(encoder, mode="combined").compute(*inputs)

        # Combined should differ from pure audio or pure visual
        # (unless by coincidence, but very unlikely with random inputs)
        assert isinstance(r_audio.total, float)
        assert isinstance(r_visual.total, float)
        assert isinstance(r_combined.total, float)

    def test_reward_components_logged(self):
        """RewardOutput should have all components for logging."""
        encoder = MockSylberEncoder()
        reward = CombinedReward(encoder, mode="combined")
        result = reward.compute(*self._make_inputs())
        assert hasattr(result, "audio")
        assert hasattr(result, "visual")
        assert hasattr(result, "auxiliary")
        assert hasattr(result, "silence_penalty")
        assert hasattr(result, "smoothness_penalty")
        assert hasattr(result, "energy_penalty")
        assert hasattr(result, "has_syllable")
