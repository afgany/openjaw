"""Step 12: Evaluation metrics tests."""

from __future__ import annotations

import numpy as np
import pytest

from openjaw.evaluation.metrics import (
    _levenshtein_distance,
    lip_vertex_error,
    mel_cepstral_distortion,
    sylber_cosine_similarity,
)


# ── MCD ─────────────────────────────────────────────────────────────────────

class TestMelCepstralDistortion:
    def _make_audio(self, freq: float = 200.0, duration: float = 0.5) -> np.ndarray:
        t = np.arange(int(16000 * duration)) / 16000
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_identical_audio_low_mcd(self):
        audio = self._make_audio()
        mcd = mel_cepstral_distortion(audio, audio)
        assert mcd < 1.0  # Should be near 0

    def test_different_audio_higher_mcd(self):
        a1 = self._make_audio(freq=200.0)
        a2 = self._make_audio(freq=800.0)
        mcd = mel_cepstral_distortion(a1, a2)
        assert mcd > 0.0

    def test_mcd_nonnegative(self):
        rng = np.random.default_rng(42)
        a1 = (rng.standard_normal(16000) * 0.3).astype(np.float32)
        a2 = (rng.standard_normal(16000) * 0.3).astype(np.float32)
        mcd = mel_cepstral_distortion(a1, a2)
        assert mcd >= 0.0

    def test_mcd_returns_float(self):
        audio = self._make_audio()
        mcd = mel_cepstral_distortion(audio, audio)
        assert isinstance(mcd, float)

    def test_different_lengths_aligned(self):
        a1 = self._make_audio(duration=0.5)
        a2 = self._make_audio(duration=1.0)
        mcd = mel_cepstral_distortion(a1, a2)
        assert isinstance(mcd, float)
        assert mcd >= 0.0

    def test_empty_audio_still_computes(self):
        """Empty audio should still return a finite float (librosa pads internally)."""
        empty = np.array([], dtype=np.float32)
        audio = self._make_audio()
        mcd = mel_cepstral_distortion(empty, audio)
        assert isinstance(mcd, float)
        assert mcd >= 0.0


# ── LVE ─────────────────────────────────────────────────────────────────────

class TestLipVertexError:
    def test_identical_vertices_zero_error(self):
        verts = np.random.randn(5, 3).astype(np.float32)
        lve = lip_vertex_error(verts, verts)
        assert abs(lve) < 1e-6

    def test_known_distance(self):
        v1 = np.zeros((1, 3), dtype=np.float32)
        v2 = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        lve = lip_vertex_error(v1, v2)
        assert lve == pytest.approx(5.0, abs=1e-5)

    def test_3d_input(self):
        """Test sequence input (T, N, 3)."""
        v1 = np.random.randn(10, 5, 3).astype(np.float32)
        lve = lip_vertex_error(v1, v1)
        assert abs(lve) < 1e-6

    def test_3d_nonzero_error(self):
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal((10, 5, 3)).astype(np.float32) * 0.01
        v2 = rng.standard_normal((10, 5, 3)).astype(np.float32) * 0.01
        lve = lip_vertex_error(v1, v2)
        assert lve > 0.0

    def test_shape_mismatch_raises(self):
        v1 = np.zeros((3, 3), dtype=np.float32)
        v2 = np.zeros((4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            lip_vertex_error(v1, v2)

    def test_invalid_ndim_raises(self):
        v1 = np.zeros((2,), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            lip_vertex_error(v1, v1)

    def test_returns_float(self):
        verts = np.random.randn(3, 3).astype(np.float32)
        assert isinstance(lip_vertex_error(verts, verts), float)


# ── Sylber cosine similarity ────────────────────────────────────────────────

class TestSylberCosineSimilarity:
    def test_identical_vectors(self):
        v = np.random.randn(768).astype(np.float32)
        v /= np.linalg.norm(v)
        sim = sylber_cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = np.random.randn(768).astype(np.float32)
        v /= np.linalg.norm(v)
        sim = sylber_cosine_similarity(v, -v)
        assert sim == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        v1 = np.zeros(768, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(768, dtype=np.float32)
        v2[1] = 1.0
        sim = sylber_cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_zero_vector_returns_zero(self):
        z = np.zeros(768, dtype=np.float32)
        v = np.random.randn(768).astype(np.float32)
        sim = sylber_cosine_similarity(z, v)
        assert sim == 0.0

    def test_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal(768).astype(np.float32)
            b = rng.standard_normal(768).astype(np.float32)
            sim = sylber_cosine_similarity(a, b)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


# ── Levenshtein distance ───────────────────────────────────────────────────

class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "xyz") == 3

    def test_single_insertion(self):
        assert _levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        assert _levenshtein_distance("cats", "cat") == 1

    def test_substitution(self):
        assert _levenshtein_distance("cat", "bat") == 1

    def test_known_distance(self):
        assert _levenshtein_distance("kitten", "sitting") == 3
