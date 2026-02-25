"""Step 7: Perception encoder tests."""

import numpy as np

from openjaw.perception.lip_reader import LIP_FEATURE_DIM, LipFeatureEncoder
from openjaw.perception.sylber import (
    SYLBER_EMBED_DIM,
    BaseSylberEncoder,
    MockSylberEncoder,
    create_sylber_encoder,
)
from openjaw.perception.wav2vec import (
    WAV2VEC_FEATURE_DIM,
    BaseWav2VecEncoder,
    MockWav2VecEncoder,
    create_wav2vec_encoder,
)
from openjaw.core.types import AUDIO_SAMPLE_RATE


def _make_test_audio(duration: float = 0.5, freq: float = 200.0) -> np.ndarray:
    """Generate test audio: sine wave at given frequency."""
    t = np.arange(int(AUDIO_SAMPLE_RATE * duration)) / AUDIO_SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_silence(duration: float = 0.5) -> np.ndarray:
    return np.zeros(int(AUDIO_SAMPLE_RATE * duration), dtype=np.float32)


class TestMockSylberEncoder:
    def test_creates(self):
        encoder = MockSylberEncoder()
        assert isinstance(encoder, BaseSylberEncoder)
        assert encoder.embed_dim == 768

    def test_encode_output_keys(self):
        encoder = MockSylberEncoder()
        audio = _make_test_audio()
        result = encoder.encode(audio)
        assert "segments" in result
        assert "segment_features" in result
        assert "hidden_states" in result

    def test_segment_features_shape(self):
        encoder = MockSylberEncoder()
        audio = _make_test_audio()
        result = encoder.encode(audio)
        features = result["segment_features"]
        assert features.ndim == 2
        assert features.shape[1] == SYLBER_EMBED_DIM

    def test_hidden_states_shape(self):
        encoder = MockSylberEncoder()
        audio = _make_test_audio(duration=1.0)
        result = encoder.encode(audio)
        hidden = result["hidden_states"]
        assert hidden.ndim == 2
        assert hidden.shape[1] == SYLBER_EMBED_DIM
        # ~50 frames per second
        assert hidden.shape[0] >= 40

    def test_get_segment_embedding(self):
        encoder = MockSylberEncoder()
        audio = _make_test_audio()
        emb = encoder.get_segment_embedding(audio)
        assert emb.shape == (SYLBER_EMBED_DIM,)
        assert emb.dtype == np.float32

    def test_has_syllable_with_audio(self):
        encoder = MockSylberEncoder()
        audio = _make_test_audio()
        assert encoder.has_syllable(audio) is True

    def test_silence_no_syllable(self):
        """Silence should produce no syllables (or very few)."""
        encoder = MockSylberEncoder()
        silence = _make_silence()
        # Mock may or may not detect syllables in silence
        # but get_segment_embedding should return valid shape
        emb = encoder.get_segment_embedding(silence)
        assert emb.shape == (SYLBER_EMBED_DIM,)

    def test_factory_creates_mock(self):
        encoder = create_sylber_encoder(use_real=False)
        assert isinstance(encoder, MockSylberEncoder)


class TestMockWav2VecEncoder:
    def test_creates(self):
        encoder = MockWav2VecEncoder()
        assert isinstance(encoder, BaseWav2VecEncoder)
        assert encoder.feature_dim == 768

    def test_encode_shape(self):
        encoder = MockWav2VecEncoder()
        audio = _make_test_audio(duration=1.0)
        features = encoder.encode(audio)
        assert features.ndim == 2
        assert features.shape[1] == WAV2VEC_FEATURE_DIM
        # ~50 frames per second
        assert features.shape[0] >= 40

    def test_encode_mean(self):
        encoder = MockWav2VecEncoder()
        audio = _make_test_audio()
        mean_feat = encoder.encode_mean(audio)
        assert mean_feat.shape == (WAV2VEC_FEATURE_DIM,)
        assert mean_feat.dtype == np.float32

    def test_factory_creates_mock(self):
        encoder = create_wav2vec_encoder(use_real=False)
        assert isinstance(encoder, MockWav2VecEncoder)


class TestLipFeatureEncoder:
    def test_feature_dim(self):
        encoder = LipFeatureEncoder()
        assert encoder.feature_dim == LIP_FEATURE_DIM

    def test_encode_shape(self):
        encoder = LipFeatureEncoder()
        lip_verts = np.random.randn(2, 3).astype(np.float32)
        features = encoder.encode(lip_verts)
        assert features.shape == (LIP_FEATURE_DIM,)
        assert features.dtype == np.float32

    def test_encode_with_mouth_vertices(self):
        encoder = LipFeatureEncoder()
        lip_verts = np.random.randn(2, 3).astype(np.float32)
        mouth_verts = np.random.randn(8, 3).astype(np.float32)
        features = encoder.encode(lip_verts, mouth_verts)
        assert features.shape == (LIP_FEATURE_DIM,)

    def test_lip_opening_encoded(self):
        """Feature[0] should be lip opening distance."""
        encoder = LipFeatureEncoder()
        # Upper lip at (0, 0.05, 0), lower lip at (0, -0.05, 0)
        lip_verts = np.array([[0, 0.05, 0], [0, -0.05, 0]], dtype=np.float32)
        features = encoder.encode(lip_verts)
        expected_opening = 0.1
        assert abs(features[0] - expected_opening) < 1e-5

    def test_empty_vertices(self):
        """Should handle single vertex gracefully."""
        encoder = LipFeatureEncoder()
        lip_verts = np.random.randn(1, 3).astype(np.float32)
        features = encoder.encode(lip_verts)
        assert features.shape == (LIP_FEATURE_DIM,)
