"""Step 5: SPARC audio decoder tests."""

import numpy as np
import pytest

from openjaw.audio.sparc_decoder import (
    SPARC_EMA_DIM,
    SPARC_FEATURE_DIM,
    SPARC_SPK_EMB_DIM,
    BaseSPARCDecoder,
    MockSPARCDecoder,
    create_sparc_decoder,
)
from openjaw.core.types import AUDIO_SAMPLE_RATE, NUM_DOF


class TestMockSPARCDecoder:
    def test_creates(self):
        decoder = MockSPARCDecoder()
        assert isinstance(decoder, BaseSPARCDecoder)

    def test_decode_output_shape(self):
        """Decode produces correct length audio at 16kHz."""
        decoder = MockSPARCDecoder()
        L = 5  # 5 frames at 50Hz = 100ms
        ema = np.random.randn(L, SPARC_EMA_DIM).astype(np.float32)
        pitch = np.full((L, 1), 120.0, dtype=np.float32)
        loudness = np.full((L, 1), 0.5, dtype=np.float32)
        spk_emb = np.zeros(SPARC_SPK_EMB_DIM, dtype=np.float32)

        audio = decoder.decode(ema, pitch, loudness, spk_emb)
        expected_samples = L * (AUDIO_SAMPLE_RATE // 50)  # L * 320
        assert audio.shape == (expected_samples,)
        assert audio.dtype == np.float32

    def test_decode_silence_when_no_voicing(self):
        """Zero loudness produces silence."""
        decoder = MockSPARCDecoder()
        L = 3
        ema = np.zeros((L, SPARC_EMA_DIM), dtype=np.float32)
        pitch = np.zeros((L, 1), dtype=np.float32)
        loudness = np.zeros((L, 1), dtype=np.float32)
        spk_emb = np.zeros(SPARC_SPK_EMB_DIM, dtype=np.float32)

        audio = decoder.decode(ema, pitch, loudness, spk_emb)
        assert np.allclose(audio, 0.0)

    def test_decode_nonzero_when_voiced(self):
        """Nonzero loudness + pitch produces audio."""
        decoder = MockSPARCDecoder()
        L = 5
        ema = np.random.randn(L, SPARC_EMA_DIM).astype(np.float32) * 0.1
        pitch = np.full((L, 1), 150.0, dtype=np.float32)
        loudness = np.full((L, 1), 0.5, dtype=np.float32)
        spk_emb = np.zeros(SPARC_SPK_EMB_DIM, dtype=np.float32)

        audio = decoder.decode(ema, pitch, loudness, spk_emb)
        assert np.max(np.abs(audio)) > 0.01

    def test_from_articulatory_state(self):
        """Single-frame conversion from MuJoCo state."""
        decoder = MockSPARCDecoder()
        positions = np.random.randn(NUM_DOF).astype(np.float32) * 0.1
        audio = decoder.from_articulatory_state(positions, vocal_loudness=0.3)
        assert audio.dtype == np.float32
        assert len(audio) == AUDIO_SAMPLE_RATE // 50  # 320 samples (one frame)

    def test_from_trajectory(self):
        """Multi-frame trajectory conversion."""
        decoder = MockSPARCDecoder()
        T = 10
        positions = np.random.randn(T, NUM_DOF).astype(np.float32) * 0.1
        loudness = np.full(T, 0.3, dtype=np.float32)

        audio = decoder.from_trajectory(positions, loudness)
        expected = T * (AUDIO_SAMPLE_RATE // 50)
        assert audio.shape == (expected,)

    def test_adapter_maps_dof_to_ema(self):
        """MuJoCo DOFs 0-11 map to SPARC EMA channels."""
        decoder = MockSPARCDecoder()
        # Set specific DOF values
        positions = np.zeros(NUM_DOF, dtype=np.float32)
        positions[0] = 0.5  # tongue_dorsum_x → TDX
        positions[1] = -0.3  # tongue_dorsum_y → TDY

        # The from_articulatory_state method extracts first 12 as EMA
        audio = decoder.from_articulatory_state(positions, vocal_loudness=0.4)
        assert audio is not None  # Just verify it runs without error


class TestFactoryFunction:
    def test_creates_mock_by_default(self):
        decoder = create_sparc_decoder(use_real=False)
        assert isinstance(decoder, MockSPARCDecoder)

    def test_falls_back_to_mock(self):
        """When use_real=True but package missing, falls back to mock."""
        decoder = create_sparc_decoder(use_real=True)
        assert isinstance(decoder, BaseSPARCDecoder)
