"""SPARC audio decoder: converts articulatory state to audio waveform.

SPARC (Speech Articulatory Coding) maps 12-channel EMA articulatory trajectories
+ pitch + loudness to speech audio via a HiFi-GAN decoder.

The MuJoCo 13-DOF state maps to SPARC's 14-dim input:
  DOFs 0-11: EMA channels (TDX, TDY, TBX, TBY, TTX, TTY, LIX, LIY, ULX, ULY, LLX, LLY)
  DOF 12: vocal loudness → SPARC pitch + loudness

When the `speech-articulatory-coding` package is not installed, a MockSPARCDecoder
is used that generates sine-wave audio proportional to vocal loudness — sufficient
for testing the pipeline and reward computation structure.

Reference: Cho et al. (2024) "SPARC: Coding Speech Through Vocal Tract Kinematics"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray, NUM_DOF

logger = logging.getLogger(__name__)

# SPARC feature dimensions
SPARC_EMA_DIM = 12  # 6 articulators × 2 axes (X, Y)
SPARC_FEATURE_DIM = 14  # 12 EMA + 1 pitch + 1 loudness
SPARC_FEATURE_RATE = 50  # Hz
SPARC_SPK_EMB_DIM = 64

# Default pitch for voiced sounds (Hz)
DEFAULT_PITCH_HZ = 120.0


class BaseSPARCDecoder(ABC):
    """Abstract interface for SPARC-compatible articulatory-to-audio decoding."""

    @abstractmethod
    def decode(
        self,
        ema: FloatArray,
        pitch: FloatArray,
        loudness: FloatArray,
        spk_emb: FloatArray,
    ) -> FloatArray:
        """Decode articulatory features to audio waveform.

        Args:
            ema: Articulatory positions, shape (L, 12).
            pitch: F0 in Hz, shape (L, 1).
            loudness: Frame-level amplitude, shape (L, 1).
            spk_emb: Speaker embedding, shape (64,).

        Returns:
            Audio waveform at 16kHz, shape (N,) where N = L * 320.
        """
        ...

    def from_articulatory_state(
        self,
        positions: FloatArray,
        vocal_loudness: float,
        spk_emb: FloatArray | None = None,
    ) -> FloatArray:
        """Convert a single-frame MuJoCo state to audio.

        Args:
            positions: 13-DOF articulator positions from MuJoCo.
            vocal_loudness: Scalar loudness value from DOF 12.
            spk_emb: Optional speaker embedding. Uses default if None.

        Returns:
            Audio waveform for one control step (~40ms at 16kHz = 640 samples).
        """
        # Map MuJoCo DOFs 0-11 to SPARC EMA (single frame → (1, 12))
        ema = positions[:SPARC_EMA_DIM].reshape(1, SPARC_EMA_DIM)

        # Derive pitch and loudness from vocal_loudness
        # Pitch: scale loudness to a reasonable F0 range
        loudness_val = max(0.0, float(vocal_loudness))
        pitch_hz = DEFAULT_PITCH_HZ if loudness_val > 0.05 else 0.0
        pitch = np.array([[pitch_hz]], dtype=np.float32)
        loudness = np.array([[loudness_val]], dtype=np.float32)

        if spk_emb is None:
            spk_emb = np.zeros(SPARC_SPK_EMB_DIM, dtype=np.float32)

        return self.decode(ema, pitch, loudness, spk_emb)

    def from_trajectory(
        self,
        positions_seq: FloatArray,
        vocal_loudness_seq: FloatArray,
        spk_emb: FloatArray | None = None,
    ) -> FloatArray:
        """Convert a sequence of MuJoCo states to audio.

        Args:
            positions_seq: Articulator positions over time, shape (T, 13).
            vocal_loudness_seq: Loudness over time, shape (T,).
            spk_emb: Optional speaker embedding.

        Returns:
            Audio waveform for the full trajectory.
        """
        T = positions_seq.shape[0]
        ema = positions_seq[:, :SPARC_EMA_DIM]  # (T, 12)

        # Derive pitch: voiced when loudness > threshold
        loudness = np.abs(vocal_loudness_seq).reshape(T, 1).astype(np.float32)
        pitch = np.where(loudness > 0.05, DEFAULT_PITCH_HZ, 0.0).astype(np.float32)

        if spk_emb is None:
            spk_emb = np.zeros(SPARC_SPK_EMB_DIM, dtype=np.float32)

        return self.decode(ema, pitch, loudness, spk_emb)


class MockSPARCDecoder(BaseSPARCDecoder):
    """Mock SPARC decoder for testing.

    Generates sine-wave audio proportional to loudness and pitch.
    Preserves the correct API contract and output shapes.
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._samples_per_frame = sample_rate // SPARC_FEATURE_RATE  # 320

    def decode(
        self,
        ema: FloatArray,
        pitch: FloatArray,
        loudness: FloatArray,
        spk_emb: FloatArray,
    ) -> FloatArray:
        L = ema.shape[0]
        N = L * self._samples_per_frame
        audio = np.zeros(N, dtype=np.float32)

        for i in range(L):
            start = i * self._samples_per_frame
            end = start + self._samples_per_frame
            f0 = float(pitch[i, 0])
            amp = float(loudness[i, 0])

            if f0 > 0 and amp > 0.01:
                t = np.arange(self._samples_per_frame) / self.sample_rate
                # Add harmonics modulated by articulatory state for variation
                formant_shift = float(np.mean(np.abs(ema[i]))) * 500 + 300
                audio[start:end] = amp * (
                    0.5 * np.sin(2 * np.pi * f0 * t)
                    + 0.3 * np.sin(2 * np.pi * formant_shift * t)
                    + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
                )

        return audio.astype(np.float32)


class SPARCDecoder(BaseSPARCDecoder):
    """Real SPARC decoder using the speech-articulatory-coding package.

    Requires: pip install speech-articulatory-coding
    """

    def __init__(
        self,
        model_tag: str = "en",
        device: str = "cpu",
    ) -> None:
        try:
            from sparc import load_model
        except ImportError:
            raise ImportError(
                "SPARC requires the speech-articulatory-coding package. "
                "Install with: pip install speech-articulatory-coding"
            )

        logger.info(f"Loading SPARC model '{model_tag}' on {device}")
        self._coder = load_model(model_tag, device=device)
        self.device = device
        self.sample_rate = AUDIO_SAMPLE_RATE

    def decode(
        self,
        ema: FloatArray,
        pitch: FloatArray,
        loudness: FloatArray,
        spk_emb: FloatArray,
    ) -> FloatArray:
        audio = self._coder.decode(
            ema=ema,
            pitch=pitch,
            loudness=loudness,
            spk_emb=spk_emb,
        )
        return np.asarray(audio, dtype=np.float32)

    def encode(self, wav_path: str) -> dict:
        """Encode audio file to SPARC features (for reference data processing).

        Returns dict with keys: ema, pitch, loudness, periodicity, spk_emb, pitch_stats.
        """
        return self._coder.encode(wav_path)


def create_sparc_decoder(
    use_real: bool = False,
    model_tag: str = "en",
    device: str = "cpu",
) -> BaseSPARCDecoder:
    """Factory function for creating SPARC decoder.

    Args:
        use_real: If True, attempt to load real SPARC. Falls back to mock on failure.
        model_tag: SPARC model variant ("en", "multi", "en+").
        device: Torch device string.

    Returns:
        SPARC decoder instance.
    """
    if use_real:
        try:
            return SPARCDecoder(model_tag=model_tag, device=device)
        except ImportError:
            logger.warning("SPARC not available, falling back to MockSPARCDecoder")
    return MockSPARCDecoder()
