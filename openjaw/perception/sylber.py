"""Sylber encoder: self-supervised syllabic embeddings for audio reward.

Sylber segments audio into syllable-level units and produces 768-dim embeddings
(HuBERT-base hidden states averaged over each segment).

When the `sylber` package is not installed, a MockSylberEncoder produces
random-but-deterministic embeddings from audio energy — sufficient for
pipeline testing.

Reference: arXiv:2410.07168 "Sylber" (ICLR 2025)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray

logger = logging.getLogger(__name__)

SYLBER_EMBED_DIM = 768
SYLBER_FEATURE_RATE = 50  # Hz (frame-level)


class BaseSylberEncoder(ABC):
    """Abstract interface for Sylber-compatible syllabic encoding."""

    @property
    def embed_dim(self) -> int:
        return SYLBER_EMBED_DIM

    @abstractmethod
    def encode(self, audio: FloatArray, sample_rate: int = AUDIO_SAMPLE_RATE) -> dict:
        """Encode audio to syllabic segments and embeddings.

        Args:
            audio: Mono audio waveform, shape (N,).
            sample_rate: Audio sample rate (must be 16kHz).

        Returns:
            Dictionary with keys:
                segments: (num_segments, 2) — start/end times in seconds
                segment_features: (num_segments, 768) — segment embeddings
                hidden_states: (num_frames, 768) — frame-level features
        """
        ...

    def get_segment_embedding(self, audio: FloatArray) -> FloatArray:
        """Get the mean segment embedding for an audio clip.

        Useful for computing reward: average all segment embeddings into one vector.

        Returns:
            Mean embedding, shape (768,).
        """
        result = self.encode(audio)
        features = result["segment_features"]
        if len(features) == 0:
            return np.zeros(SYLBER_EMBED_DIM, dtype=np.float32)
        return np.mean(features, axis=0).astype(np.float32)

    def has_syllable(self, audio: FloatArray) -> bool:
        """Check if audio contains at least one detected syllable."""
        result = self.encode(audio)
        return len(result["segments"]) > 0


class MockSylberEncoder(BaseSylberEncoder):
    """Mock Sylber encoder for testing.

    Produces deterministic embeddings based on audio energy characteristics.
    Different audio inputs produce different embeddings; similar audio produces
    similar embeddings (via energy-based hashing).
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._rng_base_seed = 12345

    def encode(self, audio: FloatArray, sample_rate: int = AUDIO_SAMPLE_RATE) -> dict:
        duration = len(audio) / sample_rate
        num_frames = max(1, int(np.ceil(duration * SYLBER_FEATURE_RATE)))

        # Detect "syllables" from energy peaks
        frame_size = max(1, len(audio) // num_frames)
        energy = np.array([
            np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2)
            for i in range(num_frames)
        ])

        # Simple peak detection for segment boundaries
        threshold = np.mean(energy) * 0.5
        segments = []
        in_segment = False
        seg_start = 0.0

        for i, e in enumerate(energy):
            t = i / SYLBER_FEATURE_RATE
            if e > threshold and not in_segment:
                in_segment = True
                seg_start = t
            elif e <= threshold and in_segment:
                in_segment = False
                segments.append([seg_start, t])

        if in_segment:
            segments.append([seg_start, duration])

        segments_arr = np.array(segments, dtype=np.float32).reshape(-1, 2) if segments else np.zeros((0, 2), dtype=np.float32)
        num_segments = len(segments)

        # Generate deterministic embeddings from audio content
        segment_features = np.zeros((num_segments, SYLBER_EMBED_DIM), dtype=np.float32)
        for i, (start, end) in enumerate(segments):
            s_idx = int(start * sample_rate)
            e_idx = min(int(end * sample_rate), len(audio))
            chunk = audio[s_idx:e_idx] if e_idx > s_idx else audio[:frame_size]
            # Deterministic embedding from audio chunk statistics
            rng = np.random.default_rng(self._rng_base_seed + hash(chunk.tobytes()) % (2**31))
            segment_features[i] = rng.standard_normal(SYLBER_EMBED_DIM).astype(np.float32)
            # Embed some actual audio information
            segment_features[i, 0] = float(np.mean(chunk ** 2))  # energy
            segment_features[i, 1] = float(np.std(chunk))  # variation
            # Normalize
            norm = np.linalg.norm(segment_features[i])
            if norm > 0:
                segment_features[i] /= norm

        # Frame-level hidden states
        hidden_states = np.zeros((num_frames, SYLBER_EMBED_DIM), dtype=np.float32)
        for i in range(num_frames):
            s_idx = i * frame_size
            e_idx = min(s_idx + frame_size, len(audio))
            chunk = audio[s_idx:e_idx] if e_idx > s_idx else np.zeros(1)
            rng = np.random.default_rng(self._rng_base_seed + i + abs(int(np.sum(chunk * 1000))))
            hidden_states[i] = rng.standard_normal(SYLBER_EMBED_DIM).astype(np.float32)
            norm = np.linalg.norm(hidden_states[i])
            if norm > 0:
                hidden_states[i] /= norm

        return {
            "segments": segments_arr,
            "segment_features": segment_features,
            "hidden_states": hidden_states,
        }


class SylberEncoder(BaseSylberEncoder):
    """Real Sylber encoder using the sylber package.

    Requires: pip install sylber
    """

    def __init__(self, device: str = "cpu") -> None:
        try:
            from sylber import Segmenter
        except ImportError:
            raise ImportError(
                "Sylber requires the sylber package. "
                "Install with: pip install sylber"
            )

        logger.info(f"Loading Sylber segmenter on {device}")
        self._segmenter = Segmenter(model_ckpt="sylber", device=device)
        self.device = device

    def encode(self, audio: FloatArray, sample_rate: int = AUDIO_SAMPLE_RATE) -> dict:
        import torch

        # Sylber expects (1, num_samples) tensor
        wav_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        outputs = self._segmenter(wav=wav_tensor, in_second=True)

        return {
            "segments": np.asarray(outputs["segments"], dtype=np.float32),
            "segment_features": np.asarray(outputs["segment_features"], dtype=np.float32),
            "hidden_states": np.asarray(outputs["hidden_states"], dtype=np.float32),
        }


def create_sylber_encoder(
    use_real: bool = False,
    device: str = "cpu",
) -> BaseSylberEncoder:
    """Factory function for creating Sylber encoder."""
    if use_real:
        try:
            return SylberEncoder(device=device)
        except ImportError:
            logger.warning("Sylber not available, falling back to MockSylberEncoder")
    return MockSylberEncoder()
