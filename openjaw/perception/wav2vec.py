"""wav2vec 2.0 encoder: frame-level audio features for observation.

Extracts 768-dim frame-level features from audio using wav2vec 2.0
(or WavLM) for use in the observation vector.

When transformers is available but model loading is expensive,
a lightweight mock produces random features for testing.

Reference: Baevski et al. (2020) "wav2vec 2.0"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray

logger = logging.getLogger(__name__)

WAV2VEC_FEATURE_DIM = 768
WAV2VEC_FEATURE_RATE = 50  # Hz (approximately, depends on model)


class BaseWav2VecEncoder(ABC):
    """Abstract interface for wav2vec-style audio encoding."""

    @property
    def feature_dim(self) -> int:
        return WAV2VEC_FEATURE_DIM

    @abstractmethod
    def encode(self, audio: FloatArray, sample_rate: int = AUDIO_SAMPLE_RATE) -> FloatArray:
        """Encode audio to frame-level features.

        Args:
            audio: Mono audio waveform, shape (N,).
            sample_rate: Audio sample rate (must be 16kHz).

        Returns:
            Frame-level features, shape (L, 768) where L = num_frames at ~50Hz.
        """
        ...

    def encode_mean(self, audio: FloatArray) -> FloatArray:
        """Encode audio and return mean-pooled feature.

        Returns:
            Mean feature vector, shape (768,).
        """
        features = self.encode(audio)
        return np.mean(features, axis=0).astype(np.float32)


class MockWav2VecEncoder(BaseWav2VecEncoder):
    """Mock wav2vec encoder for testing.

    Produces deterministic features from audio spectral characteristics.
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    def encode(self, audio: FloatArray, sample_rate: int = AUDIO_SAMPLE_RATE) -> FloatArray:
        duration = len(audio) / sample_rate
        num_frames = max(1, int(np.ceil(duration * WAV2VEC_FEATURE_RATE)))

        features = np.zeros((num_frames, WAV2VEC_FEATURE_DIM), dtype=np.float32)
        frame_size = max(1, len(audio) // num_frames)

        for i in range(num_frames):
            s = i * frame_size
            e = min(s + frame_size, len(audio))
            chunk = audio[s:e] if e > s else np.zeros(1, dtype=np.float32)

            # Deterministic features from chunk
            rng = np.random.default_rng(42 + i)
            features[i] = rng.standard_normal(WAV2VEC_FEATURE_DIM).astype(np.float32)
            # Embed real audio statistics
            features[i, 0] = float(np.mean(chunk ** 2))
            features[i, 1] = float(np.std(chunk)) if len(chunk) > 1 else 0.0

            norm = np.linalg.norm(features[i])
            if norm > 0:
                features[i] /= norm

        return features


class Wav2VecEncoder(BaseWav2VecEncoder):
    """Real wav2vec 2.0 / WavLM encoder using HuggingFace transformers.

    Requires: pip install transformers torch
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        logger.info(f"Loading wav2vec model '{model_name}' on {device}")
        self._processor = Wav2Vec2Processor.from_pretrained(model_name)
        self._model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()
        self.device = device
        self._torch = torch

    def encode(self, audio: FloatArray, sample_rate: int = AUDIO_SAMPLE_RATE) -> FloatArray:
        inputs = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with self._torch.no_grad():
            outputs = self._model(input_values)

        # last_hidden_state: (1, L, 768)
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return features.astype(np.float32)


def create_wav2vec_encoder(
    use_real: bool = False,
    model_name: str = "facebook/wav2vec2-base-960h",
    device: str = "cpu",
) -> BaseWav2VecEncoder:
    """Factory function for creating wav2vec encoder."""
    if use_real:
        try:
            return Wav2VecEncoder(model_name=model_name, device=device)
        except Exception as e:
            logger.warning(f"wav2vec not available ({e}), falling back to mock")
    return MockWav2VecEncoder()
