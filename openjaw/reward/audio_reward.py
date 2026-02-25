"""Audio reward: cosine similarity in Sylber embedding space.

R_audio = cos_sim(Sylber(audio_gen), Sylber(audio_target)) ∈ [-1, 1]

Reference: Anand et al. (2025) — uses Sylber cosine similarity as reward.
"""

from __future__ import annotations

import numpy as np

from openjaw.core.types import FloatArray
from openjaw.perception.sylber import BaseSylberEncoder


class AudioReward:
    """Compute audio similarity reward using Sylber embeddings."""

    def __init__(self, encoder: BaseSylberEncoder) -> None:
        self.encoder = encoder

    def compute(
        self,
        generated_audio: FloatArray,
        target_embedding: FloatArray,
    ) -> float:
        """Compute cosine similarity between generated audio and target embedding.

        Args:
            generated_audio: Generated audio waveform, shape (N,).
            target_embedding: Pre-computed Sylber embedding of target, shape (768,).

        Returns:
            Cosine similarity in [-1, 1].
        """
        gen_embedding = self.encoder.get_segment_embedding(generated_audio)
        return self._cosine_similarity(gen_embedding, target_embedding)

    def compute_from_embeddings(
        self,
        generated_embedding: FloatArray,
        target_embedding: FloatArray,
    ) -> float:
        """Compute cosine similarity between two pre-computed embeddings.

        Args:
            generated_embedding: shape (768,).
            target_embedding: shape (768,).

        Returns:
            Cosine similarity in [-1, 1].
        """
        return self._cosine_similarity(generated_embedding, target_embedding)

    @staticmethod
    def _cosine_similarity(a: FloatArray, b: FloatArray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return -1.0  # No valid embedding → minimum similarity
        return float(np.dot(a, b) / (norm_a * norm_b))
