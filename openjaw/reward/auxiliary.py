"""Auxiliary reward terms: silence penalty, smoothness, energy regularization.

R_aux = -λ_silence * 𝟙[no_syllable]
        -λ_smooth  * ||a_t - a_{t-1}||²
        -λ_energy  * ||a_t||²
"""

from __future__ import annotations

import numpy as np

from openjaw.core.types import FloatArray


class AuxiliaryReward:
    """Compute auxiliary reward penalties and bonuses."""

    def __init__(
        self,
        lambda_silence: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_energy: float = 0.001,
    ) -> None:
        self.lambda_silence = lambda_silence
        self.lambda_smooth = lambda_smooth
        self.lambda_energy = lambda_energy

    def compute(
        self,
        action: FloatArray,
        prev_action: FloatArray,
        has_syllable: bool,
    ) -> float:
        """Compute auxiliary reward.

        Args:
            action: Current action, shape (13,).
            prev_action: Previous action, shape (13,).
            has_syllable: Whether Sylber detected a syllable.

        Returns:
            Auxiliary reward (typically negative — penalties).
        """
        silence_penalty = -self.lambda_silence * (0.0 if has_syllable else 1.0)
        smoothness_penalty = -self.lambda_smooth * float(np.sum((action - prev_action) ** 2))
        energy_penalty = -self.lambda_energy * float(np.sum(action ** 2))

        return silence_penalty + smoothness_penalty + energy_penalty

    def compute_components(
        self,
        action: FloatArray,
        prev_action: FloatArray,
        has_syllable: bool,
    ) -> dict[str, float]:
        """Compute auxiliary reward with individual components for logging.

        Returns:
            Dictionary with keys: silence, smoothness, energy, total.
        """
        silence = -self.lambda_silence * (0.0 if has_syllable else 1.0)
        smoothness = -self.lambda_smooth * float(np.sum((action - prev_action) ** 2))
        energy = -self.lambda_energy * float(np.sum(action ** 2))
        return {
            "silence": silence,
            "smoothness": smoothness,
            "energy": energy,
            "total": silence + smoothness + energy,
        }
