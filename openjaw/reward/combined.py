"""Combined multi-modal reward with ablation support.

R(s_t, a_t) = w_a * R_audio + w_v * R_visual + R_aux

Supports ablation modes:
  - "combined": full multi-modal reward (default)
  - "audio_only": w_v = 0
  - "visual_only": w_a = 0
  - "binary_sound": +1 if syllable detected, -1 otherwise (babbling phase)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openjaw.core.types import FloatArray
from openjaw.perception.sylber import BaseSylberEncoder
from openjaw.reward.audio_reward import AudioReward
from openjaw.reward.auxiliary import AuxiliaryReward
from openjaw.reward.visual_reward import VisualReward


@dataclass
class RewardOutput:
    """Structured reward output with components for logging."""

    total: float
    audio: float
    visual: float
    auxiliary: float
    silence_penalty: float
    smoothness_penalty: float
    energy_penalty: float
    has_syllable: bool


class CombinedReward:
    """Multi-modal reward function with ablation support.

    Combines audio (Sylber cosine sim), visual (LVE), and auxiliary
    (silence/smoothness/energy) rewards with configurable weights.
    """

    def __init__(
        self,
        sylber_encoder: BaseSylberEncoder,
        w_audio: float = 0.7,
        w_visual: float = 0.3,
        lambda_silence: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_energy: float = 0.001,
        max_lve: float = 0.05,
        mode: str = "combined",
    ) -> None:
        """
        Args:
            sylber_encoder: Sylber encoder for audio reward.
            w_audio: Weight for audio reward.
            w_visual: Weight for visual reward.
            lambda_silence: Silence penalty weight.
            lambda_smooth: Action smoothness penalty weight.
            lambda_energy: Action energy penalty weight.
            max_lve: Max LVE for normalization.
            mode: Reward mode — "combined", "audio_only", "visual_only", "binary_sound".
        """
        self.audio_reward = AudioReward(sylber_encoder)
        self.visual_reward = VisualReward(max_lve=max_lve)
        self.auxiliary_reward = AuxiliaryReward(
            lambda_silence=lambda_silence,
            lambda_smooth=lambda_smooth,
            lambda_energy=lambda_energy,
        )
        self.sylber_encoder = sylber_encoder

        # Apply mode
        self.mode = mode
        if mode == "audio_only":
            self.w_audio = w_audio
            self.w_visual = 0.0
        elif mode == "visual_only":
            self.w_audio = 0.0
            self.w_visual = w_visual
        elif mode == "binary_sound":
            self.w_audio = 0.0
            self.w_visual = 0.0
        else:  # combined
            self.w_audio = w_audio
            self.w_visual = w_visual

    def compute(
        self,
        generated_audio: FloatArray,
        target_audio_embedding: FloatArray,
        generated_lip_vertices: FloatArray,
        target_lip_vertices: FloatArray,
        action: FloatArray,
        prev_action: FloatArray,
    ) -> RewardOutput:
        """Compute the full multi-modal reward.

        Args:
            generated_audio: Generated waveform from SPARC, shape (N,).
            target_audio_embedding: Pre-computed Sylber embedding of target, shape (768,).
            generated_lip_vertices: Generated lip positions, shape (N_verts, 3).
            target_lip_vertices: Target lip positions, shape (N_verts, 3).
            action: Current action, shape (13,).
            prev_action: Previous action, shape (13,).

        Returns:
            RewardOutput with total reward and all components.
        """
        # Binary sound mode (babbling phase)
        if self.mode == "binary_sound":
            has_syllable = self.sylber_encoder.has_syllable(generated_audio)
            aux = self.auxiliary_reward.compute_components(action, prev_action, has_syllable)
            return RewardOutput(
                total=1.0 if has_syllable else -1.0,
                audio=0.0,
                visual=0.0,
                auxiliary=aux["total"],
                silence_penalty=aux["silence"],
                smoothness_penalty=aux["smoothness"],
                energy_penalty=aux["energy"],
                has_syllable=has_syllable,
            )

        # Audio reward
        has_syllable = self.sylber_encoder.has_syllable(generated_audio)
        r_audio = self.audio_reward.compute(generated_audio, target_audio_embedding)

        # Visual reward
        r_visual = self.visual_reward.compute(generated_lip_vertices, target_lip_vertices)

        # Auxiliary reward
        aux = self.auxiliary_reward.compute_components(action, prev_action, has_syllable)

        # Weighted combination
        total = (
            self.w_audio * r_audio
            + self.w_visual * r_visual
            + aux["total"]
        )

        return RewardOutput(
            total=total,
            audio=r_audio,
            visual=r_visual,
            auxiliary=aux["total"],
            silence_penalty=aux["silence"],
            smoothness_penalty=aux["smoothness"],
            energy_penalty=aux["energy"],
            has_syllable=has_syllable,
        )
