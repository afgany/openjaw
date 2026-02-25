"""Audio/video preprocessing: segmentation and feature extraction.

Processes reference speaker data into episode-level targets
(audio embeddings + lip vertex targets) for RL training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray
from openjaw.perception.sylber import BaseSylberEncoder

logger = logging.getLogger(__name__)


@dataclass
class SyllableTarget:
    """Pre-processed target for a single syllable episode."""

    label: str  # Syllable label (e.g., "ba", "ka")
    audio_embedding: FloatArray  # Sylber embedding, shape (768,)
    audio_segment: FloatArray  # Raw audio waveform for the syllable
    lip_vertices: FloatArray | None  # Target lip vertices if available, shape (N, 3)
    start_time: float  # Start time in source audio (seconds)
    end_time: float  # End time in source audio (seconds)


def segment_audio_to_syllables(
    audio: FloatArray,
    encoder: BaseSylberEncoder,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> list[SyllableTarget]:
    """Segment audio into syllables and extract embeddings.

    Args:
        audio: Full audio waveform, shape (N,).
        encoder: Sylber encoder for segmentation.
        sample_rate: Audio sample rate.

    Returns:
        List of SyllableTarget, one per detected syllable.
    """
    result = encoder.encode(audio, sample_rate)
    segments = result["segments"]  # (num_segments, 2)
    features = result["segment_features"]  # (num_segments, 768)

    targets = []
    for i in range(len(segments)):
        start, end = float(segments[i, 0]), float(segments[i, 1])
        start_sample = int(start * sample_rate)
        end_sample = min(int(end * sample_rate), len(audio))

        if end_sample <= start_sample:
            continue

        targets.append(SyllableTarget(
            label=f"seg_{i}",
            audio_embedding=features[i].astype(np.float32),
            audio_segment=audio[start_sample:end_sample].astype(np.float32),
            lip_vertices=None,
            start_time=start,
            end_time=end,
        ))

    return targets


def create_vowel_targets(
    encoder: BaseSylberEncoder,
    vowels: list[str] | None = None,
    duration: float = 0.5,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> list[SyllableTarget]:
    """Create synthetic vowel targets for curriculum Phase 1.

    Generates sine-wave audio at different frequencies to represent
    different vowels. In production, these would be extracted from
    real recordings.

    Args:
        encoder: Sylber encoder for embedding extraction.
        vowels: List of vowel labels. Default: ["a", "e", "i", "o", "u"].
        duration: Duration per vowel in seconds.
        sample_rate: Audio sample rate.

    Returns:
        List of SyllableTarget, one per vowel.
    """
    if vowels is None:
        vowels = ["a", "e", "i", "o", "u"]

    # Approximate formant frequencies for each vowel
    formant_map = {
        "a": (730, 1090),
        "e": (530, 1840),
        "i": (270, 2290),
        "o": (570, 840),
        "u": (300, 870),
    }

    targets = []
    t = np.arange(int(sample_rate * duration)) / sample_rate

    for vowel in vowels:
        f1, f2 = formant_map.get(vowel, (500, 1500))
        # Synthesize simple vowel-like audio
        audio = (
            0.4 * np.sin(2 * np.pi * 120 * t)  # F0
            + 0.3 * np.sin(2 * np.pi * f1 * t)  # F1
            + 0.2 * np.sin(2 * np.pi * f2 * t)  # F2
        ).astype(np.float32)

        embedding = encoder.get_segment_embedding(audio)

        targets.append(SyllableTarget(
            label=vowel,
            audio_embedding=embedding,
            audio_segment=audio,
            lip_vertices=None,
            start_time=0.0,
            end_time=duration,
        ))

    return targets
