"""Custom speaker data loader.

Loads audio + video recordings of a target speaker for person-specific imitation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray

logger = logging.getLogger(__name__)


class CustomSpeakerLoader:
    """Load custom speaker recordings.

    Expected directory structure:
        speaker_path/
        ├── audio/          # WAV files (16kHz+)
        ├── video/          # MP4 files (optional)
        └── flame/          # FLAME parameters (.npy, optional)
    """

    def __init__(self, data_dir: str = "data/custom/") -> None:
        self.data_dir = Path(data_dir)
        self._available = self.data_dir.exists()

    @property
    def is_available(self) -> bool:
        return self._available

    def list_recordings(self) -> list[str]:
        """List available audio recordings."""
        if not self._available:
            return []
        audio_dir = self.data_dir / "audio"
        if not audio_dir.exists():
            return []
        return sorted(p.stem for p in audio_dir.glob("*.wav"))

    def load_audio(self, name: str) -> tuple[FloatArray, int]:
        """Load audio recording by name.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        import soundfile as sf
        path = self.data_dir / "audio" / f"{name}.wav"
        audio, sr = sf.read(str(path), dtype="float32")
        return audio.astype(np.float32), sr

    @staticmethod
    def create_mock_recording(
        duration: float = 2.0,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> tuple[FloatArray, int]:
        """Create mock speaker recording for testing."""
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        audio = (
            0.3 * np.sin(2 * np.pi * 150 * t)
            + 0.2 * np.sin(2 * np.pi * 700 * t)
            + 0.1 * np.sin(2 * np.pi * 1200 * t)
        ).astype(np.float32)
        return audio, sample_rate
