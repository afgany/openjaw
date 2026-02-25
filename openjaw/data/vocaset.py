"""VOCASET dataset loader.

VOCASET: 480 audio-facial motion sequences, 12 subjects, 60 FPS,
5023-vertex FLAME topology meshes.

Standard benchmark used by SelfTalk, FaceFormer, CodeTalker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from openjaw.core.types import FloatArray

logger = logging.getLogger(__name__)


class VOCASETLoader:
    """Load and serve VOCASET data.

    Expected directory structure:
        vocaset_path/
        ├── audio/          # WAV files
        ├── vertices/       # FLAME vertex arrays (.npy)
        └── metadata.json   # Subject and sequence info
    """

    def __init__(self, data_dir: str = "data/vocaset/") -> None:
        self.data_dir = Path(data_dir)
        self._available = self.data_dir.exists()
        self._sequences: list[dict[str, Any]] = []

        if self._available:
            self._scan_sequences()
        else:
            logger.warning(
                f"VOCASET not found at {self.data_dir}. "
                "Use mock data or download VOCASET."
            )

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def num_sequences(self) -> int:
        return len(self._sequences)

    def _scan_sequences(self) -> None:
        """Scan data directory for available sequences."""
        audio_dir = self.data_dir / "audio"
        if not audio_dir.exists():
            return

        for wav_path in sorted(audio_dir.glob("*.wav")):
            name = wav_path.stem
            vert_path = self.data_dir / "vertices" / f"{name}.npy"
            self._sequences.append({
                "name": name,
                "audio_path": wav_path,
                "vertices_path": vert_path if vert_path.exists() else None,
            })

    def load_sequence(self, index: int) -> dict[str, Any]:
        """Load a single sequence (audio + vertices).

        Returns:
            Dict with keys: name, audio, vertices (optional), sample_rate.
        """
        if not self._available or index >= len(self._sequences):
            raise IndexError(f"Sequence {index} not available")

        import soundfile as sf

        seq = self._sequences[index]
        audio, sr = sf.read(str(seq["audio_path"]), dtype="float32")

        result: dict[str, Any] = {
            "name": seq["name"],
            "audio": audio,
            "sample_rate": sr,
            "vertices": None,
        }

        if seq["vertices_path"] is not None:
            result["vertices"] = np.load(str(seq["vertices_path"]))

        return result

    @staticmethod
    def create_mock_sequence(
        duration: float = 2.0,
        sample_rate: int = 16000,
        num_vertices: int = 2,
    ) -> dict[str, Any]:
        """Create a mock VOCASET-like sequence for testing.

        Args:
            duration: Audio duration in seconds.
            sample_rate: Audio sample rate.
            num_vertices: Number of lip vertices.

        Returns:
            Mock sequence dict.
        """
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        n_frames = int(duration * 60)  # 60 FPS
        vertices = np.random.randn(n_frames, num_vertices, 3).astype(np.float32) * 0.01

        return {
            "name": "mock_sequence",
            "audio": audio,
            "sample_rate": sample_rate,
            "vertices": vertices,
        }
