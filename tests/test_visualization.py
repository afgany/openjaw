"""Step 14: Visualization tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openjaw.evaluation.visualization import (
    plot_ablation_comparison,
    plot_articulator_trajectories,
    plot_reward_curves,
    plot_spectrogram_comparison,
)


class TestPlotRewardCurves:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "reward_curves.pdf")
        rewards = {
            "condition_a": list(np.random.randn(100).cumsum()),
            "condition_b": list(np.random.randn(100).cumsum()),
        }
        result = plot_reward_curves(rewards, output_path=out)
        assert Path(result).exists()
        assert result == out

    def test_short_series_no_smoothing(self, tmp_path):
        out = str(tmp_path / "short.pdf")
        rewards = {"short": [1.0, 2.0, 3.0]}
        result = plot_reward_curves(rewards, output_path=out, window=50)
        assert Path(result).exists()

    def test_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "nested" / "dir" / "plot.pdf")
        rewards = {"test": list(range(100))}
        result = plot_reward_curves(rewards, output_path=out)
        assert Path(result).exists()


class TestPlotArticulatorTrajectories:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "trajectories.pdf")
        positions = np.random.randn(100, 13).astype(np.float32)
        result = plot_articulator_trajectories(positions, output_path=out)
        assert Path(result).exists()

    def test_short_trajectory(self, tmp_path):
        out = str(tmp_path / "short_traj.pdf")
        positions = np.random.randn(10, 13).astype(np.float32)
        result = plot_articulator_trajectories(positions, output_path=out)
        assert Path(result).exists()


class TestPlotSpectrogramComparison:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "spectrogram.pdf")
        t = np.arange(16000) / 16000
        gen = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        tgt = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        result = plot_spectrogram_comparison(gen, tgt, output_path=out)
        assert Path(result).exists()

    def test_empty_audio_handled(self, tmp_path):
        out = str(tmp_path / "empty_spec.pdf")
        gen = np.array([], dtype=np.float32)
        tgt = np.array([], dtype=np.float32)
        result = plot_spectrogram_comparison(gen, tgt, output_path=out)
        assert Path(result).exists()


class TestPlotAblationComparison:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "ablation.pdf")
        results = [
            {"condition": "combined", "mean_reward": 5.0, "final_reward": 8.0},
            {"condition": "audio_only", "mean_reward": 3.0, "final_reward": 6.0},
        ]
        result = plot_ablation_comparison(results, output_path=out)
        assert Path(result).exists()
