"""Generate all paper figures from training results or mock data.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --from-results results/ablations/ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from openjaw.data.custom_speaker import CustomSpeakerLoader
from openjaw.evaluation.visualization import (
    plot_ablation_comparison,
    plot_articulator_trajectories,
    plot_reward_curves,
    plot_spectrogram_comparison,
)
from openjaw.training.trainer import OpenJawTrainer, TrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_figures(output_dir: str) -> None:
    """Generate all paper figures from mock/demo data."""
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # ── 1. Reward Curves ─────────────────────────────────────────────
    logger.info("Generating reward curves...")
    n_eps = 500
    noise = rng.standard_normal(n_eps)
    rewards = {
        "Combined (ours)": list(np.cumsum(noise * 0.1 + 0.05)),
        "Audio-only": list(np.cumsum(noise * 0.1 + 0.03)),
        "Visual-only": list(np.cumsum(noise * 0.1 + 0.01)),
        "No curriculum": list(np.cumsum(noise * 0.1 + 0.02)),
    }
    plot_reward_curves(
        rewards,
        output_path=str(fig_dir / "reward_curves.pdf"),
        title="Training Reward Curves",
        window=20,
    )

    # ── 2. Articulator Trajectories ──────────────────────────────────
    logger.info("Generating articulator trajectories...")
    # Run a short episode to get real trajectories
    config = TrainerConfig(num_envs=1, episode_length=50, seed=42)
    trainer = OpenJawTrainer(config)
    trainer.setup()
    result = trainer.run_episode()
    trainer.close()

    plot_articulator_trajectories(
        result["positions_history"],
        output_path=str(fig_dir / "articulator_trajectories.pdf"),
        title="Articulator Trajectories (Single Episode)",
    )

    # ── 3. Spectrogram Comparison ────────────────────────────────────
    logger.info("Generating spectrogram comparison...")
    mock_audio, sr = CustomSpeakerLoader.create_mock_recording(duration=1.0)
    # Create a "generated" version with slight modification
    generated = mock_audio + rng.standard_normal(len(mock_audio)).astype(np.float32) * 0.05

    plot_spectrogram_comparison(
        generated, mock_audio,
        output_path=str(fig_dir / "spectrogram_comparison.pdf"),
    )

    # ── 4. Ablation Comparison ───────────────────────────────────────
    logger.info("Generating ablation comparison...")
    ablation_results = [
        {"condition": "Combined", "mean_reward": 12.5, "final_reward": 18.3},
        {"condition": "Audio-only", "mean_reward": 8.2, "final_reward": 14.1},
        {"condition": "Visual-only", "mean_reward": 4.1, "final_reward": 7.8},
        {"condition": "No curriculum", "mean_reward": 9.3, "final_reward": 13.6},
    ]
    plot_ablation_comparison(
        ablation_results,
        output_path=str(fig_dir / "ablation_comparison.pdf"),
    )

    logger.info(f"All figures saved to {fig_dir}/")


def generate_from_results(results_path: str, output_dir: str) -> None:
    """Generate figures from actual ablation results JSON."""
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    # Reward curves
    reward_data = {r["condition"]: r["rewards"] for r in results}
    window = max(1, len(next(iter(reward_data.values()))) // 20)
    plot_reward_curves(
        reward_data,
        output_path=str(fig_dir / "reward_curves.pdf"),
        title="Ablation Study: Reward Curves",
        window=window,
    )

    # Ablation bar chart
    plot_ablation_comparison(
        results,
        output_path=str(fig_dir / "ablation_comparison.pdf"),
    )

    logger.info(f"Figures generated from {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--from-results", type=str, default=None,
                        help="Path to ablation_results.json for real figures")
    parser.add_argument("--output-dir", type=str, default="paper/figures/")
    args = parser.parse_args()

    if args.from_results:
        generate_from_results(args.from_results, args.output_dir)
    else:
        generate_mock_figures(args.output_dir)


if __name__ == "__main__":
    main()
