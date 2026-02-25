"""Visualization tools: reward curves, articulator trajectories, spectrograms.

Generates paper-ready PDF figures from training data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray
from openjaw.env.articulators import ARTICULATOR_NAMES


def plot_reward_curves(
    episode_rewards: dict[str, list[float]],
    output_path: str = "paper/figures/reward_curves.pdf",
    title: str = "Training Reward Curves",
    window: int = 50,
) -> str:
    """Plot smoothed reward curves for multiple conditions.

    Args:
        episode_rewards: Dict mapping condition name to list of per-episode rewards.
        output_path: Path to save figure.
        title: Figure title.
        window: Smoothing window size.

    Returns:
        Output path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rewards in episode_rewards.items():
        episodes = np.arange(len(rewards))
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(episodes[:len(smoothed)], smoothed, label=name, linewidth=1.5)
        else:
            ax.plot(episodes, rewards, label=name, linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_articulator_trajectories(
    positions: FloatArray,
    output_path: str = "paper/figures/articulator_trajectories.pdf",
    title: str = "Articulator Trajectories",
    control_freq: int = 25,
) -> str:
    """Plot 13-DOF articulator positions over time.

    Args:
        positions: Articulator positions, shape (T, 13).
        output_path: Path to save figure.
        title: Figure title.
        control_freq: Control frequency in Hz.

    Returns:
        Output path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    T = positions.shape[0]
    time = np.arange(T) / control_freq

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Group by body part
    groups = {
        "Tongue": [0, 1, 2, 3, 4, 5],
        "Jaw": [6, 7],
        "Lips": [8, 9, 10, 11],
        "Voicing": [12],
    }

    for ax, (group_name, indices) in zip(axes, groups.items()):
        for idx in indices:
            label = ARTICULATOR_NAMES[idx].replace("_", " ")
            ax.plot(time, positions[:, idx], label=label, linewidth=1)
        ax.set_ylabel(group_name)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_spectrogram_comparison(
    generated: FloatArray,
    target: FloatArray,
    output_path: str = "paper/figures/spectrogram_comparison.pdf",
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> str:
    """Side-by-side spectrogram comparison.

    Args:
        generated: Generated audio, shape (N,).
        target: Target audio, shape (M,).
        output_path: Path to save figure.
        sample_rate: Audio sample rate.

    Returns:
        Output path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, audio, label in [(ax1, target, "Target"), (ax2, generated, "Generated")]:
        if len(audio) == 0:
            ax.text(0.5, 0.5, "No audio", ha="center", va="center")
            ax.set_title(label)
            continue
        ax.specgram(audio, Fs=sample_rate, NFFT=512, noverlap=256, cmap="magma")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ablation_comparison(
    ablation_results: list[dict[str, Any]],
    output_path: str = "paper/figures/ablation_comparison.pdf",
) -> str:
    """Bar chart comparing ablation conditions.

    Args:
        ablation_results: List of dicts with keys: condition, mean_reward, final_reward.
        output_path: Path to save figure.

    Returns:
        Output path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    conditions = [r["condition"] for r in ablation_results]
    mean_rewards = [r["mean_reward"] for r in ablation_results]
    final_rewards = [r["final_reward"] for r in ablation_results]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mean_rewards, width, label="Mean Reward", color="#4C72B0")
    ax.bar(x + width / 2, final_rewards, width, label="Final Reward", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Reward")
    ax.set_title("Ablation Study Results")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
