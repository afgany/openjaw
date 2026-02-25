"""Run PPO training and generate diagnostic plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from openjaw.training.trainer import OpenJawTrainer, TrainerConfig


def main() -> None:
    config = TrainerConfig(
        num_envs=1,
        episode_length=50,
        seed=42,
        log_dir="logs/",
        checkpoint_dir="checkpoints/",
        experiment_name="plot_run",
        learning_rate=3e-4,
    )

    trainer = OpenJawTrainer(config)
    trainer.setup()

    try:
        results = trainer.train_ppo(num_episodes=1000)
    finally:
        trainer.close()

    # Extract metrics
    rollouts = [r["rollout"] for r in results]
    rewards = [r["mean_episode_reward"] for r in results]
    policy_losses = [r["policy_loss"] for r in results]
    value_losses = [r["value_loss"] for r in results]
    entropies = [r["entropy"] for r in results]
    explained_vars = [r["explained_variance"] for r in results]
    episodes_trained = [r["episodes_trained"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("OpenJaw PPO Training — 1,000 Episodes (log_std clamp + KL stop + reward norm)", fontsize=14, fontweight="bold")

    # 1. Mean episode reward
    ax = axes[0, 0]
    ax.plot(rollouts, rewards, "o-", color="#2196F3", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward per Rollout")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Policy loss
    ax = axes[0, 1]
    ax.plot(rollouts, policy_losses, "o-", color="#F44336", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss (Clipped Surrogate)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 3. Value loss
    ax = axes[0, 2]
    ax.plot(rollouts, value_losses, "o-", color="#4CAF50", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Value Loss (MSE)")
    ax.set_title("Value Loss")
    ax.grid(True, alpha=0.3)

    # 4. Entropy
    ax = axes[1, 0]
    ax.plot(rollouts, entropies, "o-", color="#FF9800", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # 5. Explained variance
    ax = axes[1, 1]
    ax.plot(rollouts, explained_vars, "o-", color="#9C27B0", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Explained Variance (Critic)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Perfect")
    ax.set_ylim(-1.5, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Cumulative episodes
    ax = axes[1, 2]
    ax.plot(rollouts, episodes_trained, "o-", color="#607D8B", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Cumulative Episodes")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "figures/training_diagnostics.png"
    import os
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
