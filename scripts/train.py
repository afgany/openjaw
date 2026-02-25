"""Training entry point for OpenJaw.

Usage:
    python scripts/train.py
    python scripts/train.py --num-episodes 100
    python scripts/train.py --config configs/ppo_syllable.yaml
"""

from __future__ import annotations

import argparse
import logging

from openjaw.training.trainer import OpenJawTrainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenJaw RL agent")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--experiment", type=str, default="openjaw_ppo")
    parser.add_argument("--use-real-sparc", action="store_true")
    parser.add_argument("--use-real-sylber", action="store_true")
    args = parser.parse_args()

    config = TrainerConfig(
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment,
        use_real_sparc=args.use_real_sparc,
        use_real_sylber=args.use_real_sylber,
    )

    trainer = OpenJawTrainer(config)
    trainer.setup()

    try:
        results = trainer.train(num_episodes=args.num_episodes)
        avg_reward = sum(r["reward_total"] for r in results) / len(results)
        print(f"\nTraining complete: {len(results)} episodes, avg reward: {avg_reward:.4f}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
