"""Step 16: Run Curriculum Phase 0 (Babbling) + Phase 1 (Vowels).

Usage:
    python scripts/run_babbling_vowels.py
    python scripts/run_babbling_vowels.py --use-real-sparc --use-real-sylber
    python scripts/run_babbling_vowels.py --device cuda

Expected runtime: ~10-20 GPU-hours on RTX 3090/4090.
Expected outcome: agent learns to produce sound (babbling) and vowel-like
formants (vowels), with Sylber cos_sim > 0.5 on best vowels.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from openjaw.data.preprocessing import create_vowel_targets
from openjaw.perception.sylber import create_sylber_encoder
from openjaw.training.curriculum import CurriculumPhase, CurriculumScheduler
from openjaw.training.trainer import OpenJawTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0+1: Babbling + Vowels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-real-sparc", action="store_true")
    parser.add_argument("--use-real-sylber", action="store_true")
    parser.add_argument("--babbling-episodes", type=int, default=1000)
    parser.add_argument("--vowel-episodes", type=int, default=5000)
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    total_episodes = args.babbling_episodes + args.vowel_episodes
    logger.info(f"Running Phase 0+1: {args.babbling_episodes} babbling + {args.vowel_episodes} vowel episodes")

    # Build curriculum with only phases 0 and 1
    phases = [
        CurriculumPhase(
            name="babbling",
            episodes=args.babbling_episodes,
            reward_mode="binary_sound",
        ),
        CurriculumPhase(
            name="vowels",
            episodes=args.vowel_episodes,
            reward_mode="audio_only",
            targets=["a", "e", "i", "o", "u"],
        ),
    ]

    config = TrainerConfig(
        num_envs=1,
        episode_length=50,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name="babbling_vowels",
        use_real_sparc=args.use_real_sparc,
        use_real_sylber=args.use_real_sylber,
        device=args.device,
    )

    trainer = OpenJawTrainer(config)
    trainer.setup()

    # Override curriculum
    trainer.curriculum = CurriculumScheduler(phases=phases)
    trainer._update_reward_mode()

    # Prepare vowel targets for Phase 1
    vowel_targets = create_vowel_targets(trainer.sylber)
    logger.info(f"Prepared {len(vowel_targets)} vowel targets")

    start_time = time.time()
    try:
        results = []
        for ep in range(total_episodes):
            # During vowel phase, cycle through vowel targets
            target_emb = None
            if trainer.curriculum.current_phase.name == "vowels":
                target_idx = ep % len(vowel_targets)
                target_emb = vowel_targets[target_idx].audio_embedding

            result = trainer.run_episode(target_audio_emb=target_emb)
            results.append(result)

            if (ep + 1) % 100 == 0:
                recent = results[-100:]
                avg = sum(r["reward_total"] for r in recent) / len(recent)
                elapsed = time.time() - start_time
                logger.info(
                    f"Episode {ep + 1}/{total_episodes} | "
                    f"Phase: {result['phase']} | "
                    f"Avg reward (last 100): {avg:.4f} | "
                    f"Elapsed: {elapsed / 3600:.1f}h"
                )

            if (ep + 1) % 1000 == 0:
                trainer.save_checkpoint()

        # Final checkpoint
        trainer.save_checkpoint()

        elapsed = time.time() - start_time
        logger.info(f"\nTraining complete in {elapsed / 3600:.2f} hours")
        logger.info(f"Total episodes: {len(results)}")

        # Summary stats per phase
        babbling_results = [r for r in results if r["phase"] == "babbling"]
        vowel_results = [r for r in results if r["phase"] == "vowels"]

        if babbling_results:
            avg_b = sum(r["reward_total"] for r in babbling_results) / len(babbling_results)
            logger.info(f"Babbling avg reward: {avg_b:.4f}")

        if vowel_results:
            avg_v = sum(r["reward_total"] for r in vowel_results) / len(vowel_results)
            best_v = max(r["reward_total"] for r in vowel_results)
            logger.info(f"Vowels avg reward: {avg_v:.4f}, best: {best_v:.4f}")

    finally:
        trainer.close()


if __name__ == "__main__":
    main()
