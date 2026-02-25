"""Step 17: Run Curriculum Phase 2 (Syllable Imitation).

Usage:
    python scripts/run_syllables.py --checkpoint checkpoints/checkpoint_ep6000.pt
    python scripts/run_syllables.py --use-real-sparc --use-real-sylber --device cuda

Expected runtime: ~40-60 GPU-hours on RTX 3090/4090.
Expected outcome: 10-20 syllables with Sylber cos_sim > 0.7 on best syllables.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from openjaw.data.custom_speaker import CustomSpeakerLoader
from openjaw.data.preprocessing import segment_audio_to_syllables
from openjaw.perception.sylber import create_sylber_encoder
from openjaw.training.curriculum import CurriculumPhase, CurriculumScheduler
from openjaw.training.trainer import OpenJawTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Syllable Imitation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint (e.g. from run_babbling_vowels.py)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-real-sparc", action="store_true")
    parser.add_argument("--use-real-sylber", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=25000)
    parser.add_argument("--data-dir", type=str, default="data/custom/",
                        help="Directory with speaker recordings")
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    logger.info(f"Running Phase 2: {args.num_episodes} syllable episodes")

    # Build curriculum: single syllable phase
    phases = [
        CurriculumPhase(
            name="syllables",
            episodes=args.num_episodes + 1,  # +1 to prevent early completion
            reward_mode="combined",
        ),
    ]

    config = TrainerConfig(
        num_envs=1,
        episode_length=50,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name="syllables",
        use_real_sparc=args.use_real_sparc,
        use_real_sylber=args.use_real_sylber,
        device=args.device,
    )

    trainer = OpenJawTrainer(config)
    trainer.setup()

    # Override curriculum to syllable-only
    trainer.curriculum = CurriculumScheduler(phases=phases)
    trainer._update_reward_mode()

    # Load checkpoint if available
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Resumed from {args.checkpoint}")

    # Load syllable targets from speaker recordings
    speaker = CustomSpeakerLoader(data_dir=args.data_dir)
    syllable_targets = []
    if speaker.is_available:
        for name in speaker.list_recordings():
            audio, sr = speaker.load_audio(name)
            segments = segment_audio_to_syllables(audio, trainer.sylber, sr)
            syllable_targets.extend(segments)
            logger.info(f"Loaded {len(segments)} syllables from {name}")

    if not syllable_targets:
        # Fall back to mock recording
        logger.warning("No speaker recordings found; using mock data")
        audio, sr = CustomSpeakerLoader.create_mock_recording(duration=3.0)
        syllable_targets = segment_audio_to_syllables(audio, trainer.sylber, sr)

    logger.info(f"Total syllable targets: {len(syllable_targets)}")

    start_time = time.time()
    try:
        results = []
        for ep in range(args.num_episodes):
            target_idx = ep % len(syllable_targets)
            target_emb = syllable_targets[target_idx].audio_embedding

            result = trainer.run_episode(target_audio_emb=target_emb)
            results.append(result)

            if (ep + 1) % 100 == 0:
                recent = results[-100:]
                avg = sum(r["reward_total"] for r in recent) / len(recent)
                avg_audio = sum(r["reward_audio"] for r in recent) / len(recent)
                avg_visual = sum(r["reward_visual"] for r in recent) / len(recent)
                elapsed = time.time() - start_time
                logger.info(
                    f"Episode {ep + 1}/{args.num_episodes} | "
                    f"Avg total: {avg:.4f} | "
                    f"Avg audio: {avg_audio:.4f} | "
                    f"Avg visual: {avg_visual:.4f} | "
                    f"Elapsed: {elapsed / 3600:.1f}h"
                )

            if (ep + 1) % 1000 == 0:
                trainer.save_checkpoint()

        trainer.save_checkpoint()

        elapsed = time.time() - start_time
        logger.info(f"\nTraining complete in {elapsed / 3600:.2f} hours")
        logger.info(f"Total episodes: {len(results)}")

        if results:
            final_100 = results[-min(100, len(results)):]
            avg_r = sum(r["reward_total"] for r in final_100) / len(final_100)
            best_r = max(r["reward_total"] for r in results)
            logger.info(f"Final avg reward (last 100): {avg_r:.4f}")
            logger.info(f"Best single-episode reward: {best_r:.4f}")

    finally:
        trainer.close()


if __name__ == "__main__":
    main()
