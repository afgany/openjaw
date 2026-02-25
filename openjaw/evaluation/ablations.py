"""Ablation experiment runner.

Runs training under different reward configurations to validate
the contribution of each component.

Ablation conditions:
  1. audio_only:    w_a > 0, w_v = 0
  2. visual_only:   w_a = 0, w_v > 0
  3. combined:      w_a > 0, w_v > 0 (default)
  4. no_curriculum:  combined reward, no curriculum phasing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from openjaw.training.curriculum import CurriculumPhase, CurriculumScheduler
from openjaw.training.trainer import OpenJawTrainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class AblationCondition:
    """Definition of an ablation experiment condition."""

    name: str
    reward_mode: str  # Override for all curriculum phases
    use_curriculum: bool = True
    w_audio: float = 0.7
    w_visual: float = 0.3


# Standard ablation conditions
STANDARD_ABLATIONS = [
    AblationCondition(name="combined", reward_mode="combined"),
    AblationCondition(name="audio_only", reward_mode="audio_only", w_visual=0.0),
    AblationCondition(name="visual_only", reward_mode="visual_only", w_audio=0.0),
    AblationCondition(
        name="no_curriculum",
        reward_mode="combined",
        use_curriculum=False,
    ),
]


def run_ablation(
    condition: AblationCondition,
    base_config: TrainerConfig,
    num_episodes: int = 100,
) -> dict[str, Any]:
    """Run a single ablation condition.

    Args:
        condition: Ablation condition to run.
        base_config: Base training config (modified per condition).
        num_episodes: Number of episodes to train.

    Returns:
        Results dict with condition name, rewards, and metrics.
    """
    config = TrainerConfig(
        num_envs=base_config.num_envs,
        episode_length=base_config.episode_length,
        seed=base_config.seed,
        w_audio=condition.w_audio,
        w_visual=condition.w_visual,
        log_dir=base_config.log_dir,
        checkpoint_dir=base_config.checkpoint_dir,
        experiment_name=f"ablation_{condition.name}",
        use_real_sparc=base_config.use_real_sparc,
        use_real_sylber=base_config.use_real_sylber,
    )

    trainer = OpenJawTrainer(config)
    trainer.setup()

    # Override curriculum if needed
    if not condition.use_curriculum:
        single_phase = CurriculumPhase(
            name="no_curriculum",
            episodes=num_episodes + 1,
            reward_mode=condition.reward_mode,
        )
        trainer.curriculum = CurriculumScheduler(phases=[single_phase])
        trainer._update_reward_mode()

    try:
        results = trainer.train(num_episodes)
        rewards = [r["reward_total"] for r in results]

        return {
            "condition": condition.name,
            "num_episodes": num_episodes,
            "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "final_reward": float(rewards[-1]) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "rewards": rewards,
        }
    finally:
        trainer.close()


def run_all_ablations(
    base_config: TrainerConfig,
    num_episodes: int = 100,
    conditions: list[AblationCondition] | None = None,
) -> list[dict[str, Any]]:
    """Run all standard ablation conditions.

    Returns:
        List of result dicts, one per condition.
    """
    if conditions is None:
        conditions = STANDARD_ABLATIONS

    results = []
    for condition in conditions:
        logger.info(f"Running ablation: {condition.name}")
        result = run_ablation(condition, base_config, num_episodes)
        results.append(result)
        logger.info(
            f"  {condition.name}: mean_reward={result['mean_reward']:.4f}, "
            f"final_reward={result['final_reward']:.4f}"
        )

    return results
