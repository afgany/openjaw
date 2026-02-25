"""Training logger: TensorBoard and optional W&B support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Unified logging to TensorBoard (and optionally W&B).

    Logs scalars, histograms, and text for training monitoring.
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        experiment_name: str = "openjaw",
        use_wandb: bool = False,
        wandb_project: str = "openjaw",
    ) -> None:
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

        # W&B (optional)
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=experiment_name)
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")

        self._step = 0

    @property
    def global_step(self) -> int:
        return self._step

    def set_step(self, step: int) -> None:
        self._step = step

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a scalar value."""
        s = step if step is not None else self._step
        self._writer.add_scalar(tag, value, s)
        if self._wandb:
            self._wandb.log({tag: value}, step=s)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalars under a group."""
        s = step if step is not None else self._step
        for key, val in values.items():
            self._writer.add_scalar(f"{main_tag}/{key}", val, s)
        if self._wandb:
            self._wandb.log({f"{main_tag}/{k}": v for k, v in values.items()}, step=s)

    def log_episode(
        self,
        episode: int,
        reward_total: float,
        reward_audio: float,
        reward_visual: float,
        reward_aux: float,
        episode_length: int,
        curriculum_phase: str = "",
        extra: dict[str, float] | None = None,
    ) -> None:
        """Log a complete episode summary."""
        self.log_scalars("reward", {
            "total": reward_total,
            "audio": reward_audio,
            "visual": reward_visual,
            "auxiliary": reward_aux,
        }, step=episode)
        self.log_scalar("episode/length", episode_length, step=episode)
        if curriculum_phase:
            self.log_scalar("episode/phase", hash(curriculum_phase) % 100, step=episode)
        if extra:
            self.log_scalars("extra", extra, step=episode)

    def log_training_step(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        learning_rate: float,
    ) -> None:
        """Log PPO training step metrics."""
        self.log_scalars("train", {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "learning_rate": learning_rate,
        }, step=step)

    def close(self) -> None:
        """Flush and close all loggers."""
        self._writer.close()
        if self._wandb:
            self._wandb.finish()
