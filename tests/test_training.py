"""Step 10: Training pipeline tests (trainer, curriculum, logger)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from openjaw.training.curriculum import CurriculumPhase, CurriculumScheduler, DEFAULT_PHASES
from openjaw.training.logger import TrainingLogger
from openjaw.training.trainer import OpenJawTrainer, TrainerConfig


# ── Curriculum ──────────────────────────────────────────────────────────────

class TestCurriculumPhase:
    def test_dataclass_fields(self):
        phase = CurriculumPhase(name="test", episodes=100, reward_mode="combined")
        assert phase.name == "test"
        assert phase.episodes == 100
        assert phase.reward_mode == "combined"
        assert phase.targets == []

    def test_targets_default_empty(self):
        phase = CurriculumPhase(name="a", episodes=1, reward_mode="binary_sound")
        assert phase.targets == []

    def test_targets_specified(self):
        phase = CurriculumPhase(
            name="vowels", episodes=500, reward_mode="audio_only",
            targets=["a", "e", "i"],
        )
        assert len(phase.targets) == 3


class TestCurriculumScheduler:
    def test_default_phases(self):
        sched = CurriculumScheduler()
        assert len(sched.phases) == 4
        assert sched.phases[0].name == "babbling"

    def test_custom_phases(self):
        phases = [
            CurriculumPhase(name="p1", episodes=5, reward_mode="binary_sound"),
            CurriculumPhase(name="p2", episodes=10, reward_mode="combined"),
        ]
        sched = CurriculumScheduler(phases=phases)
        assert len(sched.phases) == 2

    def test_initial_state(self):
        sched = CurriculumScheduler()
        assert sched.phase_index == 0
        assert sched.total_episodes == 0
        assert not sched.is_complete

    def test_step_within_phase(self):
        phases = [CurriculumPhase(name="p1", episodes=3, reward_mode="combined")]
        sched = CurriculumScheduler(phases=phases)
        changed = sched.step()
        assert not changed
        assert sched.phase_index == 0

    def test_step_transitions_phase(self):
        phases = [
            CurriculumPhase(name="p1", episodes=2, reward_mode="binary_sound"),
            CurriculumPhase(name="p2", episodes=5, reward_mode="combined"),
        ]
        sched = CurriculumScheduler(phases=phases)
        sched.step()  # episode 1 in p1
        changed = sched.step()  # episode 2 in p1 → transition
        assert changed
        assert sched.phase_index == 1
        assert sched.current_phase.name == "p2"

    def test_completion(self):
        phases = [CurriculumPhase(name="p1", episodes=1, reward_mode="combined")]
        sched = CurriculumScheduler(phases=phases)
        sched.step()  # completes p1
        assert sched.is_complete
        # Further steps should not crash
        changed = sched.step()
        assert not changed

    def test_reset(self):
        phases = [CurriculumPhase(name="p1", episodes=2, reward_mode="combined")]
        sched = CurriculumScheduler(phases=phases)
        sched.step()
        sched.step()
        assert sched.is_complete
        sched.reset()
        assert sched.phase_index == 0
        assert sched.total_episodes == 0
        assert not sched.is_complete

    def test_progress_dict(self):
        phases = [CurriculumPhase(name="p1", episodes=10, reward_mode="combined")]
        sched = CurriculumScheduler(phases=phases)
        sched.step()
        sched.step()
        prog = sched.progress()
        assert prog["phase"] == "p1"
        assert prog["phase_idx"] == 0
        assert prog["episodes_in_phase"] == 2
        assert prog["total_episodes"] == 2
        assert prog["phase_progress"] == pytest.approx(0.2)

    def test_progress_complete(self):
        phases = [CurriculumPhase(name="p1", episodes=1, reward_mode="combined")]
        sched = CurriculumScheduler(phases=phases)
        sched.step()
        prog = sched.progress()
        assert prog["phase"] == "complete"
        assert prog["phase_progress"] == 1.0

    def test_total_episodes_tracks_across_phases(self):
        phases = [
            CurriculumPhase(name="p1", episodes=2, reward_mode="binary_sound"),
            CurriculumPhase(name="p2", episodes=3, reward_mode="combined"),
        ]
        sched = CurriculumScheduler(phases=phases)
        for _ in range(5):
            sched.step()
        assert sched.total_episodes == 5
        assert sched.is_complete


# ── Logger ──────────────────────────────────────────────────────────────────

class TestTrainingLogger:
    def test_create_and_close(self, tmp_path):
        logger = TrainingLogger(
            log_dir=str(tmp_path), experiment_name="test_run"
        )
        assert (tmp_path / "test_run").exists()
        logger.close()

    def test_log_scalar(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        logger.log_scalar("train/loss", 0.5, step=1)
        logger.log_scalar("train/loss", 0.3, step=2)
        logger.close()

    def test_log_scalars(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        logger.log_scalars("reward", {"total": 1.0, "audio": 0.7, "visual": 0.3}, step=0)
        logger.close()

    def test_log_episode(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        logger.log_episode(
            episode=1,
            reward_total=5.0,
            reward_audio=3.5,
            reward_visual=1.0,
            reward_aux=-0.5,
            episode_length=50,
            curriculum_phase="babbling",
        )
        logger.close()

    def test_log_training_step(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        logger.log_training_step(
            step=100, policy_loss=0.01, value_loss=0.5,
            entropy=0.3, learning_rate=3e-4,
        )
        logger.close()

    def test_global_step(self, tmp_path):
        logger = TrainingLogger(log_dir=str(tmp_path), experiment_name="test")
        assert logger.global_step == 0
        logger.set_step(42)
        assert logger.global_step == 42
        logger.close()


# ── Trainer ─────────────────────────────────────────────────────────────────

class TestTrainerConfig:
    def test_defaults(self):
        config = TrainerConfig()
        assert config.num_envs == 16
        assert config.episode_length == 50
        assert config.w_audio == 0.7
        assert config.w_visual == 0.3
        assert config.seed == 42
        assert config.device == "cpu"
        assert not config.use_real_sparc
        assert not config.use_real_sylber

    def test_custom_values(self):
        config = TrainerConfig(num_envs=4, seed=123, w_audio=0.5, w_visual=0.5)
        assert config.num_envs == 4
        assert config.seed == 123


class TestOpenJawTrainer:
    @pytest.fixture
    def trainer(self, tmp_path):
        config = TrainerConfig(
            num_envs=1,
            episode_length=10,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_name="test_trainer",
        )
        t = OpenJawTrainer(config)
        yield t
        if t._setup_done:
            t.close()

    def test_setup(self, trainer):
        trainer.setup()
        assert trainer._setup_done
        assert trainer.env is not None
        assert trainer.sparc is not None
        assert trainer.sylber is not None
        assert trainer.curriculum is not None
        assert trainer.reward_module is not None

    def test_run_episode(self, trainer):
        trainer.setup()
        result = trainer.run_episode()
        assert "reward_total" in result
        assert "reward_audio" in result
        assert "reward_visual" in result
        assert "reward_aux" in result
        assert "steps" in result
        assert "positions_history" in result
        assert result["episode"] == 1
        assert result["steps"] == trainer.config.episode_length

    def test_train_multiple_episodes(self, trainer):
        trainer.setup()
        results = trainer.train(num_episodes=3)
        assert len(results) == 3
        assert all("reward_total" in r for r in results)
        assert trainer._episode_count == 3

    def test_checkpoint_save_load(self, trainer, tmp_path):
        trainer.setup()
        trainer.train(num_episodes=2)

        # Save
        ckpt_path = trainer.save_checkpoint()
        assert Path(ckpt_path).exists()

        # Load into fresh trainer
        config2 = TrainerConfig(
            num_envs=1,
            episode_length=10,
            seed=42,
            log_dir=str(tmp_path / "logs2"),
            checkpoint_dir=str(tmp_path / "checkpoints2"),
            experiment_name="test_load",
        )
        trainer2 = OpenJawTrainer(config2)
        trainer2.setup()
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2._episode_count == 2
        assert trainer2._total_steps == trainer._total_steps
        trainer2.close()

    def test_auto_setup(self, tmp_path):
        """Train without explicit setup should auto-setup."""
        config = TrainerConfig(
            num_envs=1,
            episode_length=5,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            experiment_name="auto",
        )
        t = OpenJawTrainer(config)
        result = t.run_episode()
        assert result["steps"] == 5
        t.close()

    def test_reward_mode_updates_with_curriculum(self, trainer):
        trainer.setup()
        initial_mode = trainer.reward_module.mode
        assert initial_mode == "binary_sound"  # Default first phase

    def test_positions_history_shape(self, trainer):
        trainer.setup()
        result = trainer.run_episode()
        history = result["positions_history"]
        assert history.shape[0] == trainer.config.episode_length
        assert history.shape[1] == 13  # 13 DOF
