"""Step 13: Ablation framework tests."""

from __future__ import annotations

import pytest

from openjaw.evaluation.ablations import (
    STANDARD_ABLATIONS,
    AblationCondition,
    run_ablation,
    run_all_ablations,
)
from openjaw.training.trainer import TrainerConfig


class TestAblationCondition:
    def test_default_fields(self):
        cond = AblationCondition(name="test", reward_mode="combined")
        assert cond.name == "test"
        assert cond.use_curriculum is True
        assert cond.w_audio == 0.7
        assert cond.w_visual == 0.3

    def test_custom_fields(self):
        cond = AblationCondition(
            name="visual_only", reward_mode="visual_only",
            w_audio=0.0, w_visual=1.0,
        )
        assert cond.w_audio == 0.0
        assert cond.w_visual == 1.0


class TestStandardAblations:
    def test_four_conditions(self):
        assert len(STANDARD_ABLATIONS) == 4

    def test_condition_names(self):
        names = {c.name for c in STANDARD_ABLATIONS}
        assert names == {"combined", "audio_only", "visual_only", "no_curriculum"}

    def test_audio_only_weights(self):
        audio = next(c for c in STANDARD_ABLATIONS if c.name == "audio_only")
        assert audio.w_visual == 0.0

    def test_visual_only_weights(self):
        visual = next(c for c in STANDARD_ABLATIONS if c.name == "visual_only")
        assert visual.w_audio == 0.0

    def test_no_curriculum_flag(self):
        no_curr = next(c for c in STANDARD_ABLATIONS if c.name == "no_curriculum")
        assert no_curr.use_curriculum is False


class TestRunAblation:
    @pytest.fixture
    def base_config(self, tmp_path):
        return TrainerConfig(
            num_envs=1,
            episode_length=5,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_name="test_ablation",
        )

    def test_single_ablation(self, base_config):
        condition = AblationCondition(name="test_combined", reward_mode="combined")
        result = run_ablation(condition, base_config, num_episodes=3)
        assert result["condition"] == "test_combined"
        assert result["num_episodes"] == 3
        assert isinstance(result["mean_reward"], float)
        assert isinstance(result["final_reward"], float)
        assert isinstance(result["max_reward"], float)
        assert len(result["rewards"]) == 3

    def test_audio_only_ablation(self, base_config):
        condition = AblationCondition(
            name="audio_only", reward_mode="audio_only", w_visual=0.0,
        )
        result = run_ablation(condition, base_config, num_episodes=2)
        assert result["condition"] == "audio_only"
        assert len(result["rewards"]) == 2

    def test_no_curriculum_ablation(self, base_config):
        condition = AblationCondition(
            name="no_curriculum", reward_mode="combined", use_curriculum=False,
        )
        result = run_ablation(condition, base_config, num_episodes=2)
        assert result["condition"] == "no_curriculum"


class TestRunAllAblations:
    def test_run_custom_conditions(self, tmp_path):
        config = TrainerConfig(
            num_envs=1,
            episode_length=5,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_name="test_all",
        )
        conditions = [
            AblationCondition(name="c1", reward_mode="combined"),
            AblationCondition(name="c2", reward_mode="audio_only", w_visual=0.0),
        ]
        results = run_all_ablations(config, num_episodes=2, conditions=conditions)
        assert len(results) == 2
        assert results[0]["condition"] == "c1"
        assert results[1]["condition"] == "c2"
