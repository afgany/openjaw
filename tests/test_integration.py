"""Step 15: End-to-end integration test.

Full pipeline: create env → train episodes → compute metrics → visualize.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openjaw.audio.sparc_decoder import create_sparc_decoder
from openjaw.core.types import NUM_DOF
from openjaw.data.custom_speaker import CustomSpeakerLoader
from openjaw.data.preprocessing import create_vowel_targets
from openjaw.data.vocaset import VOCASETLoader
from openjaw.env.mouth_env import MouthEnv
from openjaw.evaluation.ablations import AblationCondition, run_ablation
from openjaw.evaluation.metrics import lip_vertex_error, mel_cepstral_distortion, sylber_cosine_similarity
from openjaw.evaluation.visualization import plot_reward_curves, plot_spectrogram_comparison
from openjaw.perception.sylber import MockSylberEncoder, create_sylber_encoder
from openjaw.policy.networks import ActorCritic
from openjaw.reward.combined import CombinedReward
from openjaw.training.curriculum import CurriculumScheduler
from openjaw.training.trainer import OpenJawTrainer, TrainerConfig
from openjaw.visual.flame_adapter import FLAMEAdapter


class TestFullPipeline:
    """End-to-end pipeline: load data → create env → train → evaluate → visualize."""

    @pytest.fixture
    def tmp_dirs(self, tmp_path):
        return {
            "logs": str(tmp_path / "logs"),
            "checkpoints": str(tmp_path / "checkpoints"),
            "figures": str(tmp_path / "figures"),
        }

    def test_full_pipeline(self, tmp_dirs, tmp_path):
        """Complete training → evaluation → visualization pipeline."""
        # ── 1. Data preparation ──────────────────────────────────────
        encoder = MockSylberEncoder()
        vowel_targets = create_vowel_targets(encoder)
        assert len(vowel_targets) == 5

        mock_seq = VOCASETLoader.create_mock_sequence(duration=1.0)
        assert mock_seq["audio"].dtype == np.float32

        mock_recording, sr = CustomSpeakerLoader.create_mock_recording(duration=0.5)
        assert len(mock_recording) > 0

        # ── 2. Environment + SPARC + Reward ──────────────────────────
        env = MouthEnv(render_mode="rgb_array", episode_length=10)
        sparc = create_sparc_decoder(use_real=False)
        sylber = create_sylber_encoder(use_real=False)
        flame = FLAMEAdapter(env.cavity.model)

        obs, info = env.reset(seed=42)
        assert obs.shape[0] == 1353  # 15*39 + 768

        # Step through environment, generate audio, compute reward
        reward_module = CombinedReward(
            sylber_encoder=sylber, mode="combined",
        )
        target_emb = vowel_targets[0].audio_embedding

        total_reward = 0.0
        prev_action = np.zeros(NUM_DOF, dtype=np.float32)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)

            positions = info["positions"]
            vocal_loudness = info["vocal_loudness"]
            lip_verts = flame.get_lip_vertices(env.cavity.data)
            audio = sparc.from_articulatory_state(positions, vocal_loudness)

            reward_out = reward_module.compute(
                generated_audio=audio,
                target_audio_embedding=target_emb,
                generated_lip_vertices=lip_verts,
                target_lip_vertices=lip_verts,  # Self-imitation
                action=action,
                prev_action=prev_action,
            )
            total_reward += reward_out.total
            prev_action = action.copy()

        env.close()
        assert isinstance(total_reward, float)

        # ── 3. Policy network forward pass ───────────────────────────
        import torch
        ac = ActorCritic(obs_dim=1353, action_dim=NUM_DOF)
        obs_tensor = torch.randn(1, 1353)
        dist = ac.get_action_distribution(obs_tensor)
        action_tensor = dist.sample()
        assert action_tensor.shape == (1, NUM_DOF)
        value = ac.critic(obs_tensor)
        assert value.shape == (1, 1)

        # ── 4. Trainer (short run) ───────────────────────────────────
        config = TrainerConfig(
            num_envs=1,
            episode_length=5,
            seed=42,
            log_dir=tmp_dirs["logs"],
            checkpoint_dir=tmp_dirs["checkpoints"],
            experiment_name="integration_test",
        )
        trainer = OpenJawTrainer(config)
        trainer.setup()
        results = trainer.train(num_episodes=5)
        assert len(results) == 5

        # Checkpoint round-trip
        ckpt_path = trainer.save_checkpoint()
        assert Path(ckpt_path).exists()
        trainer.close()

        # ── 5. Evaluation metrics ────────────────────────────────────
        rng = np.random.default_rng(42)
        a1 = (rng.standard_normal(16000) * 0.3).astype(np.float32)
        a2 = (rng.standard_normal(16000) * 0.3).astype(np.float32)

        mcd = mel_cepstral_distortion(a1, a2)
        assert isinstance(mcd, float) and mcd >= 0.0

        emb1 = rng.standard_normal(768).astype(np.float32)
        emb2 = rng.standard_normal(768).astype(np.float32)
        sim = sylber_cosine_similarity(emb1, emb2)
        assert -1.0 <= sim <= 1.0

        v1 = rng.standard_normal((5, 3)).astype(np.float32)
        v2 = rng.standard_normal((5, 3)).astype(np.float32)
        lve = lip_vertex_error(v1, v2)
        assert lve >= 0.0

        # ── 6. Visualization ─────────────────────────────────────────
        reward_history = {
            "integration": [r["reward_total"] for r in results],
        }
        fig_path = plot_reward_curves(
            reward_history,
            output_path=str(tmp_path / "figures" / "reward.pdf"),
            window=2,
        )
        assert Path(fig_path).exists()

        spec_path = plot_spectrogram_comparison(
            a1[:8000], a2[:8000],
            output_path=str(tmp_path / "figures" / "spec.pdf"),
        )
        assert Path(spec_path).exists()

    def test_curriculum_integration(self):
        """Curriculum transitions work across training."""
        from openjaw.training.curriculum import CurriculumPhase

        phases = [
            CurriculumPhase(name="p1", episodes=3, reward_mode="binary_sound"),
            CurriculumPhase(name="p2", episodes=5, reward_mode="combined"),
        ]
        sched = CurriculumScheduler(phases=phases)

        phase_names = []
        for _ in range(8):
            phase_names.append(sched.current_phase.name)
            sched.step()

        assert phase_names[:3] == ["p1", "p1", "p1"]
        assert phase_names[3:] == ["p2", "p2", "p2", "p2", "p2"]
        assert sched.is_complete

    def test_ablation_integration(self, tmp_path):
        """Ablation framework produces valid results."""
        config = TrainerConfig(
            num_envs=1,
            episode_length=5,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            experiment_name="ablation_int",
        )
        condition = AblationCondition(name="test", reward_mode="combined")
        result = run_ablation(condition, config, num_episodes=2)
        assert result["condition"] == "test"
        assert len(result["rewards"]) == 2
        assert isinstance(result["mean_reward"], float)

    def test_data_to_training_pipeline(self, tmp_path):
        """Data preprocessing feeds into trainer correctly."""
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder, vowels=["a", "o"])

        config = TrainerConfig(
            num_envs=1,
            episode_length=5,
            seed=42,
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            experiment_name="data_pipe",
        )
        trainer = OpenJawTrainer(config)
        trainer.setup()

        # Use vowel target as episode target
        target_emb = targets[0].audio_embedding
        result = trainer.run_episode(target_audio_emb=target_emb)
        assert result["steps"] == 5
        assert isinstance(result["reward_total"], float)
        trainer.close()
