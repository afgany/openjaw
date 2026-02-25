"""PPO training loop for OpenJaw.

Connects the MuJoCo environment, SPARC audio decoder, perception encoders,
reward module, and policy network into a complete training pipeline.

Uses Stable-Baselines3 PPO or a lightweight custom loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from openjaw.audio.sparc_decoder import BaseSPARCDecoder, create_sparc_decoder
from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray, NUM_DOF
from openjaw.env.mouth_env import MouthEnv
from openjaw.perception.sylber import BaseSylberEncoder, create_sylber_encoder
from openjaw.reward.combined import CombinedReward, RewardOutput
from openjaw.training.curriculum import CurriculumPhase, CurriculumScheduler
from openjaw.training.logger import TrainingLogger
from openjaw.visual.flame_adapter import FLAMEAdapter

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration."""

    # Environment
    num_envs: int = 16
    episode_length: int = 50

    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Reward
    w_audio: float = 0.7
    w_visual: float = 0.3
    lambda_silence: float = 1.0
    lambda_smooth: float = 0.01
    lambda_energy: float = 0.001

    # Training
    total_timesteps: int = 100_000
    checkpoint_interval: int = 1000
    log_interval: int = 10
    seed: int = 42

    # Paths
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    experiment_name: str = "openjaw_ppo"

    # Model options
    use_real_sparc: bool = False
    use_real_sylber: bool = False
    device: str = "cpu"


class OpenJawTrainer:
    """End-to-end PPO trainer for the OpenJaw environment.

    Manages:
      - Vectorized MuJoCo environments
      - SPARC audio decoding per step
      - Reward computation (audio + visual + auxiliary)
      - PPO policy updates via SB3
      - Curriculum phase transitions
      - Logging and checkpointing
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self._setup_done = False

    def setup(self) -> None:
        """Initialize all components."""
        logger.info("Setting up OpenJaw trainer...")

        # Seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Audio decoder
        self.sparc = create_sparc_decoder(
            use_real=self.config.use_real_sparc,
            device=self.config.device,
        )

        # Perception
        self.sylber = create_sylber_encoder(
            use_real=self.config.use_real_sylber,
            device=self.config.device,
        )

        # Curriculum
        self.curriculum = CurriculumScheduler()

        # Reward module (mode set by curriculum)
        self._update_reward_mode()

        # Logger
        self.logger = TrainingLogger(
            log_dir=self.config.log_dir,
            experiment_name=self.config.experiment_name,
        )

        # Checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Environment (single for custom training loop)
        import openjaw.env  # noqa: F401 — register env
        self.env = MouthEnv(
            render_mode="rgb_array",
            episode_length=self.config.episode_length,
        )

        # FLAME adapter for vertex extraction
        self.flame = FLAMEAdapter(self.env.cavity.model)

        # Tracking
        self._episode_count = 0
        self._total_steps = 0
        self._episode_rewards: list[float] = []

        self._setup_done = True
        logger.info("Trainer setup complete.")

    def _update_reward_mode(self) -> None:
        """Update reward module based on current curriculum phase."""
        phase = self.curriculum.current_phase
        self.reward_module = CombinedReward(
            sylber_encoder=self.sylber,
            w_audio=self.config.w_audio,
            w_visual=self.config.w_visual,
            lambda_silence=self.config.lambda_silence,
            lambda_smooth=self.config.lambda_smooth,
            lambda_energy=self.config.lambda_energy,
            mode=phase.reward_mode,
        )
        logger.info(f"Reward mode: {phase.reward_mode} (phase: {phase.name})")

    def _compute_step_reward(
        self,
        positions: FloatArray,
        vocal_loudness: float,
        lip_vertices: FloatArray,
        target_audio_emb: FloatArray,
        target_lip_vertices: FloatArray,
        action: FloatArray,
        prev_action: FloatArray,
    ) -> RewardOutput:
        """Compute reward for a single environment step."""
        # Generate audio from articulatory state
        audio = self.sparc.from_articulatory_state(
            positions, vocal_loudness
        )

        return self.reward_module.compute(
            generated_audio=audio,
            target_audio_embedding=target_audio_emb,
            generated_lip_vertices=lip_vertices,
            target_lip_vertices=target_lip_vertices,
            action=action,
            prev_action=prev_action,
        )

    def run_episode(
        self,
        target_audio_emb: FloatArray | None = None,
        target_lip_vertices: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Run a single episode, collecting rewards.

        Args:
            target_audio_emb: Target Sylber embedding. Uses zeros if None.
            target_lip_vertices: Target lip vertices. Uses initial pose if None.

        Returns:
            Episode summary dict.
        """
        if not self._setup_done:
            self.setup()

        if target_audio_emb is None:
            target_audio_emb = np.zeros(768, dtype=np.float32)
        if target_lip_vertices is None:
            target_lip_vertices = self.flame.get_lip_vertices(self.env.cavity.data).copy()

        obs, info = self.env.reset(seed=self.config.seed + self._episode_count)
        prev_action = np.zeros(NUM_DOF, dtype=np.float32)

        ep_reward_total = 0.0
        ep_reward_audio = 0.0
        ep_reward_visual = 0.0
        ep_reward_aux = 0.0
        ep_steps = 0
        positions_history = []

        done = False
        while not done:
            # Sample random action (will be replaced by policy in SB3 integration)
            action = self.env.action_space.sample()

            obs, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            ep_steps += 1
            self._total_steps += 1

            # Get state for reward computation
            positions = info["positions"]
            vocal_loudness = info["vocal_loudness"]
            lip_verts = self.flame.get_lip_vertices(self.env.cavity.data)
            positions_history.append(positions.copy())

            # Compute reward
            reward_out = self._compute_step_reward(
                positions=positions,
                vocal_loudness=vocal_loudness,
                lip_vertices=lip_verts,
                target_audio_emb=target_audio_emb,
                target_lip_vertices=target_lip_vertices,
                action=action,
                prev_action=prev_action,
            )

            ep_reward_total += reward_out.total
            ep_reward_audio += reward_out.audio
            ep_reward_visual += reward_out.visual
            ep_reward_aux += reward_out.auxiliary
            prev_action = action.copy()

        self._episode_count += 1

        # Log
        if self._episode_count % self.config.log_interval == 0:
            self.logger.log_episode(
                episode=self._episode_count,
                reward_total=ep_reward_total,
                reward_audio=ep_reward_audio,
                reward_visual=ep_reward_visual,
                reward_aux=ep_reward_aux,
                episode_length=ep_steps,
                curriculum_phase=self.curriculum.current_phase.name,
            )

        # Curriculum step
        phase_changed = self.curriculum.step()
        if phase_changed and not self.curriculum.is_complete:
            self._update_reward_mode()
            logger.info(
                f"Curriculum phase changed to: {self.curriculum.current_phase.name} "
                f"(episode {self._episode_count})"
            )

        self._episode_rewards.append(ep_reward_total)

        return {
            "episode": self._episode_count,
            "reward_total": ep_reward_total,
            "reward_audio": ep_reward_audio,
            "reward_visual": ep_reward_visual,
            "reward_aux": ep_reward_aux,
            "steps": ep_steps,
            "phase": self.curriculum.current_phase.name if not self.curriculum.is_complete else "complete",
            "positions_history": np.array(positions_history),
        }

    def train(self, num_episodes: int) -> list[dict[str, Any]]:
        """Run multiple training episodes.

        Args:
            num_episodes: Number of episodes to run.

        Returns:
            List of episode summaries.
        """
        if not self._setup_done:
            self.setup()

        results = []
        for i in range(num_episodes):
            result = self.run_episode()
            results.append(result)

            # Checkpoint
            if self._episode_count % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

        return results

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save training state checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        if path is None:
            path = str(
                Path(self.config.checkpoint_dir)
                / f"checkpoint_ep{self._episode_count}.pt"
            )

        checkpoint = {
            "episode_count": self._episode_count,
            "total_steps": self._total_steps,
            "curriculum_phase_idx": self.curriculum.phase_index,
            "curriculum_episodes_in_phase": self.curriculum._episodes_in_phase,
            "config": self.config,
            "episode_rewards": self._episode_rewards[-100:],  # Last 100
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load training state from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self._episode_count = checkpoint["episode_count"]
        self._total_steps = checkpoint["total_steps"]
        self.curriculum._current_phase_idx = checkpoint["curriculum_phase_idx"]
        self.curriculum._episodes_in_phase = checkpoint["curriculum_episodes_in_phase"]
        self.curriculum._total_episodes = self._episode_count
        self._episode_rewards = checkpoint.get("episode_rewards", [])
        self._update_reward_mode()
        logger.info(f"Checkpoint loaded: {path} (episode {self._episode_count})")

    def close(self) -> None:
        """Clean up resources."""
        if self._setup_done:
            self.env.close()
            self.logger.close()
