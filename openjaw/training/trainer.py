"""PPO training loop for OpenJaw.

Connects the MuJoCo environment, SPARC audio decoder, perception encoders,
reward module, and policy network into a complete training pipeline.

Implements a custom PPO loop (not SB3) because rewards are computed
externally via SPARC + Sylber, not by env.step().
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
from openjaw.core.mdp import ObservationSpace
from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray, NUM_DOF
from openjaw.env.mouth_env import MouthEnv
from openjaw.perception.sylber import BaseSylberEncoder, create_sylber_encoder
from openjaw.policy.networks import ActorCritic
from openjaw.reward.combined import CombinedReward, RewardOutput
from openjaw.training.buffer import RolloutBuffer
from openjaw.training.curriculum import CurriculumPhase, CurriculumScheduler
from openjaw.training.logger import TrainingLogger
from openjaw.training.ppo import ppo_update
from openjaw.visual.flame_adapter import FLAMEAdapter

logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance (scalar)."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4  # small epsilon to avoid div-by-zero

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: float, clip: float = 10.0) -> float:
        std = max(np.sqrt(self.var), 1e-8)
        return float(np.clip((x - self.mean) / std, -clip, clip))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class VectorRunningMeanStd:
    """Welford's online algorithm for running mean/variance (vector)."""

    def __init__(self, shape: int) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x.astype(np.float64) - self.mean
        self.mean += delta / self.count
        delta2 = x.astype(np.float64) - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        std = np.maximum(np.sqrt(self.var), 1e-8)
        return np.clip((x.astype(np.float64) - self.mean) / std, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = d["count"]


@dataclass
class TrainerConfig:
    """Training configuration."""

    # Environment
    num_envs: int = 16
    episode_length: int = 50

    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    clip_range_vf: float | None = 0.2

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

        # Policy network + optimizer
        obs_dim = ObservationSpace().dim  # 1353
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=NUM_DOF,
        ).to(self.config.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )
        logger.info(f"Policy network: {self.policy.param_count()} parameters")

        # Observation normalization (running mean/std per dimension)
        self._obs_normalizer = VectorRunningMeanStd(obs_dim)

        # Reward normalization (running std)
        self._reward_normalizer = RunningMeanStd()

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

    def _collect_rollout(
        self,
        buffer: RolloutBuffer,
        target_audio_emb: FloatArray | None = None,
        target_lip_vertices: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Collect one rollout of experience using the current policy.

        Runs episodes until the buffer is full, computing rewards externally
        via _compute_step_reward() at each step.

        Returns:
            Rollout summary dict.
        """
        buffer.reset()
        episode_rewards: list[float] = []
        episode_count = 0

        while not buffer.full:
            # Episode setup
            t_audio = target_audio_emb if target_audio_emb is not None else np.zeros(768, dtype=np.float32)

            obs, info = self.env.reset(
                seed=self.config.seed + self._episode_count + episode_count
            )

            t_lip = (
                target_lip_vertices
                if target_lip_vertices is not None
                else self.flame.get_lip_vertices(self.env.cavity.data).copy()
            )

            prev_action = np.zeros(NUM_DOF, dtype=np.float32)
            ep_reward = 0.0
            done = False

            while not done and not buffer.full:
                # Normalize observation
                self._obs_normalizer.update(obs)
                obs_norm = self._obs_normalizer.normalize(obs)

                # Get action from policy
                obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.config.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.policy.get_action_distribution(obs_tensor)
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor).sum(dim=-1)
                    value = self.policy.critic(obs_tensor).squeeze()

                # Scale from tanh [-1,1] to env bounds [-0.5, 0.5]
                action_np = action_tensor.squeeze(0).cpu().numpy() * 0.5
                action_np = np.clip(action_np, -0.5, 0.5).astype(np.float32)

                # Environment step
                next_obs, _, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                self._total_steps += 1

                # Compute external reward
                positions = info["positions"]
                vocal_loudness = info["vocal_loudness"]
                lip_verts = self.flame.get_lip_vertices(self.env.cavity.data)

                reward_out = self._compute_step_reward(
                    positions=positions,
                    vocal_loudness=vocal_loudness,
                    lip_vertices=lip_verts,
                    target_audio_emb=t_audio,
                    target_lip_vertices=t_lip,
                    action=action_np,
                    prev_action=prev_action,
                )

                # Normalize reward (with clipping)
                self._reward_normalizer.update(reward_out.total)
                normalized_reward = self._reward_normalizer.normalize(reward_out.total)

                # Store transition (normalized obs)
                buffer.add(
                    obs=obs_norm,
                    action=action_np,
                    log_prob=log_prob.item(),
                    reward=normalized_reward,
                    value=value.item(),
                    done=done,
                )

                obs = next_obs
                prev_action = action_np.copy()
                ep_reward += reward_out.total

            episode_rewards.append(ep_reward)
            episode_count += 1
            self._episode_count += 1

            # Curriculum step
            phase_changed = self.curriculum.step()
            if phase_changed and not self.curriculum.is_complete:
                self._update_reward_mode()

        # Bootstrap value for GAE (use normalized obs)
        obs_norm = self._obs_normalizer.normalize(obs)
        obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.config.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.policy.critic(obs_tensor).squeeze().item()
        buffer.compute_returns_and_advantages(last_value)

        return {
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "num_episodes": episode_count,
            "mean_episode_length": buffer.pos / max(episode_count, 1),
        }

    def train_ppo(
        self,
        num_episodes: int,
        target_audio_emb: FloatArray | None = None,
        target_lip_vertices: FloatArray | None = None,
    ) -> list[dict[str, Any]]:
        """Run PPO training for the specified number of episodes.

        Collects rollouts using the policy, computes GAE advantages,
        and performs PPO clipped updates.

        Args:
            num_episodes: Approximate number of episodes to train.
            target_audio_emb: Target embedding (optional).
            target_lip_vertices: Target lip vertices (optional).

        Returns:
            List of per-rollout summary dicts with loss metrics.
        """
        if not self._setup_done:
            self.setup()

        obs_dim = ObservationSpace().dim
        steps_per_rollout = self.config.episode_length * max(
            1, self.config.n_steps // self.config.episode_length
        )

        # Learning rate linear decay
        initial_lr = self.config.learning_rate

        results = []
        episodes_trained = 0
        rollout_idx = 0

        while episodes_trained < num_episodes:
            # Linear LR decay based on progress
            progress = min(episodes_trained / max(num_episodes, 1), 1.0)
            current_lr = initial_lr * (1.0 - progress)
            current_lr = max(current_lr, 1e-6)  # floor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            buffer = RolloutBuffer(
                buffer_size=steps_per_rollout,
                obs_dim=obs_dim,
                action_dim=NUM_DOF,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                device=self.config.device,
            )

            # Collect experience
            rollout_info = self._collect_rollout(
                buffer, target_audio_emb, target_lip_vertices,
            )
            episodes_trained += rollout_info["num_episodes"]

            # PPO update
            update_result = ppo_update(
                policy=self.policy,
                optimizer=self.optimizer,
                buffer=buffer,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                clip_range=self.config.clip_range,
                vf_coef=self.config.vf_coef,
                ent_coef=self.config.ent_coef,
                max_grad_norm=self.config.max_grad_norm,
                target_kl=self.config.target_kl,
                clip_range_vf=self.config.clip_range_vf,
            )

            # Log
            rollout_idx += 1
            self.logger.log_training_step(
                step=rollout_idx,
                policy_loss=update_result.policy_loss,
                value_loss=update_result.value_loss,
                entropy=update_result.entropy,
                learning_rate=current_lr,
            )
            self.logger.log_scalar(
                "train/explained_variance", update_result.explained_variance, step=rollout_idx,
            )
            self.logger.log_scalar(
                "reward/mean_episode", rollout_info["mean_episode_reward"], step=rollout_idx,
            )

            result = {
                "rollout": rollout_idx,
                "episodes_trained": episodes_trained,
                **rollout_info,
                "policy_loss": update_result.policy_loss,
                "value_loss": update_result.value_loss,
                "entropy": update_result.entropy,
                "explained_variance": update_result.explained_variance,
            }
            results.append(result)
            self._episode_rewards.append(rollout_info["mean_episode_reward"])

            # Checkpoint
            if self._episode_count % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            logger.info(
                f"Rollout {rollout_idx}: {rollout_info['num_episodes']} eps, "
                f"mean_reward={rollout_info['mean_episode_reward']:.4f}, "
                f"policy_loss={update_result.policy_loss:.4f}, "
                f"value_loss={update_result.value_loss:.4f}"
            )

        # Always save final checkpoint
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
            "reward_normalizer": self._reward_normalizer.state_dict(),
            "obs_normalizer": self._obs_normalizer.state_dict(),
        }
        if hasattr(self, "policy"):
            checkpoint["policy_state_dict"] = self.policy.state_dict()
        if hasattr(self, "optimizer"):
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
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
        if "reward_normalizer" in checkpoint:
            self._reward_normalizer.load_state_dict(checkpoint["reward_normalizer"])
        if "obs_normalizer" in checkpoint:
            self._obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        self._update_reward_mode()
        if "policy_state_dict" in checkpoint and hasattr(self, "policy"):
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded: {path} (episode {self._episode_count})")

    def close(self) -> None:
        """Clean up resources."""
        if self._setup_done:
            self.env.close()
            self.logger.close()
