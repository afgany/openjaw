"""Gymnasium environment wrapper for the MuJoCo oral cavity simulation.

Implements the standard Gymnasium interface with:
  - 13-DOF continuous action space (muscle activation velocities)
  - Frame-stacked observation space (state + target embedding)
  - Configurable episode length and physics substeps
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openjaw.core.types import (
    ACTION_HIGH,
    ACTION_LOW,
    EPISODE_LENGTH,
    FloatArray,
    FRAME_STACK_K,
    NUM_DOF,
)
from openjaw.env.oral_cavity import OralCavityModel


class MouthEnv(gym.Env):
    """OpenJaw mouth simulation environment.

    Observation: frame-stacked [positions(13) + velocities(13) + prev_action(13)] + target_embed
    Action: 13-DOF continuous muscle activations in [-0.5, 0.5]
    Reward: placeholder (0.0) — replaced by the reward module during training
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_stack_k: int = FRAME_STACK_K,
        episode_length: int = EPISODE_LENGTH,
        physics_substeps: int = 20,
        target_embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.frame_stack_k = frame_stack_k
        self.episode_length = episode_length
        self.physics_substeps = physics_substeps
        self.target_embed_dim = target_embed_dim

        # Create physics model
        self.cavity = OralCavityModel()

        # State dimensions
        self._state_dim = 3 * NUM_DOF  # pos + vel + prev_action = 39
        self._obs_dim = self.frame_stack_k * self._state_dim + self.target_embed_dim

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(NUM_DOF,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Episode state
        self._step_count = 0
        self._prev_action = np.zeros(NUM_DOF, dtype=np.float32)
        self._frame_buffer: deque[FloatArray] = deque(maxlen=self.frame_stack_k)
        self._target_embed = np.zeros(self.target_embed_dim, dtype=np.float32)

        # Rendering
        self._renderer = None

    def set_target_embedding(self, embedding: FloatArray) -> None:
        """Set the target syllable embedding for reward computation."""
        assert embedding.shape == (self.target_embed_dim,)
        self._target_embed = embedding.astype(np.float32)

    def _get_state_vector(self) -> FloatArray:
        """Get current state: [positions, velocities, prev_action]."""
        pos = self.cavity.get_positions()
        vel = self.cavity.get_velocities()
        return np.concatenate([pos, vel, self._prev_action]).astype(np.float32)

    def _get_observation(self) -> FloatArray:
        """Get frame-stacked observation + target embedding."""
        stacked = np.concatenate(list(self._frame_buffer), axis=0)
        return np.concatenate([stacked, self._target_embed]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.cavity.reset()
        self._step_count = 0
        self._prev_action = np.zeros(NUM_DOF, dtype=np.float32)

        # Fill frame buffer with initial state
        initial_state = self._get_state_vector()
        self._frame_buffer.clear()
        for _ in range(self.frame_stack_k):
            self._frame_buffer.append(initial_state.copy())

        obs = self._get_observation()
        info: dict[str, Any] = {
            "positions": self.cavity.get_positions(),
            "vocal_loudness": self.cavity.vocal_loudness,
        }
        return obs, info

    def step(
        self, action: FloatArray
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Execute one control step.

        Args:
            action: 13-DOF action in [-0.5, 0.5].

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)

        # Step physics
        self.cavity.step(action, n_substeps=self.physics_substeps)
        self._step_count += 1

        # Update state
        state = self._get_state_vector()
        self._frame_buffer.append(state)
        self._prev_action = action.copy()

        # Observation
        obs = self._get_observation()

        # Placeholder reward (replaced by reward module during training)
        reward = 0.0

        # Episode termination
        terminated = False  # No early termination
        truncated = self._step_count >= self.episode_length

        info: dict[str, Any] = {
            "positions": self.cavity.get_positions(),
            "velocities": self.cavity.get_velocities(),
            "vocal_loudness": self.cavity.vocal_loudness,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> FloatArray | None:
        """Render the current mouth state."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.cavity.model, height=256, width=256
                )
            self._renderer.update_scene(self.cavity.data, camera="frontal")
            return self._renderer.render()
        return None

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Try importing mujoco.Renderer for render support
try:
    import mujoco as _mj
except ImportError:
    pass
