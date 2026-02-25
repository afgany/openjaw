"""Shared type aliases and constants for the OpenJaw framework."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Array types
FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int32]

# Dimensions (13-DOF Anand et al. 2025)
NUM_DOF = 13
FRAME_STACK_K = 15
CONTROL_FREQ_HZ = 25
PHYSICS_SUBSTEPS = 20
EPISODE_LENGTH = 50  # steps (2 seconds at 25 Hz)
DISCOUNT_GAMMA = 0.99

# Action bounds (articulator velocities)
ACTION_LOW = -0.5
ACTION_HIGH = 0.5

# Audio
AUDIO_SAMPLE_RATE = 16000

# Visual
RENDER_WIDTH = 256
RENDER_HEIGHT = 256
