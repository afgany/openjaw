"""MDP formulation for articulatory speech imitation.

Defines the mathematical specification of the Markov Decision Process:
  - State space S: articulator positions + velocities + previous action
  - Action space A: 13-DOF continuous muscle activation velocities
  - Observation space O: frame-stacked state + target embedding
  - Reward R: multi-modal (audio + visual + auxiliary)
  - Transition T: MuJoCo physics step

Reference: Anand et al. (2025) "Teaching Machines to Speak Using Articulatory Control"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openjaw.core.types import (
    ACTION_HIGH,
    ACTION_LOW,
    DISCOUNT_GAMMA,
    EPISODE_LENGTH,
    FloatArray,
    FRAME_STACK_K,
    NUM_DOF,
)


@dataclass(frozen=True)
class Articulator:
    """Single articulatory degree of freedom."""

    name: str
    index: int
    dof_x: bool  # True if this is an X-axis DOF
    dof_y: bool  # True if this is a Y-axis DOF
    muscle_group: str  # Biomechanical muscle group
    description: str


# 13-DOF articulator definitions following Anand et al. (2025)
ARTICULATORS: tuple[Articulator, ...] = (
    Articulator("tongue_dorsum_x", 0, True, False, "genioglossus_posterior/styloglossus",
                "Tongue dorsum horizontal position"),
    Articulator("tongue_dorsum_y", 1, False, True, "genioglossus_posterior/styloglossus",
                "Tongue dorsum vertical position"),
    Articulator("tongue_blade_x", 2, True, False, "genioglossus_middle/verticalis",
                "Tongue blade horizontal position"),
    Articulator("tongue_blade_y", 3, False, True, "genioglossus_middle/verticalis",
                "Tongue blade vertical position"),
    Articulator("tongue_tip_x", 4, True, False, "genioglossus_anterior/superior_longitudinal",
                "Tongue tip horizontal position"),
    Articulator("tongue_tip_y", 5, False, True, "genioglossus_anterior/superior_longitudinal",
                "Tongue tip vertical position"),
    Articulator("lower_incisor_x", 6, True, False, "masseter/pterygoids/digastric",
                "Jaw (lower incisor) horizontal position"),
    Articulator("lower_incisor_y", 7, False, True, "masseter/pterygoids/digastric",
                "Jaw (lower incisor) vertical position"),
    Articulator("upper_lip_x", 8, True, False, "orbicularis_oris_superior/levator_labii",
                "Upper lip horizontal position"),
    Articulator("upper_lip_y", 9, False, True, "orbicularis_oris_superior/levator_labii",
                "Upper lip vertical position"),
    Articulator("lower_lip_x", 10, True, False, "orbicularis_oris_inferior/depressor_labii",
                "Lower lip horizontal position"),
    Articulator("lower_lip_y", 11, False, True, "orbicularis_oris_inferior/depressor_labii",
                "Lower lip vertical position"),
    Articulator("vocal_loudness", 12, False, False, "cricothyroid/thyroarytenoid",
                "Vocal loudness (glottal source amplitude)"),
)

ARTICULATOR_NAMES: tuple[str, ...] = tuple(a.name for a in ARTICULATORS)


@dataclass(frozen=True)
class StateSpace:
    """State space S: internal simulation state.

    s_t = [q_t, q̇_t, a_{t-1}] ∈ ℝ^{d_s}

    where:
      q_t     = articulator positions (13 DOF)
      q̇_t    = articulator velocities (13 DOF)
      a_{t-1} = previous action (13 DOF)
      d_s     = 3 * NUM_DOF = 39
    """

    num_dof: int = NUM_DOF
    positions_dim: int = field(init=False)
    velocities_dim: int = field(init=False)
    prev_action_dim: int = field(init=False)
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions_dim", self.num_dof)
        object.__setattr__(self, "velocities_dim", self.num_dof)
        object.__setattr__(self, "prev_action_dim", self.num_dof)
        object.__setattr__(self, "dim", 3 * self.num_dof)


@dataclass(frozen=True)
class ActionSpace:
    """Action space A: continuous muscle activation velocities.

    a_t ∈ [-0.5, 0.5]^{d_a}

    where d_a = NUM_DOF = 13
    Actions represent velocity commands for each articulator DOF.
    """

    num_dof: int = NUM_DOF
    dim: int = field(init=False)
    low: float = ACTION_LOW
    high: float = ACTION_HIGH

    def __post_init__(self) -> None:
        object.__setattr__(self, "dim", self.num_dof)

    def sample(self, rng: np.random.Generator | None = None) -> FloatArray:
        """Sample a random action."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, size=self.dim).astype(np.float32)

    def clip(self, action: FloatArray) -> FloatArray:
        """Clip action to valid bounds."""
        return np.clip(action, self.low, self.high).astype(np.float32)


@dataclass(frozen=True)
class ObservationSpace:
    """Observation space O: frame-stacked state + target embedding.

    o_t = [s_{t-K+1:t}, e_target] ∈ ℝ^{d_o}

    where:
      s_{t-K+1:t} = frame-stacked state vectors (K frames)
      e_target     = target syllable embedding (from Sylber)
      d_o          = K * d_s + d_target
    """

    state_dim: int = field(init=False)
    frame_stack_k: int = FRAME_STACK_K
    target_embed_dim: int = 768  # Sylber segment_features dimension (HuBERT-base hidden size)
    stacked_state_dim: int = field(init=False)
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        state_space = StateSpace()
        object.__setattr__(self, "state_dim", state_space.dim)
        object.__setattr__(self, "stacked_state_dim", self.frame_stack_k * state_space.dim)
        object.__setattr__(self, "dim", self.frame_stack_k * state_space.dim + self.target_embed_dim)


@dataclass(frozen=True)
class RewardSpec:
    """Reward function specification.

    R(s_t, a_t) = w_a * R_audio + w_v * R_visual + R_aux

    R_audio  = cos_sim(Sylber(audio_gen), Sylber(audio_target))  ∈ [-1, 1]
    R_visual = 1 - LVE_norm(mesh_gen, mesh_target)               ∈ [0, 1]
    R_aux    = -λ_silence * 𝟙[no_syllable]
               -λ_smooth  * ||a_t - a_{t-1}||²
               -λ_energy  * ||a_t||²
    """

    w_audio: float = 0.7
    w_visual: float = 0.3
    lambda_silence: float = 1.0
    lambda_smooth: float = 0.01
    lambda_energy: float = 0.001

    @property
    def weights_sum(self) -> float:
        return self.w_audio + self.w_visual


@dataclass(frozen=True)
class TransitionSpec:
    """Transition dynamics specification.

    s_{t+1} = PhysicsStep(s_t, a_t, Δt)

    Physics substeps per control step ensure simulation stability.
    """

    control_freq_hz: int = 25
    physics_substeps: int = 20
    episode_length: int = EPISODE_LENGTH
    discount: float = DISCOUNT_GAMMA

    @property
    def control_dt(self) -> float:
        """Control timestep in seconds."""
        return 1.0 / self.control_freq_hz

    @property
    def physics_dt(self) -> float:
        """Physics timestep in seconds."""
        return self.control_dt / self.physics_substeps

    @property
    def episode_duration_seconds(self) -> float:
        """Total episode duration in seconds."""
        return self.episode_length * self.control_dt


@dataclass(frozen=True)
class MDPSpec:
    """Complete MDP specification for the OpenJaw articulatory RL problem."""

    state: StateSpace = field(default_factory=StateSpace)
    action: ActionSpace = field(default_factory=ActionSpace)
    observation: ObservationSpace = field(default_factory=ObservationSpace)
    reward: RewardSpec = field(default_factory=RewardSpec)
    transition: TransitionSpec = field(default_factory=TransitionSpec)
