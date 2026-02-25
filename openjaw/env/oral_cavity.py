"""MuJoCo oral cavity model: loader, state extraction, actuation mapping."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from openjaw.core.types import FloatArray, NUM_DOF

# Path to the MuJoCo XML asset
MOUTH_XML_PATH = Path(__file__).parent / "assets" / "mouth.xml"

# Number of physical actuators in the XML (12 joints, vocal loudness is virtual)
NUM_PHYSICAL_ACTUATORS = 12

# Joint names in order matching the 13-DOF articulator spec (minus vocal loudness)
JOINT_NAMES = [
    "tongue_dorsum_x", "tongue_dorsum_y",
    "tongue_blade_x", "tongue_blade_y",
    "tongue_tip_x", "tongue_tip_y",
    "jaw_x", "jaw_y",
    "upper_lip_x", "upper_lip_y",
    "lower_lip_x", "lower_lip_y",
]


class OralCavityModel:
    """MuJoCo oral cavity simulation model.

    Manages the MuJoCo model/data lifecycle and provides state extraction
    and actuation interfaces matching the 13-DOF MDP specification.
    """

    def __init__(self, xml_path: Path | None = None) -> None:
        self.xml_path = xml_path or MOUTH_XML_PATH
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Vocal loudness is stored separately (not a physical joint)
        self._vocal_loudness: float = 0.0

        # Cache joint IDs for fast access
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]

    @property
    def num_actuators(self) -> int:
        """Total DOF count including virtual vocal loudness."""
        return NUM_DOF

    @property
    def num_physical_actuators(self) -> int:
        """Number of physical MuJoCo actuators."""
        return NUM_PHYSICAL_ACTUATORS

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self._vocal_loudness = 0.0

    def step(self, action: FloatArray, n_substeps: int = 20) -> None:
        """Apply action and advance simulation.

        Args:
            action: 13-DOF action vector. First 12 are joint controls, last is vocal loudness.
            n_substeps: Number of physics substeps per control step.
        """
        assert action.shape == (NUM_DOF,), f"Expected action shape ({NUM_DOF},), got {action.shape}"

        # Apply physical actuator controls (first 12 DOFs)
        self.data.ctrl[:NUM_PHYSICAL_ACTUATORS] = action[:NUM_PHYSICAL_ACTUATORS]

        # Store vocal loudness (virtual, not a physical joint)
        self._vocal_loudness = float(action[12])

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def get_positions(self) -> FloatArray:
        """Get 13-DOF articulator positions.

        Returns:
            Array of shape (13,): 12 joint positions + vocal loudness.
        """
        positions = np.zeros(NUM_DOF, dtype=np.float32)
        for i, jid in enumerate(self._joint_ids):
            positions[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        positions[12] = self._vocal_loudness
        return positions

    def get_velocities(self) -> FloatArray:
        """Get 13-DOF articulator velocities.

        Returns:
            Array of shape (13,): 12 joint velocities + 0 for vocal loudness.
        """
        velocities = np.zeros(NUM_DOF, dtype=np.float32)
        for i, jid in enumerate(self._joint_ids):
            velocities[i] = self.data.qvel[self.model.jnt_dofadr[jid]]
        # Vocal loudness has no velocity (it's a scalar state)
        velocities[12] = 0.0
        return velocities

    def get_state(self) -> FloatArray:
        """Get full state vector [positions, velocities, vocal_loudness_scalar].

        Returns:
            Array of shape (39,): [13 positions, 13 velocities, 13 prev_action placeholder].
            Note: prev_action is NOT included here — it's managed by the env wrapper.
            This returns [positions(13), velocities(13)] = 26 dims.
            The env wrapper adds prev_action to make 39.
        """
        return np.concatenate([self.get_positions(), self.get_velocities()])

    @property
    def vocal_loudness(self) -> float:
        """Current vocal loudness value."""
        return self._vocal_loudness

    def has_nan(self) -> bool:
        """Check if simulation has diverged (NaN in state)."""
        return bool(np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)))
