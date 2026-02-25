"""FLAME mesh compatibility adapter.

Extracts lip vertices from MuJoCo simulation state and maps them
to a FLAME-compatible format for LVE (Lip Vertex Error) computation.

FLAME uses 5023 vertices. The lip region comprises ~200 vertices.
Since MuJoCo uses a simplified geometry, we extract the key lip landmarks
and provide them in a format compatible with LVE computation against
FLAME ground truth.

For full FLAME compatibility, a learned mapping would be trained —
for now, we extract the core lip landmark positions from MuJoCo
body positions as a simplified vertex set.
"""

from __future__ import annotations

import mujoco
import numpy as np

from openjaw.core.types import FloatArray

# Lip-related body names in the MuJoCo model
LIP_BODY_NAMES = [
    "upper_lip",
    "lower_lip",
]

# All mouth landmark body names for broader vertex extraction
MOUTH_BODY_NAMES = [
    "upper_lip",
    "lower_lip",
    "jaw",
    "tongue_dorsum",
    "tongue_blade",
    "tongue_tip",
    "teeth_upper",
    "teeth_lower",
]

# Number of landmark points we extract per body (center position = 1 point)
POINTS_PER_BODY = 1
NUM_LIP_LANDMARKS = len(LIP_BODY_NAMES) * POINTS_PER_BODY  # 2
NUM_MOUTH_LANDMARKS = len(MOUTH_BODY_NAMES) * POINTS_PER_BODY  # 8


class FLAMEAdapter:
    """Extract lip/mouth vertex positions from MuJoCo for LVE computation.

    This provides a simplified vertex extraction compatible with
    Lip Vertex Error metrics used in 3D talking face evaluation.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        # Cache body IDs
        self._lip_body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in LIP_BODY_NAMES
        ]
        self._mouth_body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in MOUTH_BODY_NAMES
        ]

    def get_lip_vertices(self, data: mujoco.MjData) -> FloatArray:
        """Extract lip landmark positions.

        Returns:
            Array of shape (NUM_LIP_LANDMARKS, 3) — XYZ positions of lip bodies.
        """
        vertices = np.zeros((NUM_LIP_LANDMARKS, 3), dtype=np.float32)
        for i, body_id in enumerate(self._lip_body_ids):
            vertices[i] = data.xpos[body_id]
        return vertices

    def get_mouth_vertices(self, data: mujoco.MjData) -> FloatArray:
        """Extract all mouth landmark positions.

        Returns:
            Array of shape (NUM_MOUTH_LANDMARKS, 3) — XYZ positions of all mouth bodies.
        """
        vertices = np.zeros((NUM_MOUTH_LANDMARKS, 3), dtype=np.float32)
        for i, body_id in enumerate(self._mouth_body_ids):
            vertices[i] = data.xpos[body_id]
        return vertices

    def compute_lip_opening(self, data: mujoco.MjData) -> float:
        """Compute lip opening distance (upper lip to lower lip).

        Returns:
            Scalar distance between upper and lower lip centers.
        """
        verts = self.get_lip_vertices(data)
        return float(np.linalg.norm(verts[0] - verts[1]))

    @staticmethod
    def lip_vertex_error(
        generated: FloatArray,
        target: FloatArray,
    ) -> float:
        """Compute Lip Vertex Error (LVE) between generated and target lip positions.

        LVE = mean L2 distance between corresponding lip vertices.

        Args:
            generated: Generated lip vertices, shape (N, 3).
            target: Target lip vertices, shape (N, 3).

        Returns:
            Mean L2 distance (mm scale depends on model units).
        """
        assert generated.shape == target.shape
        per_vertex_error = np.linalg.norm(generated - target, axis=1)
        return float(np.mean(per_vertex_error))
