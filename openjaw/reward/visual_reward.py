"""Visual reward: 1 - normalized Lip Vertex Error (LVE).

R_visual = 1 - LVE_norm(mesh_gen, mesh_target) ∈ [0, 1]

LVE is the mean L2 distance between generated and target lip vertices,
normalized by a reference scale so that R_visual stays in [0, 1].
"""

from __future__ import annotations

import numpy as np

from openjaw.core.types import FloatArray


class VisualReward:
    """Compute visual similarity reward using Lip Vertex Error."""

    def __init__(self, max_lve: float = 0.05) -> None:
        """
        Args:
            max_lve: Maximum LVE value for normalization. Errors above this
                     are clipped to 0 reward. Default 0.05 (5cm in model units).
        """
        self.max_lve = max_lve

    def compute(
        self,
        generated_vertices: FloatArray,
        target_vertices: FloatArray,
    ) -> float:
        """Compute visual reward from lip vertex comparison.

        Args:
            generated_vertices: Generated lip positions, shape (N, 3).
            target_vertices: Target lip positions, shape (N, 3).

        Returns:
            Reward in [0, 1]. 1.0 = perfect match, 0.0 = max error.
        """
        lve = self._lip_vertex_error(generated_vertices, target_vertices)
        normalized = min(lve / self.max_lve, 1.0)
        return 1.0 - normalized

    @staticmethod
    def _lip_vertex_error(generated: FloatArray, target: FloatArray) -> float:
        """Mean L2 distance between corresponding vertices."""
        if generated.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: generated {generated.shape} vs target {target.shape}"
            )
        per_vertex = np.linalg.norm(generated - target, axis=1)
        return float(np.mean(per_vertex))
