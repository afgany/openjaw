"""Visual encoder: extracts features from mouth mesh vertices.

Converts lip/mouth vertex positions into a fixed-size feature vector
for visual observation and LVE reward computation.

This is a lightweight encoder — for the commutative verification
reward (SelfTalk paradigm), a full lip-reading model would be needed.
"""

from __future__ import annotations

import numpy as np

from openjaw.core.types import FloatArray

LIP_FEATURE_DIM = 32  # Compact feature vector from lip geometry


class LipFeatureEncoder:
    """Extract geometric features from lip vertex positions.

    Computes a fixed-size feature vector from lip/mouth landmarks
    capturing opening, spread, protrusion, and shape statistics.
    """

    @property
    def feature_dim(self) -> int:
        return LIP_FEATURE_DIM

    def encode(self, lip_vertices: FloatArray, mouth_vertices: FloatArray | None = None) -> FloatArray:
        """Encode lip geometry into feature vector.

        Args:
            lip_vertices: Lip landmark positions, shape (N_lip, 3).
            mouth_vertices: Optional full mouth landmarks, shape (N_mouth, 3).

        Returns:
            Feature vector, shape (LIP_FEATURE_DIM,).
        """
        features = np.zeros(LIP_FEATURE_DIM, dtype=np.float32)

        if lip_vertices.shape[0] < 2:
            return features

        # Lip opening (distance between upper and lower lip)
        features[0] = float(np.linalg.norm(lip_vertices[0] - lip_vertices[1]))

        # Lip center position
        center = np.mean(lip_vertices, axis=0)
        features[1:4] = center

        # Lip spread (extent along Z axis)
        features[4] = float(np.ptp(lip_vertices[:, 2])) if lip_vertices.shape[0] > 1 else 0.0

        # Vertex positions flattened (up to dim budget)
        flat = lip_vertices.flatten()
        n = min(len(flat), LIP_FEATURE_DIM - 5)
        features[5:5 + n] = flat[:n]

        # If mouth vertices provided, add additional features
        if mouth_vertices is not None and mouth_vertices.shape[0] > 0:
            # Jaw opening (tongue tip to palate proxy)
            if mouth_vertices.shape[0] >= 6:
                tongue_tip = mouth_vertices[5]  # tongue_tip body
                jaw = mouth_vertices[2]  # jaw body
                idx = min(5 + n, LIP_FEATURE_DIM - 3)
                if idx + 3 <= LIP_FEATURE_DIM:
                    features[idx:idx + 3] = tongue_tip - jaw

        return features
