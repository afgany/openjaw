"""MuJoCo offscreen rendering for visual feedback.

Captures RGB images of the simulated mouth from configurable camera views.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from openjaw.core.types import FloatArray, RENDER_HEIGHT, RENDER_WIDTH

logger = logging.getLogger(__name__)


class MouthRenderer:
    """Offscreen renderer for the MuJoCo oral cavity model.

    Produces RGB images for visual feedback and reward computation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = RENDER_WIDTH,
        height: int = RENDER_HEIGHT,
        camera_name: str = "frontal",
    ) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> FloatArray:
        """Render current simulation state to RGB image.

        Args:
            data: MuJoCo simulation data.

        Returns:
            RGB image as uint8 array, shape (H, W, 3).
        """
        self._renderer.update_scene(data, camera=self.camera_name)
        img = self._renderer.render()
        return np.asarray(img, dtype=np.uint8)

    def close(self) -> None:
        """Release rendering resources."""
        self._renderer.close()
