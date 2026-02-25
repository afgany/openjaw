"""Step 6: Visual rendering and vertex extraction tests."""

import numpy as np

from openjaw.env.oral_cavity import OralCavityModel
from openjaw.visual.flame_adapter import (
    FLAMEAdapter,
    NUM_LIP_LANDMARKS,
    NUM_MOUTH_LANDMARKS,
)
from openjaw.visual.renderer import MouthRenderer


class TestMouthRenderer:
    def test_render_shape(self):
        """Offscreen render produces 256x256 RGB image."""
        model = OralCavityModel()
        model.reset()
        renderer = MouthRenderer(model.model, width=256, height=256)

        img = renderer.render(model.data)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8
        renderer.close()

    def test_render_after_action(self):
        """Image changes after simulation step."""
        model = OralCavityModel()
        model.reset()
        renderer = MouthRenderer(model.model)

        img1 = renderer.render(model.data).copy()

        action = np.zeros(13, dtype=np.float32)
        action[7] = -0.5  # Open jaw
        model.step(action, n_substeps=20)

        img2 = renderer.render(model.data)
        # Images should differ (jaw moved)
        assert not np.array_equal(img1, img2)
        renderer.close()

    def test_custom_resolution(self):
        model = OralCavityModel()
        model.reset()
        renderer = MouthRenderer(model.model, width=128, height=128)
        img = renderer.render(model.data)
        assert img.shape == (128, 128, 3)
        renderer.close()


class TestFLAMEAdapter:
    def test_lip_vertices_shape(self):
        model = OralCavityModel()
        model.reset()
        adapter = FLAMEAdapter(model.model)

        verts = adapter.get_lip_vertices(model.data)
        assert verts.shape == (NUM_LIP_LANDMARKS, 3)
        assert verts.dtype == np.float32

    def test_mouth_vertices_shape(self):
        model = OralCavityModel()
        model.reset()
        adapter = FLAMEAdapter(model.model)

        verts = adapter.get_mouth_vertices(model.data)
        assert verts.shape == (NUM_MOUTH_LANDMARKS, 3)
        assert verts.dtype == np.float32

    def test_lip_opening(self):
        model = OralCavityModel()
        model.reset()
        adapter = FLAMEAdapter(model.model)

        opening = adapter.compute_lip_opening(model.data)
        assert isinstance(opening, float)
        assert opening >= 0.0

    def test_vertices_change_with_action(self):
        """Lip vertices should change when jaw opens."""
        model = OralCavityModel()
        model.reset()
        adapter = FLAMEAdapter(model.model)

        verts_before = adapter.get_lip_vertices(model.data).copy()

        action = np.zeros(13, dtype=np.float32)
        action[7] = -0.5  # Open jaw (lower_incisor_y)
        model.step(action, n_substeps=40)

        verts_after = adapter.get_lip_vertices(model.data)
        assert not np.allclose(verts_before, verts_after, atol=1e-6)

    def test_lve_computation(self):
        """LVE between identical vertices is zero."""
        verts = np.random.randn(NUM_LIP_LANDMARKS, 3).astype(np.float32)
        lve = FLAMEAdapter.lip_vertex_error(verts, verts)
        assert lve == 0.0

    def test_lve_nonzero(self):
        """LVE between different vertices is positive."""
        v1 = np.zeros((NUM_LIP_LANDMARKS, 3), dtype=np.float32)
        v2 = np.ones((NUM_LIP_LANDMARKS, 3), dtype=np.float32)
        lve = FLAMEAdapter.lip_vertex_error(v1, v2)
        assert lve > 0.0
