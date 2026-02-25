"""Step 1: Scaffold tests — verify project structure and config loading."""

import importlib


def test_import_openjaw():
    """Verify openjaw package is importable."""
    import openjaw
    assert openjaw.__version__ == "0.1.0"


def test_import_subpackages():
    """Verify all subpackages are importable."""
    subpackages = [
        "openjaw.core",
        "openjaw.env",
        "openjaw.audio",
        "openjaw.visual",
        "openjaw.perception",
        "openjaw.reward",
        "openjaw.policy",
        "openjaw.training",
        "openjaw.data",
        "openjaw.evaluation",
    ]
    for pkg in subpackages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_config_loads():
    """Verify Hydra config loads correctly."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/default.yaml")
    assert cfg.project.name == "OpenJaw"
    assert cfg.mdp.dof == 13
    assert cfg.mdp.action_bounds == [-0.5, 0.5]
    assert cfg.mdp.frame_stack == 15
    assert cfg.reward.w_audio == 0.7
    assert cfg.reward.w_visual == 0.3
    assert cfg.policy.hidden_sizes == [256, 256]
