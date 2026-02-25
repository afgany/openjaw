"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_ep100.pt
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from openjaw.audio.sparc_decoder import create_sparc_decoder
from openjaw.evaluation.metrics import lip_vertex_error, mel_cepstral_distortion, sylber_cosine_similarity
from openjaw.perception.sylber import create_sylber_encoder

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenJaw model")
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    args = parser.parse_args()

    print("OpenJaw Evaluation")
    print("=" * 40)

    # Verify metrics work
    rng = np.random.default_rng(42)
    a1 = (rng.standard_normal(16000) * 0.3).astype(np.float32)
    a2 = (rng.standard_normal(16000) * 0.3).astype(np.float32)

    mcd = mel_cepstral_distortion(a1, a2)
    print(f"MCD (random vs random): {mcd:.2f} dB")

    emb1 = rng.standard_normal(768).astype(np.float32)
    emb2 = rng.standard_normal(768).astype(np.float32)
    sim = sylber_cosine_similarity(emb1, emb2)
    print(f"Sylber similarity (random): {sim:.4f}")

    v1 = rng.standard_normal((2, 3)).astype(np.float32) * 0.01
    v2 = rng.standard_normal((2, 3)).astype(np.float32) * 0.01
    lve = lip_vertex_error(v1, v2)
    print(f"LVE (random): {lve:.6f}")


if __name__ == "__main__":
    main()
