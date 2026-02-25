"""Evaluation metrics: MCD, LVE, CER, Sylber similarity.

Standard metrics for evaluating articulatory speech synthesis quality.
"""

from __future__ import annotations

import logging

import numpy as np

from openjaw.core.types import AUDIO_SAMPLE_RATE, FloatArray

logger = logging.getLogger(__name__)


def mel_cepstral_distortion(
    generated: FloatArray,
    reference: FloatArray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    n_mfcc: int = 13,
) -> float:
    """Compute Mel-Cepstral Distortion (MCD) between two audio signals.

    MCD = (10/ln(10)) * sqrt(2 * sum((mc_gen - mc_ref)^2))

    Lower is better. Typical good values: < 6 dB.

    Args:
        generated: Generated audio, shape (N,).
        reference: Reference audio, shape (M,).
        sample_rate: Audio sample rate.
        n_mfcc: Number of MFCCs to use.

    Returns:
        MCD in dB (averaged over frames).
    """
    import librosa

    # Extract MFCCs
    mfcc_gen = librosa.feature.mfcc(y=generated, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_ref = librosa.feature.mfcc(y=reference, sr=sample_rate, n_mfcc=n_mfcc)

    # Align lengths (truncate to shorter)
    min_len = min(mfcc_gen.shape[1], mfcc_ref.shape[1])
    if min_len == 0:
        return float("inf")
    mfcc_gen = mfcc_gen[:, :min_len]
    mfcc_ref = mfcc_ref[:, :min_len]

    # Skip 0th coefficient (energy)
    diff = mfcc_gen[1:] - mfcc_ref[1:]
    frame_mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))

    return float(np.mean(frame_mcd))


def lip_vertex_error(
    generated: FloatArray,
    target: FloatArray,
) -> float:
    """Compute Lip Vertex Error (LVE).

    Mean L2 distance between corresponding lip vertices.

    Args:
        generated: Generated lip positions, shape (N, 3) or (T, N, 3).
        target: Target lip positions, same shape as generated.

    Returns:
        Mean LVE (scalar).
    """
    if generated.shape != target.shape:
        raise ValueError(f"Shape mismatch: {generated.shape} vs {target.shape}")

    if generated.ndim == 2:
        # Single frame: (N, 3)
        return float(np.mean(np.linalg.norm(generated - target, axis=1)))
    elif generated.ndim == 3:
        # Sequence: (T, N, 3)
        per_frame = np.mean(np.linalg.norm(generated - target, axis=2), axis=1)
        return float(np.mean(per_frame))
    else:
        raise ValueError(f"Expected 2D or 3D array, got {generated.ndim}D")


def sylber_cosine_similarity(
    embedding_a: FloatArray,
    embedding_b: FloatArray,
) -> float:
    """Cosine similarity between Sylber embeddings.

    Args:
        embedding_a: Shape (768,).
        embedding_b: Shape (768,).

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def character_error_rate(
    generated_audio: FloatArray,
    reference_text: str,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    model_name: str = "openai/whisper-tiny",
) -> float:
    """Compute Character Error Rate (CER) via ASR on generated audio.

    Uses Whisper (or similar) to transcribe generated audio, then
    computes CER against reference text.

    Args:
        generated_audio: Generated audio, shape (N,).
        reference_text: Ground truth transcription.
        sample_rate: Audio sample rate.
        model_name: ASR model to use.

    Returns:
        CER (0.0 = perfect, 1.0+ = poor).
    """
    try:
        import torch
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model=model_name)
        result = asr({"raw": generated_audio, "sampling_rate": sample_rate})
        hypothesis = result["text"].strip().lower()
    except Exception as e:
        logger.warning(f"ASR failed: {e}. Returning CER=1.0")
        return 1.0

    reference = reference_text.strip().lower()

    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Levenshtein distance
    return _levenshtein_distance(hypothesis, reference) / max(len(reference), 1)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
