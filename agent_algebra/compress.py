"""MDL Signal/Noise Classification (Theorem 6) — Kolmogorov/Rissanen.

The best model compresses data most. If a pattern can be described
more concisely than the raw data, it's signal. Otherwise, noise.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class CompressionResult:
    """Result of MDL signal/noise classification."""

    raw_size: int
    summary_size: int
    compression_ratio: float
    reconstruction_score: float
    is_signal: bool


def compression_ratio(data: str | bytes, summary: str | bytes) -> float:
    """Compute compression ratio: |summary| / |data|.

    Lower ratio = more compression = more structure = more likely signal.
    """
    raw = data.encode() if isinstance(data, str) else data
    compressed = summary.encode() if isinstance(summary, str) else summary
    if len(raw) == 0:
        return 1.0
    return len(compressed) / len(raw)


def algorithmic_compression_ratio(data: str | bytes) -> float:
    """Estimate Kolmogorov complexity via zlib compression.

    High compressibility → structured data → likely signal.
    Low compressibility → random data → likely noise.

    Returns compressed_size / raw_size.
    """
    raw = data.encode() if isinstance(data, str) else data
    if len(raw) == 0:
        return 1.0
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)


def reconstruction_score(
    data: str,
    summary: str,
    reconstruct_fn: Callable[[str], str],
) -> float:
    """Score how well the summary can reconstruct the original data.

    reconstruct_fn: takes a summary, returns a reconstruction attempt.
    Score is 1 - (edit_distance_ratio), where edit distance is normalized
    by the length of the original.

    Returns a score in [0, 1] where 1 = perfect reconstruction.
    """
    reconstruction = reconstruct_fn(summary)
    if not data:
        return 1.0 if not reconstruction else 0.0

    # Simple character-level similarity (normalized)
    matches = sum(
        1 for a, b in zip(data, reconstruction) if a == b
    )
    max_len = max(len(data), len(reconstruction))
    if max_len == 0:
        return 1.0
    return matches / max_len


def is_signal(
    data: str,
    summary: str,
    reconstruct_fn: Callable[[str], str],
    compression_threshold: float = 0.5,
    reconstruction_threshold: float = 0.3,
) -> bool:
    """Determine if data contains signal (not noise).

    Signal = low compression ratio AND decent reconstruction.
    Both conditions must be met.
    """
    cr = compression_ratio(data, summary)
    rs = reconstruction_score(data, summary, reconstruct_fn)
    return cr < compression_threshold and rs >= reconstruction_threshold


def classify(
    data: str,
    summary: str,
    reconstruct_fn: Callable[[str], str],
    compression_threshold: float = 0.5,
    reconstruction_threshold: float = 0.3,
) -> CompressionResult:
    """Full MDL classification with detailed result."""
    cr = compression_ratio(data, summary)
    rs = reconstruction_score(data, summary, reconstruct_fn)
    return CompressionResult(
        raw_size=len(data.encode()),
        summary_size=len(summary.encode()),
        compression_ratio=cr,
        reconstruction_score=rs,
        is_signal=cr < compression_threshold and rs >= reconstruction_threshold,
    )


def mdl_filter(
    items: list[str],
    summarize_fn: Callable[[str], str],
    reconstruct_fn: Callable[[str], str],
    compression_threshold: float = 0.5,
    reconstruction_threshold: float = 0.3,
) -> list[str]:
    """Filter items, keeping only those that contain signal.

    summarize_fn: compresses an item into a summary
    reconstruct_fn: attempts to reconstruct the item from the summary

    Returns only items classified as signal.
    """
    results: list[str] = []
    for item in items:
        summary = summarize_fn(item)
        if is_signal(
            item, summary, reconstruct_fn,
            compression_threshold, reconstruction_threshold,
        ):
            results.append(item)
    return results
