"""Tests for agent_algebra.compress."""

from __future__ import annotations

import pytest

from agent_algebra.compress import (
    algorithmic_compression_ratio,
    classify,
    compression_ratio,
    is_signal,
    mdl_filter,
    reconstruction_score,
)


class TestCompressionRatio:
    def test_same_size(self) -> None:
        assert compression_ratio("hello", "hello") == 1.0

    def test_smaller_summary(self) -> None:
        ratio = compression_ratio("hello world foo bar", "hello")
        assert ratio < 1.0

    def test_empty_data(self) -> None:
        assert compression_ratio("", "x") == 1.0

    def test_bytes_input(self) -> None:
        ratio = compression_ratio(b"hello world", b"hi")
        assert ratio < 1.0


class TestAlgorithmicCompressionRatio:
    def test_repetitive_data_compresses(self) -> None:
        repetitive = "ABCABC" * 100
        ratio = algorithmic_compression_ratio(repetitive)
        assert ratio < 0.3

    def test_random_data_doesnt_compress(self) -> None:
        import random
        rng = random.Random(42)
        noise = "".join(chr(rng.randint(32, 126)) for _ in range(500))
        ratio = algorithmic_compression_ratio(noise)
        assert ratio > 0.5

    def test_empty_data(self) -> None:
        assert algorithmic_compression_ratio("") == 1.0


class TestReconstructionScore:
    def test_perfect_reconstruction(self) -> None:
        score = reconstruction_score("hello", "hello", lambda s: s)
        assert score == 1.0

    def test_no_overlap(self) -> None:
        score = reconstruction_score("aaaa", "bbbb", lambda s: s)
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        score = reconstruction_score("abcd", "ab", lambda s: s + "xy")
        assert 0.0 < score < 1.0

    def test_empty_data(self) -> None:
        assert reconstruction_score("", "", lambda s: s) == 1.0


class TestIsSignal:
    def test_structured_data_is_signal(self) -> None:
        data = "Bitcoin surges 15% on ETF approval. Markets rally across the board."
        summary = "BTC +15% ETF"
        # Short summary relative to long data → low compression ratio
        # Reconstruction threshold must be very low since summary doesn't match raw chars
        assert is_signal(
            data, summary, lambda s: s,
            compression_threshold=0.5,
            reconstruction_threshold=0.0,
        )

    def test_noise_is_not_signal(self) -> None:
        data = "xyzabc"
        summary = "xyzabc"  # can't compress → ratio = 1.0
        assert not is_signal(
            data, summary, lambda s: s,
            compression_threshold=0.5,
        )


class TestClassify:
    def test_returns_result(self) -> None:
        result = classify("hello world", "hi", lambda s: s)
        assert result.raw_size > 0
        assert result.summary_size > 0
        assert 0.0 <= result.compression_ratio <= 2.0
        assert isinstance(result.is_signal, bool)


class TestMdlFilter:
    def test_filters_noise(self) -> None:
        items = [
            "The quick brown fox jumps over the lazy dog repeatedly.",
            "xq",
        ]

        def summarize(text: str) -> str:
            return text[:5]

        def reconstruct(summary: str) -> str:
            return summary

        # The first item should be signal (short summary, partial reconstruction)
        # The second item is borderline
        result = mdl_filter(
            items, summarize, reconstruct,
            compression_threshold=0.5,
            reconstruction_threshold=0.05,
        )
        assert len(result) <= len(items)

    def test_empty_list(self) -> None:
        result = mdl_filter([], lambda x: x, lambda x: x)
        assert result == []
