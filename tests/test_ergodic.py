"""Tests for agent_algebra.ergodic."""

from __future__ import annotations

import pytest

from agent_algebra.ergodic import (
    ergodic_correction,
    ergodic_kelly,
    geometric_growth_rate,
    kelly_fraction,
    simulate_paths,
)


class TestKellyFraction:
    def test_coin_flip_even_odds(self) -> None:
        # Fair coin, even odds → don't bet
        assert kelly_fraction(0.5, 1.0) == 0.0

    def test_favorable_bet(self) -> None:
        # 60% win rate, 1:1 odds → f* = 0.60 - 0.40 = 0.20
        f = kelly_fraction(0.60, 1.0)
        assert abs(f - 0.20) < 1e-10

    def test_unfavorable_bet(self) -> None:
        # 40% win rate, 1:1 odds → negative → clamped to 0
        assert kelly_fraction(0.40, 1.0) == 0.0

    def test_zero_win_loss_ratio(self) -> None:
        assert kelly_fraction(0.60, 0.0) == 0.0

    def test_high_edge(self) -> None:
        f = kelly_fraction(0.70, 2.0)
        assert f > 0.3


class TestGeometricGrowthRate:
    def test_positive_returns(self) -> None:
        rate = geometric_growth_rate([0.1, 0.1, 0.1])
        assert rate > 0

    def test_mixed_returns(self) -> None:
        rate = geometric_growth_rate([0.1, -0.1, 0.1, -0.1])
        # Geometric mean of mixed returns should be slightly negative
        # (1.1 * 0.9 = 0.99 < 1.0)
        assert rate < 0.01

    def test_empty_returns_zero(self) -> None:
        assert geometric_growth_rate([]) == 0.0

    def test_total_loss(self) -> None:
        rate = geometric_growth_rate([-1.0])
        assert rate == float("-inf")


class TestSimulatePaths:
    def test_shape(self) -> None:
        returns = [0.01, -0.01, 0.02, -0.02, 0.01]
        paths = simulate_paths(returns, n_paths=10, n_steps=50, seed=42)
        assert len(paths) == 10
        assert len(paths[0]) == 51  # n_steps + 1 (initial wealth)

    def test_starts_at_one(self) -> None:
        paths = simulate_paths([0.01, -0.01], n_paths=5, n_steps=10, seed=42)
        for path in paths:
            assert path[0] == 1.0

    def test_empty_returns(self) -> None:
        paths = simulate_paths([], n_paths=3, n_steps=5)
        assert len(paths) == 3
        # All values should be 1.0
        assert all(v == 1.0 for path in paths for v in path)

    def test_reproducible(self) -> None:
        returns = [0.02, -0.01, 0.03, -0.02]
        p1 = simulate_paths(returns, n_paths=5, n_steps=20, seed=42)
        p2 = simulate_paths(returns, n_paths=5, n_steps=20, seed=42)
        assert p1 == p2


class TestErgodicCorrection:
    def test_correction_bounded(self) -> None:
        returns = [0.02, -0.01, 0.03, -0.02, 0.01] * 10
        correction = ergodic_correction(returns, n_paths=100, seed=42)
        assert 0.0 < correction <= 2.0

    def test_empty_returns_one(self) -> None:
        assert ergodic_correction([]) == 1.0


class TestErgodicKelly:
    def test_returns_result(self) -> None:
        returns = [0.02, -0.01, 0.03, -0.02, 0.01] * 20
        result = ergodic_kelly(0.60, 1.5, returns, seed=42)
        assert result.kelly_fraction > 0
        assert result.ergodic_fraction > 0
        assert result.ergodic_fraction <= result.kelly_fraction
        assert 0.1 <= result.correction_factor <= 1.0

    def test_no_returns(self) -> None:
        result = ergodic_kelly(0.60, 1.5, [])
        assert result.kelly_fraction > 0
        assert result.ergodic_fraction == result.kelly_fraction
        assert result.correction_factor == 1.0

    def test_zero_kelly(self) -> None:
        result = ergodic_kelly(0.40, 1.0, [0.01, -0.01])
        assert result.kelly_fraction == 0.0
        assert result.ergodic_fraction == 0.0
