"""Tests for agent_algebra.contraction."""

from __future__ import annotations

import pytest

from agent_algebra.contraction import (
    ContractionResult,
    bayesian_seed_to_priors,
    contraction_loop,
    contraction_step,
    priors_to_bayesian_seed,
)


class TestContractionStep:
    def test_basic_blend(self) -> None:
        current = {"a": 0.50, "b": 0.50}
        realized = {"a": 0.70, "b": 0.30}
        new, dist = contraction_step(current, realized, k=0.5)
        assert abs(new["a"] - 0.60) < 0.01
        assert abs(new["b"] - 0.40) < 0.01
        assert dist > 0

    def test_already_converged(self) -> None:
        current = {"a": 0.60}
        realized = {"a": 0.60}
        new, dist = contraction_step(current, realized, k=0.5)
        assert dist < 1e-6

    def test_clamps_to_valid_range(self) -> None:
        current = {"a": 0.02}
        realized = {"a": -0.50}  # would push below 0
        new, _ = contraction_step(current, realized, k=0.9)
        assert new["a"] >= 0.01

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="contraction factor"):
            contraction_step({"a": 0.5}, {"a": 0.6}, k=1.5)

    def test_missing_realized_key_uses_old(self) -> None:
        current = {"a": 0.50, "b": 0.60}
        realized = {"a": 0.70}  # b is missing
        new, _ = contraction_step(current, realized, k=0.5)
        assert new["b"] == 0.60  # unchanged


class TestContractionLoop:
    def test_converges(self) -> None:
        # Simple case: realized always returns the same target
        target = {"x": 0.75, "y": 0.25}

        def generate(current: dict[str, float]) -> dict[str, float]:
            return target

        result = contraction_loop(generate, {"x": 0.50, "y": 0.50}, k=0.5, tol=1e-3)
        assert result.converged
        assert abs(result.values["x"] - 0.75) < 0.01
        assert abs(result.values["y"] - 0.25) < 0.01

    def test_max_iter_reached(self) -> None:
        # Generate function that always moves the target
        step = [0]

        def generate(current: dict[str, float]) -> dict[str, float]:
            step[0] += 1
            return {"x": current["x"] + 0.1}  # keeps changing

        result = contraction_loop(generate, {"x": 0.50}, k=0.5, tol=1e-6, max_iter=3)
        assert not result.converged
        assert result.iteration == 3


class TestBayesianSeed:
    def test_round_trip(self) -> None:
        priors = {"LONG": 0.55, "SHORT": 0.65}
        seed = priors_to_bayesian_seed(priors, equivalent_n=20)
        recovered = bayesian_seed_to_priors(seed)
        # Should be close (rounding differences)
        assert abs(recovered["LONG"] - 0.55) < 0.05
        assert abs(recovered["SHORT"] - 0.65) < 0.05

    def test_alpha_beta_positive(self) -> None:
        seed = priors_to_bayesian_seed({"x": 0.01}, equivalent_n=10)
        assert seed["x"]["alpha"] >= 1
        assert seed["x"]["beta"] >= 1

    def test_equivalent_n_effect(self) -> None:
        seed_low = priors_to_bayesian_seed({"x": 0.50}, equivalent_n=10)
        seed_high = priors_to_bayesian_seed({"x": 0.50}, equivalent_n=100)
        total_low = seed_low["x"]["alpha"] + seed_low["x"]["beta"]
        total_high = seed_high["x"]["alpha"] + seed_high["x"]["beta"]
        assert total_high > total_low
