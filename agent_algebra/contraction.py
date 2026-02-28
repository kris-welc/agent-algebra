"""Contraction Mapping (Theorem 1) — Banach Fixed-Point iteration.

If T satisfies d(T(x), T(y)) <= k * d(x, y) for k < 1,
repeated application converges to a unique fixed point.

Ported and generalized from moneytrees/agent_algebra.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ContractionResult:
    """Result of a contraction mapping iteration."""

    values: dict[str, float]
    distance: float
    iteration: int
    converged: bool


def contraction_step(
    current: dict[str, float],
    realized: dict[str, float],
    k: float = 0.5,
) -> tuple[dict[str, float], float]:
    """One step: blend current values toward realized values.

    current: {key: value} — current parameter estimates
    realized: {key: realized_value} — observed ground truth
    k: contraction factor (0 < k < 1). Lower = more conservative.

    Returns: (new_values, max_distance)
    """
    if not 0 < k < 1:
        raise ValueError(f"contraction factor k must be in (0, 1), got {k}")

    new_values: dict[str, float] = {}
    max_dist = 0.0
    for key, old in current.items():
        target = realized.get(key, old)
        new = old + k * (target - old)
        new = max(0.01, min(0.99, new))
        new_values[key] = round(new, 6)
        max_dist = max(max_dist, abs(new - old))
    return new_values, max_dist


def contraction_loop(
    generate: Callable[[dict[str, float]], dict[str, float]],
    initial: dict[str, float],
    k: float = 0.5,
    tol: float = 1e-3,
    max_iter: int = 20,
) -> ContractionResult:
    """Run contraction mapping to convergence.

    generate: function that takes current params and returns realized values
              (e.g., run backtest → compute win rates)
    initial: starting parameter estimates
    k: contraction factor
    tol: convergence tolerance (max distance between iterations)
    max_iter: safety bound on iterations

    Returns ContractionResult with final values and convergence info.
    """
    current = dict(initial)
    for i in range(1, max_iter + 1):
        realized = generate(current)
        new_values, distance = contraction_step(current, realized, k)
        if distance < tol:
            return ContractionResult(
                values=new_values,
                distance=distance,
                iteration=i,
                converged=True,
            )
        current = new_values

    return ContractionResult(
        values=current,
        distance=distance,
        iteration=max_iter,
        converged=False,
    )


def priors_to_bayesian_seed(
    priors: dict[str, float], equivalent_n: int = 20
) -> dict[str, dict[str, int]]:
    """Convert win rate priors to Beta distribution parameters.

    priors: {signal_type: win_rate}
    equivalent_n: pseudo-observations the prior is worth.
        Higher = more confident, slower to adapt.

    Returns: {signal_type: {"alpha": int, "beta": int}}
    """
    seed: dict[str, dict[str, int]] = {}
    for key, wr in priors.items():
        alpha = max(1, round(wr * equivalent_n))
        beta = max(1, round((1 - wr) * equivalent_n))
        seed[key] = {"alpha": alpha, "beta": beta}
    return seed


def bayesian_seed_to_priors(
    seed: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Extract win rate priors from Beta parameters."""
    return {
        key: round(ab["alpha"] / (ab["alpha"] + ab["beta"]), 4)
        for key, ab in seed.items()
    }
