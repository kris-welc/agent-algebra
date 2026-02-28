"""Ergodicity Economics (Theorem 4) — Ole Peters, 2019.

For multiplicative processes, the time average != the ensemble average.
Standard Kelly maximizes E[log(wealth)] (time average) for i.i.d. bets,
but real trading has serial correlation, regime shifts, and path-dependent
drawdowns. This module corrects for that.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ErgodicResult:
    """Result of ergodicity-corrected Kelly calculation."""

    kelly_fraction: float
    ergodic_fraction: float
    correction_factor: float
    median_growth: float
    mean_growth: float


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Standard Kelly criterion fraction.

    f* = win_rate - (1 - win_rate) / win_loss_ratio

    Returns the optimal fraction of bankroll to bet.
    Negative means don't bet.
    """
    if win_loss_ratio <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, f)


def geometric_growth_rate(returns: list[float]) -> float:
    """Compute the geometric (time-average) growth rate.

    Returns the annualized log-growth rate.
    Equivalent to exp(mean(log(1 + r))) - 1.
    """
    if not returns:
        return 0.0
    log_returns = []
    for r in returns:
        val = 1.0 + r
        if val <= 0:
            return float("-inf")
        log_returns.append(math.log(val))
    return math.exp(statistics.mean(log_returns)) - 1


def simulate_paths(
    returns: list[float],
    n_paths: int = 1000,
    n_steps: int = 252,
    seed: int | None = None,
) -> list[list[float]]:
    """Monte Carlo simulation preserving serial correlation via block bootstrap.

    returns: historical return series
    n_paths: number of simulated paths
    n_steps: steps per path
    seed: random seed for reproducibility

    Returns list of paths, each a list of cumulative wealth values.
    """
    if not returns:
        return [[1.0] * n_steps for _ in range(n_paths)]

    rng = random.Random(seed)
    block_size = max(1, len(returns) // 10)
    paths: list[list[float]] = []

    for _ in range(n_paths):
        wealth = 1.0
        path = [wealth]
        for _ in range(n_steps):
            # Block bootstrap: pick a random starting point, take block_size returns
            start = rng.randint(0, max(0, len(returns) - block_size))
            idx = start + (len(path) % block_size)
            if idx >= len(returns):
                idx = rng.randint(0, len(returns) - 1)
            r = returns[idx]
            wealth *= (1.0 + r)
            wealth = max(wealth, 1e-10)  # prevent zero/negative
            path.append(wealth)
        paths.append(path)

    return paths


def ergodic_correction(
    returns: list[float],
    n_paths: int = 1000,
    n_steps: int = 252,
    seed: int | None = None,
) -> float:
    """Compute the ergodicity correction factor: median/mean terminal wealth.

    A correction < 1 means the ensemble average overstates what
    a single-path agent actually experiences.
    """
    if not returns:
        return 1.0

    paths = simulate_paths(returns, n_paths, n_steps, seed)
    terminals = [p[-1] for p in paths]

    mean_terminal = statistics.mean(terminals)
    median_terminal = statistics.median(terminals)

    if mean_terminal <= 0:
        return 1.0
    return median_terminal / mean_terminal


def ergodic_kelly(
    win_rate: float,
    win_loss_ratio: float,
    returns: list[float],
    n_paths: int = 1000,
    seed: int | None = None,
) -> ErgodicResult:
    """Ergodicity-corrected Kelly fraction.

    f_ergodic = f_kelly * (median_growth / mean_growth)

    Accounts for path-dependent drawdowns and serial correlation
    that standard Kelly ignores.
    """
    f_kelly = kelly_fraction(win_rate, win_loss_ratio)

    if not returns or f_kelly == 0:
        return ErgodicResult(
            kelly_fraction=f_kelly,
            ergodic_fraction=f_kelly,
            correction_factor=1.0,
            median_growth=0.0,
            mean_growth=0.0,
        )

    paths = simulate_paths(returns, n_paths, seed=seed)
    terminals = [p[-1] for p in paths]
    mean_terminal = statistics.mean(terminals)
    median_terminal = statistics.median(terminals)

    correction = median_terminal / mean_terminal if mean_terminal > 0 else 1.0
    correction = max(0.1, min(1.0, correction))

    mean_growth = mean_terminal - 1.0
    median_growth = median_terminal - 1.0

    return ErgodicResult(
        kelly_fraction=f_kelly,
        ergodic_fraction=f_kelly * correction,
        correction_factor=correction,
        median_growth=median_growth,
        mean_growth=mean_growth,
    )
