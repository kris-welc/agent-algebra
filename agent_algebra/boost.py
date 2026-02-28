"""Boosting Cascade (Theorem 2) — Schapire 1990.

Any ensemble of weak learners (accuracy > 50%) can be combined into
an arbitrarily strong learner. Each successive learner focuses on
the errors of the previous ones. Error drops exponentially.

Generalized from moneytrees/agent_algebra.py boosting primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class BoostRound:
    """Results from one round of boosting."""

    round_num: int
    error_rate: float
    weight: float
    hard_indices: list[int]


@dataclass(frozen=True)
class BoostedEnsemble:
    """A boosted ensemble of agent callables."""

    agents: list[Callable[[object], float]]
    weights: list[float]
    rounds: list[BoostRound]

    def predict(self, inputs: object) -> float:
        """Weighted vote across all agents in the ensemble."""
        if not self.agents:
            return 0.5
        total = 0.0
        weight_sum = 0.0
        for agent, w in zip(self.agents, self.weights):
            total += w * agent(inputs)
            weight_sum += w
        if weight_sum == 0:
            return 0.5
        return total / weight_sum


def identify_hard_cases(
    predictions: list[float],
    outcomes: list[bool],
    threshold: float = 0.6,
) -> list[int]:
    """Find indices where predictions were wrong or poorly calibrated.

    A case is "hard" if the absolute error exceeds the threshold.
    Returns indices of hard cases for the next boosting round.
    """
    hard: list[int] = []
    for i, (pred, outcome) in enumerate(zip(predictions, outcomes)):
        error = abs(pred - float(outcome))
        if error >= threshold:
            hard.append(i)
    return hard


def compute_boost_weights(
    predictions: list[float],
    outcomes: list[bool],
    hard_indices: list[int],
    factor: float = 2.0,
) -> list[float]:
    """Create sample weights that upweight hard cases.

    Returns weight list with hard cases multiplied by factor.
    """
    weights = [1.0] * len(predictions)
    hard_set = set(hard_indices)
    for i in range(len(weights)):
        if i in hard_set:
            weights[i] = factor
    return weights


def boost_cascade(
    agents: list[Callable[[object], float]],
    data: list[object],
    outcomes: list[bool],
    rounds: int = 3,
    error_threshold: float = 0.6,
) -> BoostedEnsemble:
    """Run AdaBoost-style cascade over a list of agent callables.

    agents: callables that take an input and return a probability
    data: list of inputs to evaluate
    outcomes: ground truth for each input
    rounds: number of boosting rounds (uses agents in order)

    Each round:
    1. Current agent predicts on all data
    2. Compute weighted error rate
    3. Derive AdaBoost weight: ln((1-e)/e)
    4. Identify hard cases for next round

    Returns BoostedEnsemble with agents, weights, and round info.
    """
    if not agents or not data:
        return BoostedEnsemble(agents=[], weights=[], rounds=[])

    n = len(data)
    sample_weights = [1.0 / n] * n
    used_agents: list[Callable[[object], float]] = []
    agent_weights: list[float] = []
    round_info: list[BoostRound] = []

    num_rounds = min(rounds, len(agents))
    for r in range(num_rounds):
        agent = agents[r]

        # Get predictions from this agent
        preds = [agent(d) for d in data]

        # Compute weighted error
        weighted_error = 0.0
        for i, (pred, outcome) in enumerate(zip(preds, outcomes)):
            error = abs(pred - float(outcome))
            if error >= error_threshold:
                weighted_error += sample_weights[i]

        # Clamp error to avoid division by zero or log of negative
        weighted_error = max(1e-10, min(1.0 - 1e-10, weighted_error))

        # AdaBoost weight: ln((1-e)/e)
        agent_w = math.log((1 - weighted_error) / weighted_error)
        if agent_w < 0:
            agent_w = 0.0

        used_agents.append(agent)
        agent_weights.append(agent_w)

        # Find hard cases
        hard = identify_hard_cases(preds, outcomes, error_threshold)

        round_info.append(BoostRound(
            round_num=r + 1,
            error_rate=weighted_error,
            weight=agent_w,
            hard_indices=hard,
        ))

        # Update sample weights — upweight hard cases
        for i in range(n):
            error = abs(preds[i] - float(outcomes[i]))
            if error >= error_threshold:
                sample_weights[i] *= math.exp(agent_w)
            else:
                sample_weights[i] *= math.exp(-agent_w)

        # Normalize sample weights
        total = sum(sample_weights)
        if total > 0:
            sample_weights = [w / total for w in sample_weights]

    return BoostedEnsemble(
        agents=used_agents,
        weights=agent_weights,
        rounds=round_info,
    )
