"""Proper Scoring Rules (Theorem 3) — calibration tracking and weighted aggregation.

Ported and expanded from moneytrees/agent_algebra.py.
Under a proper scoring rule, an agent's expected payoff is maximized
by reporting its TRUE belief. Honesty is the dominant strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from agent_algebra.store import Store
from agent_algebra.types import AgentRecord, Outcome, Prediction


def brier_score(forecasts: list[float], outcomes: list[bool]) -> float:
    """Brier score: mean squared error of probability forecasts.

    Lower is better. 0 = perfect, 1 = worst possible.
    """
    if not forecasts:
        return 1.0
    return sum(
        (f - float(o)) ** 2 for f, o in zip(forecasts, outcomes)
    ) / len(forecasts)


def brier_skill_score(forecasts: list[float], outcomes: list[bool]) -> float:
    """Skill score relative to climatological (base rate) forecast.

    > 0 means better than always-predicting-the-base-rate.
    1.0 = perfect, 0.0 = no skill, < 0 = worse than base rate.
    """
    if not outcomes:
        return 0.0
    base_rate = sum(float(o) for o in outcomes) / len(outcomes)
    bs_model = brier_score(forecasts, outcomes)
    bs_ref = brier_score([base_rate] * len(outcomes), outcomes)
    if bs_ref == 0:
        return 0.0
    return 1.0 - bs_model / bs_ref


def log_score(forecasts: list[float], outcomes: list[bool]) -> float:
    """Logarithmic scoring rule. More negative is worse. 0 = perfect.

    Clamps probabilities to [1e-15, 1-1e-15] to avoid log(0).
    """
    if not forecasts:
        return float("-inf")
    eps = 1e-15
    total = 0.0
    for f, o in zip(forecasts, outcomes):
        f_clamped = max(eps, min(1 - eps, f))
        if o:
            total += math.log(f_clamped)
        else:
            total += math.log(1 - f_clamped)
    return total / len(forecasts)


def log_pool_aggregate(
    agent_probs: list[float], agent_weights: list[float]
) -> float:
    """Logarithmic opinion pool — proper scoring rule preserving.

    Combines multiple probability estimates using calibration-derived weights.
    """
    if not agent_probs:
        return 0.5
    eps = 1e-15
    log_sum = sum(
        w * math.log(max(eps, p)) for p, w in zip(agent_probs, agent_weights)
    )
    log_sum_c = sum(
        w * math.log(max(eps, 1 - p)) for p, w in zip(agent_probs, agent_weights)
    )
    raw = math.exp(log_sum)
    raw_c = math.exp(log_sum_c)
    denom = raw + raw_c
    if denom == 0:
        return 0.5
    return raw / denom


@dataclass
class ScoringTracker:
    """Persistent calibration tracker for multiple agents.

    Records predictions and outcomes, computes calibration-weighted
    aggregations. Backed by SQLite store for persistence.
    """

    store: Store = field(default_factory=lambda: Store(":memory:"))

    def record(
        self,
        agent_id: str,
        prediction: Prediction,
        outcome: Outcome,
    ) -> None:
        """Log a prediction and its outcome."""
        pred_id = self.store.save_prediction(agent_id, prediction)
        self.store.save_outcome(pred_id, outcome)

    def agent_scores(self, agent_id: str) -> AgentRecord | None:
        """Compute calibration scores for a single agent."""
        history = self.store.get_history(agent_id, limit=10000)
        if not history:
            return None
        forecasts = [h[0] for h in history]
        outcomes = [h[1] for h in history]
        bs = brier_score(forecasts, outcomes)
        ls = log_score(forecasts, outcomes)
        weight = 1.0 / (bs + 0.01)
        return AgentRecord(
            agent_id=agent_id,
            brier=bs,
            log_score=ls,
            n_predictions=len(history),
            weight=weight,
        )

    def calibration_weights(self) -> dict[str, float]:
        """Return Brier-weighted calibration weights for all agents."""
        agents = self.store.get_all_agents()
        weights: dict[str, float] = {}
        for aid in agents:
            rec = self.agent_scores(aid)
            if rec is not None:
                weights[aid] = rec.weight
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def aggregate(
        self,
        predictions: dict[str, float],
        method: str = "log_pool",
    ) -> float:
        """Calibration-weighted combination of agent predictions.

        predictions: {agent_id: probability}
        method: "log_pool" (default) or "linear"
        """
        weights = self.calibration_weights()
        if not weights:
            # No history — equal weighting
            vals = list(predictions.values())
            return sum(vals) / len(vals) if vals else 0.5

        probs: list[float] = []
        ws: list[float] = []
        for aid, prob in predictions.items():
            w = weights.get(aid, 0.0)
            if w > 0:
                probs.append(prob)
                ws.append(w)

        if not probs:
            vals = list(predictions.values())
            return sum(vals) / len(vals) if vals else 0.5

        if method == "log_pool":
            return log_pool_aggregate(probs, ws)

        # Linear pool
        total_w = sum(ws)
        return sum(p * w for p, w in zip(probs, ws)) / total_w

    def leaderboard(self) -> list[AgentRecord]:
        """Return all agents ranked by calibration (lowest Brier first)."""
        agents = self.store.get_all_agents()
        records = []
        for aid in agents:
            rec = self.agent_scores(aid)
            if rec is not None:
                records.append(rec)
        return sorted(records, key=lambda r: r.brier)
