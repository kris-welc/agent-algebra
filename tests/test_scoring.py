"""Tests for agent_algebra.scoring."""

from __future__ import annotations

import math
import pytest

from agent_algebra.scoring import (
    ScoringTracker,
    brier_score,
    brier_skill_score,
    log_pool_aggregate,
    log_score,
)
from agent_algebra.types import Outcome, Prediction


class TestBrierScore:
    def test_perfect_forecasts(self) -> None:
        assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == 0.0

    def test_worst_forecasts(self) -> None:
        assert brier_score([0.0, 1.0], [True, False]) == 1.0

    def test_moderate_forecasts(self) -> None:
        score = brier_score([0.7, 0.3], [True, False])
        assert 0.0 < score < 0.5

    def test_empty_returns_one(self) -> None:
        assert brier_score([], []) == 1.0

    def test_symmetric(self) -> None:
        s1 = brier_score([0.8], [True])
        s2 = brier_score([0.2], [False])
        assert abs(s1 - s2) < 1e-10


class TestBrierSkillScore:
    def test_perfect_skill(self) -> None:
        # Perfect forecaster vs base rate
        score = brier_skill_score([1.0, 0.0, 1.0, 0.0], [True, False, True, False])
        assert score == 1.0

    def test_no_skill(self) -> None:
        # Forecaster that always predicts base rate (0.5)
        score = brier_skill_score([0.5, 0.5, 0.5, 0.5], [True, False, True, False])
        assert abs(score) < 1e-10

    def test_empty_returns_zero(self) -> None:
        assert brier_skill_score([], []) == 0.0


class TestLogScore:
    def test_perfect_forecasts(self) -> None:
        # Near-perfect (can't use exactly 1.0 due to log)
        score = log_score([0.999, 0.001], [True, False])
        assert score > -0.01

    def test_terrible_forecasts(self) -> None:
        score = log_score([0.01, 0.99], [True, False])
        assert score < -2.0

    def test_empty_returns_neg_inf(self) -> None:
        assert log_score([], []) == float("-inf")

    def test_moderate(self) -> None:
        score = log_score([0.7, 0.3], [True, False])
        assert -1.0 < score < 0.0


class TestLogPoolAggregate:
    def test_uniform_weights(self) -> None:
        # With uniform weights, log pool amplifies away from 0.5
        result = log_pool_aggregate([0.6, 0.6], [1.0, 1.0])
        assert 0.5 < result < 1.0

    def test_single_agent(self) -> None:
        result = log_pool_aggregate([0.8], [1.0])
        assert abs(result - 0.8) < 0.01

    def test_empty_returns_half(self) -> None:
        assert log_pool_aggregate([], []) == 0.5

    def test_weighted_toward_better(self) -> None:
        # Agent 1 says 0.9 with high weight, Agent 2 says 0.1 with low weight
        result = log_pool_aggregate([0.9, 0.1], [10.0, 1.0])
        assert result > 0.5


class TestScoringTracker:
    def test_record_and_scores(self) -> None:
        tracker = ScoringTracker()
        tracker.record("a1", Prediction(0.8), Outcome(True))
        tracker.record("a1", Prediction(0.2), Outcome(False))
        rec = tracker.agent_scores("a1")
        assert rec is not None
        assert rec.brier < 0.1
        assert rec.n_predictions == 2

    def test_calibration_weights_normalized(self) -> None:
        tracker = ScoringTracker()
        for p, o in [(0.8, True), (0.2, False)]:
            tracker.record("good", Prediction(p), Outcome(o))
        for p, o in [(0.5, True), (0.5, False)]:
            tracker.record("ok", Prediction(p), Outcome(o))
        weights = tracker.calibration_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert weights["good"] > weights["ok"]

    def test_leaderboard_sorted(self) -> None:
        tracker = ScoringTracker()
        # Good agent
        for p, o in [(0.9, True), (0.1, False), (0.8, True)]:
            tracker.record("best", Prediction(p), Outcome(o))
        # Bad agent
        for p, o in [(0.5, True), (0.5, False), (0.5, True)]:
            tracker.record("worst", Prediction(p), Outcome(o))
        board = tracker.leaderboard()
        assert board[0].agent_id == "best"
        assert board[0].brier < board[1].brier

    def test_aggregate_log_pool(self) -> None:
        tracker = ScoringTracker()
        for p, o in [(0.8, True), (0.2, False)]:
            tracker.record("a", Prediction(p), Outcome(o))
        for p, o in [(0.6, True), (0.4, False)]:
            tracker.record("b", Prediction(p), Outcome(o))
        result = tracker.aggregate({"a": 0.7, "b": 0.6})
        assert 0.0 < result < 1.0

    def test_aggregate_linear(self) -> None:
        tracker = ScoringTracker()
        for p, o in [(0.8, True), (0.2, False)]:
            tracker.record("a", Prediction(p), Outcome(o))
        result = tracker.aggregate({"a": 0.7}, method="linear")
        assert 0.0 < result < 1.0

    def test_aggregate_no_history(self) -> None:
        tracker = ScoringTracker()
        result = tracker.aggregate({"a": 0.7, "b": 0.3})
        assert abs(result - 0.5) < 1e-10

    def test_unknown_agent_returns_none(self) -> None:
        tracker = ScoringTracker()
        assert tracker.agent_scores("nonexistent") is None
