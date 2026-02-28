"""Tests for agent_algebra.types."""

from __future__ import annotations

import pytest
from datetime import datetime

from agent_algebra.types import AgentRecord, Outcome, Prediction, ScoredPrediction


class TestPrediction:
    def test_valid_prediction(self) -> None:
        p = Prediction(probability=0.7, agent_id="test")
        assert p.probability == 0.7
        assert p.agent_id == "test"
        assert p.confidence is None
        assert p.metadata == {}

    def test_with_confidence(self) -> None:
        p = Prediction(probability=0.5, confidence=0.8)
        assert p.confidence == 0.8

    def test_with_metadata(self) -> None:
        p = Prediction(probability=0.5, metadata={"source": "elo"})
        assert p.metadata["source"] == "elo"

    def test_probability_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            Prediction(probability=-0.1)

    def test_probability_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            Prediction(probability=1.1)

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Prediction(probability=0.5, confidence=1.5)

    def test_boundary_values(self) -> None:
        p0 = Prediction(probability=0.0)
        p1 = Prediction(probability=1.0)
        assert p0.probability == 0.0
        assert p1.probability == 1.0

    def test_frozen(self) -> None:
        p = Prediction(probability=0.5)
        with pytest.raises(AttributeError):
            p.probability = 0.6  # type: ignore[misc]


class TestOutcome:
    def test_outcome_true(self) -> None:
        o = Outcome(occurred=True)
        assert o.occurred is True
        assert isinstance(o.timestamp, datetime)

    def test_outcome_false(self) -> None:
        o = Outcome(occurred=False)
        assert o.occurred is False

    def test_frozen(self) -> None:
        o = Outcome(occurred=True)
        with pytest.raises(AttributeError):
            o.occurred = False  # type: ignore[misc]


class TestAgentRecord:
    def test_creation(self) -> None:
        r = AgentRecord(
            agent_id="agent1",
            brier=0.15,
            log_score=-0.3,
            n_predictions=100,
            weight=6.67,
        )
        assert r.agent_id == "agent1"
        assert r.brier == 0.15
        assert r.n_predictions == 100


class TestScoredPrediction:
    def test_creation(self) -> None:
        p = Prediction(probability=0.7)
        o = Outcome(occurred=True)
        sp = ScoredPrediction(prediction=p, outcome=o, brier=0.09, log_score=-0.15)
        assert sp.brier == 0.09
