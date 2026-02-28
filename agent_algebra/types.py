"""Core types for agent-algebra.

An agent is any callable that returns a Prediction.
All data types are frozen dataclasses (immutable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Prediction:
    """A probability forecast from an agent."""

    probability: float
    confidence: float | None = None
    agent_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class Outcome:
    """The resolved result of a predicted event."""

    occurred: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class AgentRecord:
    """Calibration summary for a single agent."""

    agent_id: str
    brier: float
    log_score: float
    n_predictions: int
    weight: float


@dataclass(frozen=True)
class ScoredPrediction:
    """A prediction paired with its outcome and scores."""

    prediction: Prediction
    outcome: Outcome
    brier: float
    log_score: float
