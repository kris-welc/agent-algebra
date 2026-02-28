"""SQLite persistence for calibration history.

WAL mode for concurrent reads. Immutable pattern: never update rows,
only append new predictions and outcomes.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from agent_algebra.types import AgentRecord, Outcome, Prediction


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    probability REAL NOT NULL,
    confidence REAL,
    timestamp TEXT NOT NULL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    occurred INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_agent ON predictions(agent_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON outcomes(prediction_id);
"""


class Store:
    """SQLite-backed persistence for prediction/outcome history."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_prediction(
        self,
        agent_id: str,
        prediction: Prediction,
        timestamp: datetime | None = None,
    ) -> int:
        """Save a prediction. Returns the prediction row ID."""
        ts = (timestamp or datetime.now()).isoformat()
        meta = str(prediction.metadata) if prediction.metadata else None
        cursor = self._conn.execute(
            "INSERT INTO predictions (agent_id, probability, confidence, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (agent_id, prediction.probability, prediction.confidence, ts, meta),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def save_outcome(self, prediction_id: int, outcome: Outcome) -> int:
        """Save an outcome linked to a prediction. Returns the outcome row ID."""
        ts = outcome.timestamp.isoformat()
        cursor = self._conn.execute(
            "INSERT INTO outcomes (prediction_id, occurred, timestamp) VALUES (?, ?, ?)",
            (prediction_id, int(outcome.occurred), ts),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_history(
        self, agent_id: str, limit: int = 100
    ) -> list[tuple[float, bool]]:
        """Return (probability, occurred) pairs for an agent, most recent first."""
        rows = self._conn.execute(
            "SELECT p.probability, o.occurred "
            "FROM predictions p "
            "JOIN outcomes o ON o.prediction_id = p.id "
            "WHERE p.agent_id = ? "
            "ORDER BY p.id DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [(prob, bool(occ)) for prob, occ in rows]

    def get_all_agents(self) -> list[str]:
        """Return all agent IDs that have at least one scored prediction."""
        rows = self._conn.execute(
            "SELECT DISTINCT p.agent_id "
            "FROM predictions p "
            "JOIN outcomes o ON o.prediction_id = p.id "
            "ORDER BY p.agent_id",
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
