"""Tests for agent_algebra.store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_algebra.store import Store
from agent_algebra.types import Outcome, Prediction


class TestStore:
    def test_in_memory(self) -> None:
        store = Store(":memory:")
        pred = Prediction(probability=0.7)
        pid = store.save_prediction("agent1", pred)
        assert pid >= 1
        oid = store.save_outcome(pid, Outcome(occurred=True))
        assert oid >= 1
        store.close()

    def test_save_and_retrieve(self) -> None:
        with Store() as store:
            p1 = store.save_prediction("a", Prediction(0.8))
            store.save_outcome(p1, Outcome(True))
            p2 = store.save_prediction("a", Prediction(0.3))
            store.save_outcome(p2, Outcome(False))

            history = store.get_history("a")
            assert len(history) == 2
            # Most recent first
            assert history[0] == (0.3, False)
            assert history[1] == (0.8, True)

    def test_get_all_agents(self) -> None:
        with Store() as store:
            p1 = store.save_prediction("alpha", Prediction(0.5))
            store.save_outcome(p1, Outcome(True))
            p2 = store.save_prediction("beta", Prediction(0.6))
            store.save_outcome(p2, Outcome(False))

            agents = store.get_all_agents()
            assert "alpha" in agents
            assert "beta" in agents

    def test_agent_without_outcome_not_listed(self) -> None:
        with Store() as store:
            store.save_prediction("orphan", Prediction(0.5))
            agents = store.get_all_agents()
            assert "orphan" not in agents

    def test_history_limit(self) -> None:
        with Store() as store:
            for i in range(10):
                pid = store.save_prediction("x", Prediction(i / 10.0))
                store.save_outcome(pid, Outcome(i % 2 == 0))
            history = store.get_history("x", limit=3)
            assert len(history) == 3

    def test_file_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with Store(db_path) as store:
                pid = store.save_prediction("a", Prediction(0.7))
                store.save_outcome(pid, Outcome(True))
            # Reopen
            with Store(db_path) as store:
                history = store.get_history("a")
                assert len(history) == 1
                assert history[0] == (0.7, True)

    def test_context_manager(self) -> None:
        with Store() as store:
            pid = store.save_prediction("ctx", Prediction(0.5))
            store.save_outcome(pid, Outcome(True))
            assert len(store.get_history("ctx")) == 1
