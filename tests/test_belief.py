"""Tests for agent_algebra.belief."""

from __future__ import annotations

import pytest

from agent_algebra.belief import (
    BeliefNode,
    PropagationResult,
    build_graph,
    message,
    propagate,
)


class TestBeliefNode:
    def test_creation(self) -> None:
        node = BeliefNode("regime", 0.7)
        assert node.node_id == "regime"
        assert node.local_prob == 0.7
        assert node.neighbors == ()

    def test_invalid_prob_raises(self) -> None:
        with pytest.raises(ValueError, match="local_prob"):
            BeliefNode("bad", -0.1)
        with pytest.raises(ValueError, match="local_prob"):
            BeliefNode("bad", 1.5)

    def test_frozen(self) -> None:
        node = BeliefNode("x", 0.5)
        with pytest.raises(AttributeError):
            node.local_prob = 0.6  # type: ignore[misc]


class TestMessage:
    def test_same_probs_no_change(self) -> None:
        result = message(0.6, 0.6, damping=0.5)
        assert abs(result - 0.6) < 0.01

    def test_sender_influence(self) -> None:
        # Sender at 0.9, receiver at 0.5 → should move toward 0.9
        result = message(0.9, 0.5, damping=0.5)
        assert result > 0.5

    def test_no_damping_means_no_update(self) -> None:
        result = message(0.9, 0.5, damping=0.0)
        assert abs(result - 0.5) < 0.01

    def test_full_damping_moves_to_sender(self) -> None:
        result = message(0.8, 0.5, damping=1.0)
        assert abs(result - 0.8) < 0.01


class TestBuildGraph:
    def test_builds_neighbors(self) -> None:
        agents = [
            BeliefNode("a", 0.5),
            BeliefNode("b", 0.6),
            BeliefNode("c", 0.7),
        ]
        edges = [("a", "b"), ("b", "c")]
        graph = build_graph(agents, edges)
        assert "b" in graph["a"].neighbors
        assert "a" in graph["b"].neighbors
        assert "c" in graph["b"].neighbors
        assert graph["c"].neighbors == ("b",)

    def test_no_edges(self) -> None:
        agents = [BeliefNode("a", 0.5), BeliefNode("b", 0.6)]
        graph = build_graph(agents, [])
        assert graph["a"].neighbors == ()
        assert graph["b"].neighbors == ()

    def test_ignores_unknown_edges(self) -> None:
        agents = [BeliefNode("a", 0.5)]
        graph = build_graph(agents, [("a", "z")])  # z doesn't exist
        assert graph["a"].neighbors == ()


class TestPropagate:
    def test_converges_on_chain(self) -> None:
        agents = [
            BeliefNode("a", 0.3),
            BeliefNode("b", 0.5),
            BeliefNode("c", 0.7),
        ]
        edges = [("a", "b"), ("b", "c")]
        graph = build_graph(agents, edges)
        result = propagate(graph, damping=0.3)
        assert result.converged
        # All beliefs should be between the extremes
        for belief in result.beliefs.values():
            assert 0.2 < belief < 0.8

    def test_isolated_nodes_keep_local(self) -> None:
        agents = [BeliefNode("a", 0.3), BeliefNode("b", 0.7)]
        graph = build_graph(agents, [])  # no edges
        result = propagate(graph)
        assert abs(result.beliefs["a"] - 0.3) < 0.01
        assert abs(result.beliefs["b"] - 0.7) < 0.01

    def test_two_agreeing_nodes(self) -> None:
        agents = [BeliefNode("a", 0.8), BeliefNode("b", 0.8)]
        graph = build_graph(agents, [("a", "b")])
        result = propagate(graph, damping=0.3)
        assert result.converged
        assert abs(result.beliefs["a"] - 0.8) < 0.05
        assert abs(result.beliefs["b"] - 0.8) < 0.05

    def test_star_topology(self) -> None:
        agents = [
            BeliefNode("center", 0.5),
            BeliefNode("leaf1", 0.9),
            BeliefNode("leaf2", 0.1),
            BeliefNode("leaf3", 0.7),
        ]
        edges = [("center", "leaf1"), ("center", "leaf2"), ("center", "leaf3")]
        graph = build_graph(agents, edges)
        result = propagate(graph, damping=0.2)
        assert result.converged
