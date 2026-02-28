"""Tests for agent_algebra.boost."""

from __future__ import annotations

import pytest

from agent_algebra.boost import (
    BoostedEnsemble,
    boost_cascade,
    compute_boost_weights,
    identify_hard_cases,
)


class TestIdentifyHardCases:
    def test_finds_errors(self) -> None:
        preds = [0.9, 0.1, 0.8, 0.2]
        outcomes = [True, False, False, True]  # 3rd and 4th are wrong
        hard = identify_hard_cases(preds, outcomes, threshold=0.6)
        assert 2 in hard  # pred=0.8, outcome=False → error=0.8
        assert 3 in hard  # pred=0.2, outcome=True → error=0.8

    def test_no_hard_cases(self) -> None:
        preds = [0.7, 0.3]
        outcomes = [True, False]
        hard = identify_hard_cases(preds, outcomes, threshold=0.5)
        assert len(hard) == 0

    def test_all_hard(self) -> None:
        preds = [0.0, 1.0]
        outcomes = [True, False]
        hard = identify_hard_cases(preds, outcomes, threshold=0.5)
        assert len(hard) == 2


class TestComputeBoostWeights:
    def test_upweights_hard_cases(self) -> None:
        preds = [0.8, 0.2, 0.1]
        outcomes = [True, False, True]
        hard = [2]  # only index 2 is hard
        weights = compute_boost_weights(preds, outcomes, hard, factor=3.0)
        assert weights[0] == 1.0
        assert weights[1] == 1.0
        assert weights[2] == 3.0

    def test_no_hard_cases_uniform(self) -> None:
        weights = compute_boost_weights([0.5, 0.5], [True, False], [], factor=2.0)
        assert all(w == 1.0 for w in weights)


class TestBoostCascade:
    def test_basic_cascade(self) -> None:
        # Three agents with different accuracies
        agent1 = lambda x: 0.6  # weak
        agent2 = lambda x: 0.7  # better
        agent3 = lambda x: 0.8  # best

        data = list(range(10))
        outcomes = [True] * 6 + [False] * 4

        ensemble = boost_cascade(
            [agent1, agent2, agent3], data, outcomes, rounds=3
        )
        assert len(ensemble.agents) == 3
        assert len(ensemble.weights) == 3
        assert len(ensemble.rounds) == 3

    def test_predict(self) -> None:
        agent1 = lambda x: 0.8
        agent2 = lambda x: 0.6
        data = [1, 2, 3]
        outcomes = [True, True, False]

        ensemble = boost_cascade([agent1, agent2], data, outcomes, rounds=2)
        pred = ensemble.predict("test_input")
        assert 0.0 <= pred <= 1.0

    def test_empty_agents(self) -> None:
        ensemble = boost_cascade([], [1, 2], [True, False])
        assert ensemble.predict("x") == 0.5

    def test_empty_data(self) -> None:
        ensemble = boost_cascade([lambda x: 0.5], [], [])
        assert len(ensemble.agents) == 0

    def test_rounds_limited_by_agents(self) -> None:
        ensemble = boost_cascade(
            [lambda x: 0.6], [1, 2], [True, False], rounds=5
        )
        assert len(ensemble.rounds) == 1  # only 1 agent available
