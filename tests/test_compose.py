"""Tests for agent_algebra.compose."""

from __future__ import annotations

import pytest

from agent_algebra.compose import Pipeline, intel_stack, trading_stack


class TestPipeline:
    def test_single_step(self) -> None:
        p = Pipeline().add("double", lambda x: x * 2)
        assert p.run(5) == 10

    def test_chain(self) -> None:
        p = (
            Pipeline()
            .add("add1", lambda x: x + 1)
            .add("double", lambda x: x * 2)
            .add("sub3", lambda x: x - 3)
        )
        # (5 + 1) * 2 - 3 = 9
        assert p.run(5) == 9

    def test_immutable_add(self) -> None:
        p1 = Pipeline()
        p2 = p1.add("step", lambda x: x)
        assert len(p1) == 0
        assert len(p2) == 1

    def test_run_traced(self) -> None:
        p = (
            Pipeline()
            .add("add1", lambda x: x + 1)
            .add("double", lambda x: x * 2)
        )
        trace = p.run_traced(3)
        assert trace[0] == ("input", 3)
        assert trace[1] == ("add1", 4)
        assert trace[2] == ("double", 8)

    def test_empty_pipeline(self) -> None:
        p = Pipeline()
        assert p.run("passthrough") == "passthrough"
        assert len(p) == 0

    def test_len(self) -> None:
        p = Pipeline().add("a", lambda x: x).add("b", lambda x: x)
        assert len(p) == 2

    def test_with_dict_data(self) -> None:
        p = (
            Pipeline()
            .add("extract", lambda d: d["value"])
            .add("transform", lambda v: v ** 2)
        )
        assert p.run({"value": 4}) == 16


class TestPrebuiltStacks:
    def test_trading_stack_is_empty(self) -> None:
        stack = trading_stack()
        assert len(stack) == 0
        assert isinstance(stack, Pipeline)

    def test_intel_stack_is_empty(self) -> None:
        stack = intel_stack()
        assert len(stack) == 0
        assert isinstance(stack, Pipeline)
