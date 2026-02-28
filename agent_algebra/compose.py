"""Pipeline composition of algebra primitives.

Chain primitives into reusable stacks. Each step receives the output
of the previous step and returns a new value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class PipelineStep:
    """A named step in a pipeline."""

    name: str
    fn: Callable[[Any], Any]


@dataclass
class Pipeline:
    """Chain primitives into a processing pipeline.

    Each step is a callable that transforms data.
    Steps execute in order, each receiving the previous step's output.
    """

    steps: list[PipelineStep] = field(default_factory=list)

    def add(self, name: str, fn: Callable[[Any], Any]) -> Pipeline:
        """Add a step. Returns a new Pipeline (immutable pattern)."""
        return Pipeline(steps=[*self.steps, PipelineStep(name=name, fn=fn)])

    def run(self, data: Any) -> Any:
        """Execute all steps in order."""
        result = data
        for step in self.steps:
            result = step.fn(result)
        return result

    def run_traced(self, data: Any) -> list[tuple[str, Any]]:
        """Execute all steps, returning intermediate results."""
        trace: list[tuple[str, Any]] = [("input", data)]
        result = data
        for step in self.steps:
            result = step.fn(result)
            trace.append((step.name, result))
        return trace

    def __len__(self) -> int:
        return len(self.steps)


def trading_stack() -> Pipeline:
    """Pre-built pipeline for trading: belief → scoring → boost → ergodic → contraction.

    Returns an empty pipeline with the canonical step names.
    Users fill in their own functions for each step.

    Example:
        stack = trading_stack()
        stack = stack.add("belief", my_belief_fn)
        stack = stack.add("scoring", my_scoring_fn)
        ...
    """
    return Pipeline()


def intel_stack() -> Pipeline:
    """Pre-built pipeline for intel/forecasting: compress → boost → scoring → contraction.

    Returns an empty pipeline. Users fill in their own functions.
    """
    return Pipeline()
