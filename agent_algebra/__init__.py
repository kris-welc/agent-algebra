"""Agent Algebra — mathematical composition primitives for AI agent ensembles.

Six theorems, composable:
1. Contraction Mapping (self-calibrating loops)
2. Boosting Cascade (error-focusing)
3. Proper Scoring Rules (calibration tracking)
4. Ergodicity-Corrected Kelly (path-aware sizing)
5. Belief Propagation (message passing)
6. MDL Compression (signal/noise classification)
"""

from __future__ import annotations

from agent_algebra.belief import BeliefNode, build_graph, propagate
from agent_algebra.boost import boost_cascade
from agent_algebra.compose import Pipeline, intel_stack, trading_stack
from agent_algebra.compress import is_signal, mdl_filter
from agent_algebra.contraction import contraction_loop, contraction_step
from agent_algebra.ergodic import ergodic_kelly, kelly_fraction
from agent_algebra.scoring import ScoringTracker, brier_score, log_score
from agent_algebra.types import AgentRecord, Outcome, Prediction

__version__ = "0.1.0"

__all__ = [
    # Types
    "Prediction",
    "Outcome",
    "AgentRecord",
    # Scoring (Theorem 3)
    "ScoringTracker",
    "brier_score",
    "log_score",
    # Contraction (Theorem 1)
    "contraction_loop",
    "contraction_step",
    # Boosting (Theorem 2)
    "boost_cascade",
    # Ergodic (Theorem 4)
    "kelly_fraction",
    "ergodic_kelly",
    # Belief (Theorem 5)
    "propagate",
    "build_graph",
    "BeliefNode",
    # Compression (Theorem 6)
    "mdl_filter",
    "is_signal",
    # Composition
    "Pipeline",
    "trading_stack",
    "intel_stack",
]
