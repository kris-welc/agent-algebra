"""Belief Propagation (Theorem 5) — Pearl, 1988.

On tree-structured graphical models, local message passing converges
to globally optimal posteriors. Each node communicates only with
its neighbors. Loopy graphs use damping for approximate convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BeliefNode:
    """An agent node in the belief network.

    local_prob: this agent's local probability estimate
    neighbors: IDs of connected agents
    """

    node_id: str
    local_prob: float
    neighbors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.local_prob <= 1.0:
            raise ValueError(
                f"local_prob must be in [0, 1], got {self.local_prob}"
            )


@dataclass(frozen=True)
class PropagationResult:
    """Result of belief propagation."""

    beliefs: dict[str, float]
    iterations: int
    converged: bool
    max_change: float


def _clamp_prob(p: float) -> float:
    """Clamp a probability to (eps, 1-eps) to avoid log(0)."""
    eps = 1e-10
    return max(eps, min(1 - eps, p))


def message(
    sender_prob: float,
    receiver_prob: float,
    damping: float = 0.5,
) -> float:
    """Compute a probability update message from sender to receiver.

    Uses log-odds combination with damping.
    damping: 0 = ignore message, 1 = full update
    """
    s = _clamp_prob(sender_prob)
    r = _clamp_prob(receiver_prob)

    # Convert to log-odds
    sender_logodds = math.log(s / (1 - s))
    receiver_logodds = math.log(r / (1 - r))

    # Damped update in log-odds space
    updated_logodds = receiver_logodds + damping * (sender_logodds - receiver_logodds)

    # Convert back to probability
    return 1.0 / (1.0 + math.exp(-updated_logodds))


def build_graph(
    agents: list[BeliefNode],
    edges: list[tuple[str, str]],
) -> dict[str, BeliefNode]:
    """Build a belief graph from agents and edges.

    Edges are undirected: (A, B) means A and B are neighbors.
    Returns a dict of node_id → BeliefNode with neighbors populated.
    """
    # Collect neighbors for each node
    neighbor_map: dict[str, list[str]] = {a.node_id: [] for a in agents}
    for a_id, b_id in edges:
        if a_id in neighbor_map and b_id in neighbor_map:
            if b_id not in neighbor_map[a_id]:
                neighbor_map[a_id].append(b_id)
            if a_id not in neighbor_map[b_id]:
                neighbor_map[b_id].append(a_id)

    # Rebuild nodes with neighbor info
    graph: dict[str, BeliefNode] = {}
    for agent in agents:
        graph[agent.node_id] = BeliefNode(
            node_id=agent.node_id,
            local_prob=agent.local_prob,
            neighbors=tuple(sorted(neighbor_map.get(agent.node_id, []))),
        )
    return graph


def propagate(
    graph: dict[str, BeliefNode],
    max_iter: int = 50,
    tol: float = 1e-4,
    damping: float = 0.3,
) -> PropagationResult:
    """Run belief propagation until convergence.

    graph: {node_id: BeliefNode} with neighbors set
    max_iter: maximum iterations
    tol: convergence tolerance on max belief change
    damping: message damping (lower = more stable for loopy graphs)

    Returns PropagationResult with converged beliefs.
    """
    # Initialize beliefs with local probabilities
    beliefs: dict[str, float] = {
        nid: node.local_prob for nid, node in graph.items()
    }

    max_change = float("inf")

    for iteration in range(1, max_iter + 1):
        new_beliefs: dict[str, float] = {}
        max_change = 0.0

        for nid, node in graph.items():
            if not node.neighbors:
                new_beliefs[nid] = beliefs[nid]
                continue

            # Collect messages from all neighbors
            updated = beliefs[nid]
            for neighbor_id in node.neighbors:
                updated = message(beliefs[neighbor_id], updated, damping)

            # Anchor toward local evidence
            local_logodds = math.log(
                _clamp_prob(node.local_prob) / (1 - _clamp_prob(node.local_prob))
            )
            updated_logodds = math.log(
                _clamp_prob(updated) / (1 - _clamp_prob(updated))
            )
            anchored_logodds = 0.5 * local_logodds + 0.5 * updated_logodds
            anchored = 1.0 / (1.0 + math.exp(-anchored_logodds))

            new_beliefs[nid] = anchored
            max_change = max(max_change, abs(anchored - beliefs[nid]))

        beliefs = new_beliefs

        if max_change < tol:
            return PropagationResult(
                beliefs=beliefs,
                iterations=iteration,
                converged=True,
                max_change=max_change,
            )

    return PropagationResult(
        beliefs=beliefs,
        iterations=max_iter,
        converged=False,
        max_change=max_change,
    )
