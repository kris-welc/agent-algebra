"""Example: Full trading stack using Agent Algebra (Layers 1-5).

Demonstrates belief propagation → proper scoring → boosting →
ergodic Kelly → contraction mapping for a trading ensemble.
"""

from __future__ import annotations

from datetime import datetime

from agent_algebra import (
    BeliefNode,
    Outcome,
    Pipeline,
    Prediction,
    ScoringTracker,
    build_graph,
    contraction_step,
    ergodic_kelly,
    propagate,
)


def main() -> None:
    # --- Layer 1: Belief Propagation (signal fusion) ---
    print("=== Layer 1: Belief Propagation ===")
    agents = [
        BeliefNode("regime", local_prob=0.60),
        BeliefNode("micro", local_prob=0.75),
        BeliefNode("sentiment", local_prob=0.55),
        BeliefNode("macro", local_prob=0.40),
    ]
    edges = [
        ("regime", "micro"),
        ("regime", "sentiment"),
        ("micro", "sentiment"),
        ("sentiment", "macro"),
    ]
    graph = build_graph(agents, edges)
    result = propagate(graph, damping=0.3)
    print(f"  Converged: {result.converged} in {result.iterations} iterations")
    for nid, belief in sorted(result.beliefs.items()):
        print(f"  {nid}: {belief:.4f}")

    # --- Layer 2: Proper Scoring Rules (calibration weighting) ---
    print("\n=== Layer 2: Proper Scoring ===")
    tracker = ScoringTracker()

    # Simulate some history — regime agent is well-calibrated
    for p, o in [(0.60, True), (0.55, True), (0.70, True), (0.40, False), (0.35, False)]:
        tracker.record("regime", Prediction(p), Outcome(o))

    # Micro agent is excellent
    for p, o in [(0.80, True), (0.75, True), (0.85, True), (0.20, False), (0.15, False)]:
        tracker.record("micro", Prediction(p), Outcome(o))

    # Sentiment agent is poorly calibrated
    for p, o in [(0.90, False), (0.80, False), (0.70, True), (0.60, True), (0.50, False)]:
        tracker.record("sentiment", Prediction(p), Outcome(o))

    board = tracker.leaderboard()
    for rec in board:
        print(f"  {rec.agent_id}: Brier={rec.brier:.4f}, weight={rec.weight:.2f}")

    # Aggregate using belief-propagation outputs
    combined = tracker.aggregate(result.beliefs)
    print(f"  Combined probability: {combined:.4f}")

    # --- Layer 4: Ergodic Kelly Sizing ---
    print("\n=== Layer 4: Ergodic Kelly ===")
    # Simulated return series with some negative serial correlation
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.04, -0.01, 0.02, -0.02,
               0.01, 0.03, -0.04, 0.02, -0.01, 0.03, -0.02, 0.01, -0.01, 0.02]
    ek = ergodic_kelly(win_rate=0.60, win_loss_ratio=1.5, returns=returns, seed=42)
    print(f"  Kelly fraction: {ek.kelly_fraction:.4f}")
    print(f"  Ergodic fraction: {ek.ergodic_fraction:.4f}")
    print(f"  Correction: {ek.correction_factor:.4f}")

    # --- Layer 5: Contraction Mapping (meta-calibration) ---
    print("\n=== Layer 5: Contraction Mapping ===")
    current_params = {"regime_weight": 0.50, "micro_weight": 0.50, "threshold": 0.60}
    realized_params = {"regime_weight": 0.35, "micro_weight": 0.65, "threshold": 0.55}
    new_params, dist = contraction_step(current_params, realized_params, k=0.5)
    print(f"  Distance: {dist:.4f}")
    for k, v in new_params.items():
        print(f"  {k}: {v:.4f}")

    # --- Full Pipeline ---
    print("\n=== Full Pipeline ===")
    pipeline = (
        Pipeline()
        .add("belief", lambda d: propagate(build_graph(agents, edges)).beliefs)
        .add("aggregate", lambda beliefs: tracker.aggregate(beliefs))
        .add("size", lambda prob: prob * ek.ergodic_fraction)
    )
    trace = pipeline.run_traced(None)
    for name, val in trace:
        print(f"  [{name}] {val}")

    print("\nDone.")


if __name__ == "__main__":
    main()
