"""Example: Intel/forecasting stack using Agent Algebra.

Demonstrates MDL compression → boosting → proper scoring →
contraction mapping for content quality assessment.
"""

from __future__ import annotations

from agent_algebra import (
    Outcome,
    Pipeline,
    Prediction,
    ScoringTracker,
    contraction_step,
    is_signal,
    mdl_filter,
)


def simple_summarize(text: str) -> str:
    """Naive summarizer: keep first sentence."""
    parts = text.split(". ")
    return parts[0] if parts else text


def simple_reconstruct(summary: str) -> str:
    """Naive reconstruction: return summary as-is."""
    return summary


def main() -> None:
    # --- Layer 1: MDL Compression (noise filter) ---
    print("=== Layer 1: MDL Filter ===")
    articles = [
        "Bitcoin surges 15% on ETF approval news. Markets rally. Analysts predict continued growth.",
        "asdfjkl qwerty zxcvb random noise text that has no structure whatsoever gibberish",
        "Fed holds rates steady. Bond yields drop. Tech stocks respond positively to the news.",
        "x y z a b c d e f g h i j k l m n o p q r s t u v w",
    ]

    for article in articles:
        summary = simple_summarize(article)
        signal = is_signal(
            article, summary, simple_reconstruct,
            compression_threshold=0.7,
            reconstruction_threshold=0.2,
        )
        print(f"  {'SIGNAL' if signal else 'NOISE '}: {article[:60]}...")

    filtered = mdl_filter(
        articles, simple_summarize, simple_reconstruct,
        compression_threshold=0.7,
        reconstruction_threshold=0.2,
    )
    print(f"  Kept {len(filtered)} of {len(articles)} articles")

    # --- Layer 2: Source Credibility Scoring ---
    print("\n=== Layer 2: Source Scoring ===")
    tracker = ScoringTracker()

    # Reuters: well-calibrated source
    for p, o in [(0.80, True), (0.70, True), (0.30, False), (0.60, True), (0.25, False)]:
        tracker.record("reuters", Prediction(p), Outcome(o))

    # Blog: poorly calibrated
    for p, o in [(0.90, False), (0.85, False), (0.95, True), (0.80, False), (0.70, True)]:
        tracker.record("blog", Prediction(p), Outcome(o))

    board = tracker.leaderboard()
    for rec in board:
        print(f"  {rec.agent_id}: Brier={rec.brier:.4f}, weight={rec.weight:.2f}")

    # --- Layer 3: Threshold Calibration ---
    print("\n=== Layer 3: Threshold Calibration ===")
    current = {"min_score": 0.50, "source_trust": 0.70}
    realized = {"min_score": 0.60, "source_trust": 0.55}
    new_params, dist = contraction_step(current, realized, k=0.4)
    print(f"  Distance: {dist:.4f}")
    for k, v in new_params.items():
        print(f"  {k}: {current[k]:.2f} → {v:.4f}")

    # --- Full Pipeline ---
    print("\n=== Full Pipeline ===")
    pipeline = (
        Pipeline()
        .add("filter", lambda items: mdl_filter(
            items, simple_summarize, simple_reconstruct,
            compression_threshold=0.7, reconstruction_threshold=0.2,
        ))
        .add("score", lambda items: {
            f"item_{i}": tracker.aggregate({
                "reuters": 0.7, "blog": 0.5,
            })
            for i, _ in enumerate(items)
        })
    )
    result = pipeline.run(articles)
    print(f"  Pipeline output: {result}")
    print("\nDone.")


if __name__ == "__main__":
    main()
