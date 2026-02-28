# Agent Algebra

Mathematical composition primitives for AI agent ensembles.

Six theorems. Six primitives. Zero dependencies. Provable guarantees.

## Why

LLM orchestration frameworks give you tools to connect agents — chains, tools, retry loops. Agent Algebra gives you **theorems** that tell you what to expect from those connections.

A retry loop re-runs an agent hoping for a better answer. A **contraction mapping** guarantees convergence to a unique fixed point. The difference is a single inequality: `d(T(x), T(y)) <= k * d(x, y)`, where `k < 1`.

> Self-improving systems need proofs, not hopes.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Zero external dependencies (stdlib only: `math`, `sqlite3`, `zlib`, `statistics`, `random`).

## The Six Primitives

| # | Theorem | Module | Guarantee |
|---|---------|--------|-----------|
| 1 | Banach Fixed-Point (1922) | `contraction` | Convergence in O(log(1/e)) iterations |
| 2 | Schapire AdaBoost (1990) | `boost` | Exponential error reduction |
| 3 | de Finetti Scoring | `scoring` | Incentive-compatible aggregation |
| 4 | Ole Peters Ergodicity (2019) | `ergodic` | Survival under path-dependent drawdowns |
| 5 | Pearl Belief Propagation (1988) | `belief` | Globally optimal posteriors from local messages |
| 6 | Kolmogorov/Rissanen MDL | `compress` | Principled signal/noise separation |

### 1. Contraction Mapping

Self-calibrating parameter loops. Feed backtest results back as realized values, converge to true win rates in 3-5 iterations.

```python
from agent_algebra import contraction_step, contraction_loop

# One step: blend current toward realized (k < 1 guarantees convergence)
new_params, distance = contraction_step(
    current={"win_rate": 0.50, "threshold": 0.60},
    realized={"win_rate": 0.65, "threshold": 0.55},
    k=0.5,
)

# Full loop: run until convergence
result = contraction_loop(
    generate=run_backtest,  # your function: params -> realized values
    initial={"win_rate": 0.50},
    k=0.5, tol=1e-3,
)
# result.converged = True, result.iteration = 4
```

### 2. Boosting Cascade

Weak learners (>50% accuracy) combine into arbitrarily strong ensembles. Each successive agent focuses on the errors of previous ones.

```python
from agent_algebra import boost_cascade

ensemble = boost_cascade(
    agents=[agent_a, agent_b, agent_c],
    data=test_inputs,
    outcomes=ground_truth,
    rounds=3,
)
prediction = ensemble.predict(new_input)
```

### 3. Proper Scoring Rules

Under proper scoring, honesty is the dominant strategy. Agents with better calibration get higher weights automatically.

```python
from agent_algebra import ScoringTracker, Prediction, Outcome

tracker = ScoringTracker()
tracker.record("agent_a", Prediction(0.80), Outcome(True))
tracker.record("agent_b", Prediction(0.90), Outcome(False))

# Calibration-weighted combination via log-pool
combined = tracker.aggregate({"agent_a": 0.70, "agent_b": 0.60})

# Leaderboard ranked by Brier score
board = tracker.leaderboard()
```

### 4. Ergodicity-Corrected Kelly

Standard Kelly assumes i.i.d. bets. Real systems have serial correlation and path-dependent drawdowns. The correction factor is always <= 1.0.

```python
from agent_algebra import ergodic_kelly

result = ergodic_kelly(
    win_rate=0.60,
    win_loss_ratio=1.5,
    returns=historical_returns,
    n_paths=1000,
)
# result.kelly_fraction = 0.267
# result.ergodic_fraction = 0.192  (corrected for path dependence)
# result.correction_factor = 0.72
```

### 5. Belief Propagation

Multi-source signal fusion. Each agent communicates only with neighbors, converges to globally optimal posteriors.

```python
from agent_algebra import BeliefNode, build_graph, propagate

agents = [
    BeliefNode("regime", local_prob=0.60),
    BeliefNode("micro", local_prob=0.75),
    BeliefNode("sentiment", local_prob=0.55),
]
graph = build_graph(agents, [("regime", "micro"), ("micro", "sentiment")])
result = propagate(graph, damping=0.3)
# result.beliefs: globally consistent posteriors
# result.converged: True in 8 iterations
```

### 6. MDL Compression

If a pattern compresses, it's signal. Otherwise, noise. Uses zlib as a Kolmogorov complexity estimator.

```python
from agent_algebra import mdl_filter, algorithmic_compression_ratio

ratio = algorithmic_compression_ratio(data)  # low = signal, high = noise
signals = mdl_filter(items, summarize_fn, reconstruct_fn)
```

## Pipeline Composition

Chain primitives into reusable stacks. Each `.add()` returns a new Pipeline (immutable builder pattern).

```python
from agent_algebra import Pipeline

pipeline = (
    Pipeline()
    .add("belief", lambda d: propagate(build_graph(agents, edges)).beliefs)
    .add("score", lambda beliefs: tracker.aggregate(beliefs))
    .add("size", lambda prob: prob * ergodic.ergodic_fraction)
)

# Traced execution shows every intermediate value
trace = pipeline.run_traced(input_data)
for step_name, value in trace:
    print(f"[{step_name}] {value}")
```

**Pre-built stacks:**
- **Trading**: belief -> scoring -> boost -> ergodic -> contraction
- **Intel/Forecasting**: compress -> boost -> scoring -> contraction

## Architecture

```
agent_algebra/
├── types.py        # Prediction, Outcome, AgentRecord (frozen dataclasses)
├── scoring.py      # Brier, log score, ScoringTracker, log pool aggregation
├── contraction.py  # Contraction mapping loop + Bayesian seed conversion
├── boost.py        # AdaBoost cascade for agent ensembles
├── ergodic.py      # Kelly criterion + ergodicity correction via Monte Carlo
├── belief.py       # Belief propagation on agent graphs
├── compress.py     # MDL signal/noise classification
├── compose.py      # Pipeline composition (immutable builder)
└── store.py        # SQLite persistence for calibration history
```

## Design Principles

- **Immutable** — All types are frozen dataclasses. All mutations return new objects.
- **Zero dependencies** — stdlib only. No numpy, no pandas, no frameworks.
- **Composable** — Each primitive works standalone or in a Pipeline.
- **Tested** — Full test suite in `tests/`.

## Articles

Deep-dive articles with production examples from live trading systems:

- [Agent Algebra: Theorem-Guided Composition for Self-Improving AI Systems](https://kris-welc.github.io/portfolio/articles/agent-algebra)
- [Dual-Layer Regime Detection: Structural Classification Meets Statistical Drift](https://kris-welc.github.io/portfolio/articles/dual-layer-regime)
- [VPIN as Real-Time Conviction Modifier](https://kris-welc.github.io/portfolio/articles/vpin-conviction)

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
pytest --cov=agent_algebra --cov-report=term-missing
```

## License

MIT — see [LICENSE](LICENSE).
