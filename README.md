# QuantScenarioBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21097248.svg)](https://doi.org/10.5281/zenodo.21097248)

A JAX-native Python framework for generating reproducible stochastic market scenarios and benchmarking portfolio strategies against them, with built-in export to Parquet and the Hugging Face Hub.

```python
from quantscenariobench.api import simulate
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import Heston
import jax.numpy as jnp

model = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)
tg    = TimeGrid(jnp.linspace(0.0, 1.0, 253))   # 252 daily steps, 1 year

scenario = simulate(model, tg, n_paths=10_000, seed=42)

print(scenario.observation.shape)    # (10000, 253)  — asset price paths
print(scenario.latent_state.shape)   # (10000, 253)  — variance paths
```

QuantScenarioBench is more than a published dataset — it's an end-to-end, Hugging Face-native pipeline:

1. **Scenario generation** — reproducible price paths from established stochastic volatility models (`quantscenariobench.api`, `.models`)
2. **Benchmark Core** — a shared interface, baselines, and metrics for scoring portfolio strategies against those Scenarios (`quantscenariobench.benchmark`)
3. **Evaluation Results** — a versioned, JSON-native record of each benchmark run, derived from `BenchmarkResult` (`quantscenariobench.benchmark.evaluation`)
4. **Leaderboard aggregation** — a ranked strategy × Benchmark Dataset table built from every published Evaluation Result
5. **Leaderboard Space** — a hosted, sortable, filterable Hugging Face Space that renders that ranked table live, so anyone can browse results without running any code (`spaces/leaderboard/`, see [Leaderboard Space](#leaderboard-space))
6. **Hugging Face-native workflow** throughout — datasets, dataset cards, and Evaluation Results all publish to and load from the Hub with the same functions used for local storage

---

## Why QuantScenarioBench?

Quantitative finance research routinely benchmarks models against each other, but reproducible, openly published path datasets — and a standardized way to score strategies against them — are rare. QuantScenarioBench is useful for model benchmarking, evaluation, stress testing, and general experimentation with stochastic volatility models, and makes it straightforward to:

- **Generate** large batches of price paths from established stochastic volatility models with a single call
- **Compare** models on a shared schema — every `Scenario` has the same fields regardless of the model used
- **Export** results to Parquet or publish directly to the Hugging Face Hub
- **Reproduce** any result exactly — the same `(model, time_grid, n_paths, seed)` always produces bit-identical paths on the same computational backend
- **Benchmark** portfolio strategies against generated Scenarios with a shared `run_benchmark()` pipeline and standardized performance metrics
- **Publish, aggregate, and browse** benchmark runs as versioned Evaluation Results, ranked into a Leaderboard table you can query yourself or browse live on the hosted [Leaderboard Space](#leaderboard-space)

---

## Installation

```bash
pip install quantscenariobench
```

**Requirements:** Python ≥ 3.11. Core dependencies — `jax`, `diffrax`, `equinox`, `pyarrow`, `scipy` (Optimizer Solver Layer for GMV's long-only path and CVaR Optimization), and `huggingface_hub` (Hugging Face Hub publishing, both datasets and Evaluation Results) — install automatically; there is no separate opt-in step for Hugging Face support.

For development (running the test suite, loading datasets in examples):

```bash
pip install "quantscenariobench[dev]"   # adds pytest, pandas, datasets
```

---

## Quick Start

### 1. Generate a scenario

```python
import quantscenariobench   # enables JAX float64 globally
from quantscenariobench.api import simulate
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes, Heston, RoughBergomi
import jax.numpy as jnp

tg = TimeGrid(jnp.linspace(0.0, 1.0, 253))   # daily steps over 1 year

# Black-Scholes (no latent state)
bs = BlackScholes(mu=0.0, sigma=0.2, S0=100.0)
s  = simulate(bs, tg, n_paths=50_000, seed=0)
print(s.observation.shape)    # (50000, 253)
print(s.latent_state.shape)   # (50000, 0)   — empty for GBM

# Heston stochastic volatility
heston = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)
s      = simulate(heston, tg, n_paths=50_000, seed=0)
print(s.observation.shape)    # (50000, 253)  — asset price paths
print(s.latent_state.shape)   # (50000, 253)  — variance paths

# Rough Bergomi (non-Markovian, Hurst H=0.1)
rb = RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, S0=100.0, mu=0.0)
s  = simulate(rb, tg, n_paths=50_000, seed=0)
```

### 2. Export to Parquet

```python
from quantscenariobench.export import export_parquet

export_parquet([s], "my_dataset.parquet")
```

Every exported file follows the same 12-column schema regardless of model:

| Column | Type | Description |
|--------|------|-------------|
| `observation` | `list<float64>` | Asset price path (one row per path) |
| `latent_state` | `list<float64>` | Latent state path (empty for Black-Scholes) |
| `seed` | `int64` | Integer PRNG seed |
| `prng_key_info` | `string` | JAX PRNGKey derivation description |
| `model_name` | `string` | Market model class name |
| `model_version` | `string` | Model specification version |
| `parameters` | `string` | JSON-encoded model parameters |
| `time_grid` | `string` | JSON-encoded time points |
| `n_paths` | `int64` | Number of simulation paths |
| `library_version` | `string` | `quantscenariobench` version |
| `dataset_version` | `string` | Dataset version identifier |
| `generated_at` | `string` | UTC ISO-8601 generation timestamp |

### 3. Publish to Hugging Face Hub

```python
from quantscenariobench.export import publish_to_hub

url = publish_to_hub([s], "my-org/my-dataset", token="hf_...")
print(url)   # https://huggingface.co/datasets/my-org/my-dataset
```

`publish_to_hub` writes the Parquet file and a dataset card (README.md) in a single call, then uploads both to the Hub.

### 4. Replay paths deterministically

```python
scenario, dW = simulate(heston, tg, n_paths=1000, seed=7, return_randomness=True)

# Later, reproduce the exact same paths from the saved increments
replayed = simulate(heston, tg, n_paths=1000, seed=7, randomness=dW)
```

---

## Available Models

### Black-Scholes (`BlackScholes`)

Geometric Brownian Motion:

$$dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t$$

| Parameter | Description |
|-----------|-------------|
| `mu` | Drift (use `0.0` for risk-neutral) |
| `sigma` | Constant volatility |
| `S0` | Initial asset price |

### Heston (`Heston`)

Stochastic volatility with mean-reverting variance:

$$dS_t = \mu\,S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\,\sqrt{v_t}\,dW^v_t, \quad \text{Corr}(dW^S, dW^v) = \rho$$

| Parameter | Description |
|-----------|-------------|
| `mu` | Asset drift |
| `kappa` | Variance mean-reversion speed |
| `theta` | Long-run variance |
| `xi` | Vol-of-vol |
| `rho` | Asset–variance correlation (leverage effect) |
| `v0` | Initial variance |
| `S0` | Initial asset price |

The Feller condition `2κθ ≥ ξ²` ensures the variance process stays positive. Violation emits a `QuantScenarioBenchValidationWarning`.

### Rough Bergomi (`RoughBergomi`)

Non-Markovian stochastic volatility driven by fractional Brownian motion:

$$V_t = \xi_0 \exp\!\left(\eta\,W^H_t - \tfrac{1}{2}\eta^2 t^{2H}\right), \quad dS_t = \mu\,S_t\,dt + \sqrt{V_t}\,S_t\,dW^S_t$$

where $W^H_t$ is a Riemann–Liouville fractional Brownian motion with Hurst exponent $H$, discretised via the Volterra representation.

| Parameter | Description |
|-----------|-------------|
| `H` | Hurst exponent; `H < 0.5` for rough volatility (empirically `H ≈ 0.1`) |
| `eta` | Vol-of-vol amplitude |
| `rho` | Correlation between asset BM and fBM driver |
| `xi0` | Initial variance level |
| `S0` | Initial asset price |
| `mu` | Asset drift |

---

## Reproducibility

Identical `(model, time_grid, n_paths, seed)` inputs produce **bit-identical paths** on the same computational backend (CPU / GPU / TPU). Cross-backend bit-identity is not guaranteed due to floating-point differences across JAX backends. The `seed`, `prng_key_info`, and `library_version` metadata fields document full provenance for every batch.

All simulations run in **float64** (enabled automatically on `import quantscenariobench`).

---

## Pre-built Benchmark Samples

Three lightweight demo samples are published on Hugging Face. They use a single fixed configuration (10,000 paths, daily steps over 1 year, seed 42) and are sized for quick loading, not research-scale use. They are useful for inspecting the output schema and doing quick cross-model comparisons without generating anything locally.

| Model | Sample | Paths | Steps |
|-------|--------|-------|-------|
| Black-Scholes | [QuantScenarioBench/qsb-black-scholes](https://huggingface.co/datasets/QuantScenarioBench/qsb-black-scholes) | 10,000 | 253 (daily, 1 yr) |
| Heston | [QuantScenarioBench/qsb-heston](https://huggingface.co/datasets/QuantScenarioBench/qsb-heston) | 10,000 | 253 (daily, 1 yr) |
| Rough Bergomi | [QuantScenarioBench/qsb-rough-bergomi](https://huggingface.co/datasets/QuantScenarioBench/qsb-rough-bergomi) | 10,000 | 253 (daily, 1 yr) |

All three use the same time grid (`linspace(0, 1, 253)`), seed (`42`), and initial spot (`S0=100`, `mu=0`) for direct cross-model comparison.

```python
from datasets import load_dataset

ds = load_dataset("QuantScenarioBench/qsb-heston", split="train")
print(ds.column_names)
# ['observation', 'latent_state', 'seed', 'prng_key_info', 'model_name',
#  'model_version', 'parameters', 'time_grid', 'n_paths',
#  'library_version', 'dataset_version', 'generated_at']
```

For research use, generate your own dataset with your chosen horizon, time grid, parameters, and number of paths using `simulate()`, then export it with `export_parquet()` or `publish_to_hub()`.

---

## Benchmark Core

`quantscenariobench.benchmark` scores portfolio strategies against Scenarios using a shared pipeline, mirroring the Market Model layer's interface-plus-conformance-suite design.

```python
from quantscenariobench.benchmark.returns import derive_returns, compose_returns
from quantscenariobench.benchmark.strategies import EqualWeight, GlobalMinimumVariance, CVaROptimization
from quantscenariobench.benchmark.runner import run_benchmark

# One Scenario per asset, all sharing the same TimeGrid (a single path per asset)
returns = compose_returns([scenario_a, scenario_b, scenario_c])   # shape (t, n_assets)
historical_returns, evaluation_returns = returns[:126], returns[126:]

strategy = EqualWeight()
result = run_benchmark(strategy, historical_returns, evaluation_returns)

print(result.metrics)
# {'sharpe_ratio': ..., 'sortino_ratio': ..., 'max_drawdown': ..., 'final_wealth_factor': ...}
```

- **Portfolio Optimizer Interface** (`quantscenariobench.benchmark.interface`) — `BaselineStrategy` (`allocate(historical_returns)`) and `ForecastOptimizer` (`allocate(historical_returns, forecast)`), both `equinox.Module` ABCs, plus a validated `PortfolioWeights` type (long-only, sums to 1).
- **Traditional baselines** (`quantscenariobench.benchmark.strategies`) — `EqualWeight`, `GlobalMinimumVariance(long_only=...)`, `CVaROptimization(confidence_level=...)`.
- **Metrics** (`quantscenariobench.benchmark.metrics`) — `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `final_wealth_factor`, assembled in `DEFAULT_METRICS`; all pure `jax.numpy` functions with defined sentinel behavior on degenerate input (e.g. zero variance).
- **`run_benchmark()`** (`quantscenariobench.benchmark.runner`) fits the strategy once (static buy-and-hold), applies its weights across `evaluation_returns`, and returns a JSON-serializable `BenchmarkResult` (`strategy_name`, `strategy_parameters`, `metrics`, `asset_scenario_ids`, `time_grid_reference`, `library_version`, `generated_at`).
- A conformance test suite (`quantscenariobench.benchmark.testing`) verifies a custom `BaselineStrategy`/`ForecastOptimizer` implementation against the interface, with zero changes to `run_benchmark()`.

### Implementing a custom strategy

```python
import jax.numpy as jnp
from quantscenariobench.benchmark.interface import BaselineStrategy, PortfolioWeights

class MyStrategy(BaselineStrategy):
    def allocate(self, historical_returns):
        n = historical_returns.shape[1]
        return PortfolioWeights(jnp.full((n,), 1.0 / n), n_assets=n)
```

Pass it directly to `run_benchmark()` — no other integration required.

---

## Evaluation Results & Leaderboard

`quantscenariobench.benchmark.evaluation` turns a `BenchmarkResult` into a versioned, publishable record, and reads any collection of them back into a ranked comparison table. This section covers the publishing/aggregation pipeline itself — for the hosted, browsable page built on top of it, see [Leaderboard Space](#leaderboard-space).

```python
from quantscenariobench.benchmark.evaluation import (
    to_evaluation_result,
    write_evaluation_result,
    publish_evaluation_results,
    load_evaluation_results,
    aggregate_evaluation_results,
)

# 1. Convert a BenchmarkResult (from run_benchmark()) into an EvaluationResult
evaluation_result = to_evaluation_result(result)

# 2. Store it locally — one timestamped file per run, organized by dataset/strategy
write_evaluation_result(evaluation_result, root="results")

# 3. Publish it to a shared Hugging Face dataset repo (append-only)
publish_evaluation_results([evaluation_result], "my-org/qsb-evaluation-results", token="hf_...")

# 4. Aggregate every locally stored (or downloaded) result into a leaderboard table
results = load_evaluation_results("results")
leaderboard = aggregate_evaluation_results(results)
print(leaderboard)
# [{'strategy': 'EqualWeight', 'benchmark_dataset': '...', 'sharpe_ratio': ..., ...}, ...]
```

- **`EvaluationResult`** — a fixed, JSON-native schema (`schema_version`, `result_id`, `strategy`, `benchmark_dataset`, `metrics` as an ordered `{name, value}` list, `library_version`, `generated_at`), derived from `BenchmarkResult` via the pure `to_evaluation_result()` function. `BenchmarkResult` itself is unchanged — `EvaluationResult` is a separate, additive publication-layer type.
- **Local storage** (`write_evaluation_result`) writes one timestamped JSON file per run under `results/<dataset>/<strategy>/`; nothing is ever overwritten.
- **Hugging Face publishing** (`publish_evaluation_results`, `generate_evaluation_results_card`) uploads results to a shared dataset repo and regenerates a summary README/card after every publish.
- **Leaderboard aggregation** (`aggregate_evaluation_results`, `load_evaluation_results`, `load_evaluation_results_from_hub`) is a generic reader — no strategy- or dataset-specific branching — that returns a plain `list[dict]`: one row per strategy × Benchmark Dataset, the most recently generated result winning ties. It has no UI framework dependency; render it with `pandas.DataFrame(leaderboard)` or however you like.

---

## Leaderboard Space

A hosted, browsable Hugging Face Space — built with [Gradio](https://www.gradio.app/) — that renders the Leaderboard above as a live, sortable, filterable page. It lives at `spaces/leaderboard/`, alongside (not inside) the `quantscenariobench` package, and consumes the pipeline described in [Evaluation Results & Leaderboard](#evaluation-results--leaderboard) as an ordinary dependency: it calls `aggregate_evaluation_results()`/`load_evaluation_results_from_hub()` and renders the result. It adds no aggregation, ranking, or data-model logic of its own — every value shown comes directly from that pipeline.

- **Table rendering** — the current ranked Leaderboard (strategy × Benchmark Dataset rows, one column per Metric), reloaded fresh every time the page is opened, so a newly published `EvaluationResult` appears without redeploying the Space.
- **Sorting** — click any column header to reorder rows ascending or descending; this is Gradio's built-in `Dataframe` behavior, not custom code.
- **Filtering** — narrow the table by Benchmark Dataset, Strategy, or Metric, independently or combined.
- **Out of scope** — advanced analytics, visualizations (charts/plots), historical/trend views, and strategy-to-strategy comparison tooling are deliberately not part of this Space; it is a ranked table, not an analytics dashboard.

### Running it locally

```bash
cd spaces/leaderboard
pip install -r requirements.txt
python app.py
```

This starts a local Gradio server rendering the Leaderboard from whichever Evaluation Results repo is configured (see below).

### Configuring the Evaluation Results repo

The Space reads from the Hugging Face dataset repo named by the `QSB_EVAL_RESULTS_REPO` environment variable:

```bash
export QSB_EVAL_RESULTS_REPO="your-org/your-evaluation-results-repo"
```

If unset, it falls back to a placeholder default (`quantscenariobench/evaluation-results`) — **the actual namespace for a shared, public Evaluation Results repo has not been finalized yet**, so there is no canonical repo ID or hosted Space URL to publish here. Point `QSB_EVAL_RESULTS_REPO` at your own published Evaluation Results repo (see [Hugging Face publishing](#evaluation-results--leaderboard) above) to run the Space against real data.

---

## Architecture

```
quantscenariobench/
├── api/          simulate() — single public entry point
├── interface/    MarketModel ABC, Scenario, TimeGrid, Metadata
├── models/       BlackScholes, Heston, RoughBergomi
├── solver/       Euler-Maruyama SDE solver (diffrax / lineax)
├── export/       export_parquet(), generate_dataset_card(), publish_to_hub()
├── testing/      Conformance suite for custom model authors
└── benchmark/
    ├── interface/    BaselineStrategy/ForecastOptimizer ABCs, PortfolioWeights, BenchmarkResult
    ├── strategies/   EqualWeight, GlobalMinimumVariance, CVaROptimization
    ├── metrics/      sharpe_ratio, sortino_ratio, max_drawdown, final_wealth_factor
    ├── returns/      derive_returns(), compose_returns()
    ├── solver/       scipy-backed Optimizer Solver Layer (GMV long-only, CVaR)
    ├── runner/       run_benchmark() — single public entry point
    ├── evaluation/   EvaluationResult, to_evaluation_result(), local storage,
    │                 HF publishing, Leaderboard aggregation
    └── testing/      Conformance suite for custom strategy authors

spaces/
└── leaderboard/  Hugging Face Space (Gradio) — see Leaderboard Space above.
                  Sibling to quantscenariobench/, not part of the installable
                  package; depends on it like any other consumer.
```

**Dependency rule:** `models` and `export` import only from `interface`. The solver and API layers compose these. This keeps any custom model implementation minimal. The `benchmark` subpackage mirrors this rule one layer up: `strategies`/`metrics`/`returns` import only from `benchmark.interface` (plus `benchmark.solver` where noted), and only `benchmark.runner` composes a caller-supplied strategy — no benchmark module reaches back into `models` or the scenario-generation `solver`.

### Implementing a custom model

Subclass `MarketModel` (an `equinox.Module`) and implement three methods:

```python
import equinox as eqx
import jax.numpy as jnp
from quantscenariobench.interface import MarketModel

class MyModel(MarketModel):
    sigma: float
    S0: float

    def _drift(self, t, state):
        return jnp.zeros_like(state)   # zero drift

    def _diffusion(self, t, state):
        return self.sigma * state      # scalar diffusion

    def initial_state(self):
        return jnp.array(self.S0, dtype=float)
```

Pass the model directly to `simulate()` — no other integration required.

---

## Development

```bash
git clone https://github.com/tim-nish/QuantScenarioBench
cd QuantScenarioBench
pip install -e ".[dev]"
pytest
```

The test suite covers closed-form price validation (Gil-Pélaez inversion for Heston; Black-Scholes formula for GBM), statistical properties (skew monotonicity in H for rBergomi), Parquet round-trips, and dataset card conformance — plus the Benchmark Core (metrics and baselines validated against hand-derived reference values, Portfolio Optimizer conformance suite), the Evaluation Results pipeline (`BenchmarkResult` → `EvaluationResult` transform, local storage, and Leaderboard aggregation), and the Leaderboard Space (`spaces/leaderboard/`: data loading/rendering, sorting, filtering, and deployment configuration).

---

## Roadmap

| Capability | Status |
|---|---|
| Scenario generation (Black-Scholes, Heston, Rough Bergomi) | Shipped — v1.0 |
| Parquet export & Hugging Face dataset publishing | Shipped — v1.0 |
| Benchmark Core (Portfolio Optimizer Interface, baselines, metrics, `run_benchmark()`) | Shipped — v1.1 |
| EvaluationResult pipeline (transform, local storage, HF publishing) | Shipped — v1.1 |
| Leaderboard **aggregation** (ranked table from published results, no UI) | Shipped — v1.1 |
| Hugging Face Space — hosted Gradio Leaderboard **UI**, with sorting and filtering | Shipped — v1.2 |
| Advanced analytics, visualizations, historical/trend tracking, strategy-comparison tooling | Not planned |

**v1.1 shipped the data; v1.2 ships the dashboard:** `aggregate_evaluation_results()` still returns a plain `list[dict]` you can put in a `pandas.DataFrame`, a notebook, or your own app — but as of v1.2 you can also browse the same ranked table live on the hosted [Leaderboard Space](#leaderboard-space), with sorting and filtering built in. That Space was the explicit v1.2 goal referenced in earlier releases' roadmaps; it is deliberately scoped to the ranked table alone — see the Leaderboard Space section's "Out of scope" note for what's intentionally not included.

---

## License

MIT
