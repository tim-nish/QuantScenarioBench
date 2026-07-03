---
stepsCompleted: [1, 2, 3]
inputDocuments: ['_bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/prd.md', '_bmad-output/planning-artifacts/architecture/architecture-QuantScenarioBench-2026-06-30/ARCHITECTURE-SPINE.md']
---

# QuantScenarioBench - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for QuantScenarioBench, decomposing the requirements from the PRD and the Architecture Spine into implementable stories. There is no UX design document — QuantScenarioBench is an API-first Python framework with no UI.

## Requirements Inventory

### Functional Requirements

FR-1: A researcher can call `simulate(model=<MarketModel>, time_grid=<TimeGrid>, n_paths=<int>, seed=<int>) -> Scenario` identically across all Market Models.
FR-2: Every `simulate()` call returns a `Scenario` exposing `observation`, `latent_state`, and `metadata`, with identical shape and field names across all Market Models.
FR-3: Simulation time points are specified via an explicit `TimeGrid` object, supporting non-uniform spacing.
FR-4: Given the same Market Model, parameters, TimeGrid, and seed, `simulate()` produces a deterministic `Scenario` on a given backend; `Metadata` carries the provenance needed to describe generation.
FR-5: A caller can request the underlying `Randomness` used to generate a Scenario; by default it is not materialized or returned.
FR-6: Market Model parameter construction validates research-meaningful constraints where practical (e.g. the Heston Feller condition) and emits a warning — never a hard exception — when violated.
FR-7: A `BlackScholes` Market Model conforms to the State-Space Interface, with no `latent_state`.
FR-8: A `Heston` Market Model conforms to the State-Space Interface, exposing the variance process as `latent_state`.
FR-9: A `RoughBergomi` Market Model conforms to the State-Space Interface, exposing its volatility process as `latent_state`.
FR-10: A documented interface specifies exactly what a Market Model must implement to satisfy the State-Space Interface and be usable by `simulate()` and the export pipeline, with zero changes to either's source.
FR-11: A reusable conformance test suite verifies a Market Model implementation satisfies the State-Space Interface, including a test-only dummy Market Model that proves the interface holds independent of any real financial model.
FR-12: A batch of Scenarios for a given Market Model serializes to Parquet file(s) with columns mirroring the Scenario schema.
FR-13: Each v1 Market Model publishes as its own Hugging Face dataset, all three sharing the same top-level column schema.
FR-14: Published Benchmark Datasets carry their own version identifier, decoupled from the library's release version.
FR-15: Each published Benchmark Dataset includes a dataset card with: column schema, Market Model name and parameter values, TimeGrid and n_paths, library version, and dataset version.

**Benchmark capability layer (added 2026-07-02, PRD Features 4.5–4.8):**

FR-16: Given a Portfolio Return series, compute Sharpe Ratio (risk-free rate 0, no annualization); a zero-variance input returns a defined sentinel (`0.0`) rather than raising.
FR-17: Given a Portfolio Return series, compute Sortino Ratio (risk-free rate 0, no annualization); a no-downside input returns a defined sentinel (`0.0`) rather than raising.
FR-18: Given a Portfolio Return series, compute Maximum Drawdown.
FR-19: Given a Portfolio Return series, compute Final Wealth Factor.
FR-20: An `EqualWeight` `BaselineStrategy` allocates equal Portfolio Weights across all assets, independent of historical returns.
FR-21: A `GlobalMinimumVariance` `BaselineStrategy` computes Portfolio Weights minimizing portfolio variance given the historical returns' covariance structure (long-only in v1; unconstrained path is closed-form, constrained path uses the Optimizer Solver Layer).
FR-22: A `CVaROptimization` `BaselineStrategy` computes Portfolio Weights minimizing Conditional Value-at-Risk at a confidence level (v1 default 95%, confirmed by user).
FR-23: A documented `BaselineStrategy` interface specifies `historical_returns -> weights`, usable by the Benchmark Runner with zero Runner changes.
FR-24: A documented `ForecastOptimizer` interface specifies `historical_returns + forecast -> weights` (`forecast` is a fixed-shape point forecast, one predicted return per asset), usable by the Benchmark Runner with zero Runner changes.
FR-25: A reusable conformance test suite verifies a `BaselineStrategy`/`ForecastOptimizer` implementation satisfies the Portfolio Optimizer Interface, including a test-only dummy `ForecastOptimizer` that proves the interface holds independent of any real forecasting model.
FR-26: The Benchmark Runner composes a multi-asset portfolio from multiple independently generated/loaded single-asset Scenarios; all constituent Scenarios must share an identical `TimeGrid` or the Runner raises.
FR-27: The Benchmark Runner orchestrates `returns/scenarios -> strategy -> weights -> portfolio returns -> metrics -> BenchmarkResult`; a strategy is fit exactly once per run (static buy-and-hold, confirmed by user) against a caller-supplied `historical_returns`/`evaluation_returns` split drawn from the same Scenario(s) — no intra-run rebalancing.
FR-28: The Benchmark Runner derives a return series from a Scenario's `observation` using one consistent convention: simple/arithmetic period return, sampled once per `TimeGrid` step (confirmed by user).
FR-29: Each Benchmark Runner run produces a JSON-serializable `BenchmarkResult` with a fixed minimum field set: `strategy_name`, `strategy_parameters`, `metrics` (flat dict), `asset_scenario_ids`, `time_grid_reference`, `library_version`, `generated_at`.

**Evaluation Results & Leaderboard layer (added 2026-07-03, PRD Feature 4.9):**

FR-30: A fixed, JSON-native `EvaluationResult` schema, derived from `BenchmarkResult` (AD-26), carries at minimum: `schema_version`, `result_id`, strategy identity/parameters, Benchmark Dataset/Scenario identity, `metrics` as an ordered list of `{name, value}` records, `library_version`, and `generated_at`.
FR-31: A single pure function transforms a `BenchmarkResult` into an `EvaluationResult`, with no mutation of `BenchmarkResult` and zero changes required to `run_benchmark()`.
FR-32: An `EvaluationResult` can be written to a local file layout organized by Benchmark Dataset and strategy — one timestamped JSON file per run, never overwritten.
FR-33: An `EvaluationResult` (or a batch of them) can be published to a shared Hugging Face Evaluation Results dataset repo, with an auto-generated README/card summarizing its contents.
FR-34: A generic reader loads every published `EvaluationResult` and builds a ranked Leaderboard table (rows = strategy × Benchmark Dataset, columns = Metrics), with no strategy-specific or dataset-specific branching, and no dependency on any UI framework.

### NonFunctional Requirements

NFR-1: Determinism is backend-scoped — reproducibility (FR-4) holds for repeated runs on the same backend; cross-backend (CPU/GPU/TPU) bit-identity is not guaranteed.
NFR-2: All Market Model simulation logic is implemented in JAX (jit/vmap-compatible). **Extended 2026-07-02:** the benchmark layer (Metrics, return derivation, the Runner, `PortfolioWeights`/`BenchmarkResult` types, `EqualWeight`, GMV's unconstrained path) is held to the same JAX-native standard; the Optimizer Solver Layer's use of `scipy` for GMV's long-only-constrained path and CVaR Optimization is the sole, deliberately bounded, swappable exception (AD-25).
NFR-3: Every v1 Market Model has an automated correctness check appropriate to its mathematical character (closed-form, semi-closed-form, or statistical sanity check). **Extended 2026-07-02:** the same convention applies to the four v1 Metrics and three Traditional Baselines — each validated against an independently hand-derived reference value, never a bundled portfolio-analytics library (AD-10 amended).
NFR-4: Backward-incompatible changes to `simulate()`, the `Scenario` schema, or the State-Space Interface require a major version bump (public API semantic versioning). **Extended 2026-07-02:** the same policy covers `BaselineStrategy`/`ForecastOptimizer`'s `allocate()` signature and the `BenchmarkResult` schema (AD-11 amended).
NFR-5: Language/runtime targets: Python >=3.11, jax >=0.4.38 (confirmed during Architecture, supersedes the PRD's original 3.10+ assumption).
NFR-6 *(added 2026-07-02)*: A `BenchmarkResult` is losslessly JSON-serializable and deterministic for identical strategy/returns input — the portfolio-benchmarking analogue of NFR-1's reproducibility guarantee (AD-17, AD-24).
NFR-7 *(added 2026-07-03)*: An `EvaluationResult` is losslessly JSON-serializable and deterministic for an identical input `BenchmarkResult` — the publication-layer analogue of NFR-6, one layer up (AD-26).

### Additional Requirements (from Architecture Spine)

**No starter template** — this is a from-scratch Python package; the Architecture Spine's Structural Seed source tree is the initial package layout for Epic 1 Story 1, not a third-party starter.

**Stack (pinned in Architecture):** Python >=3.11, jax >=0.4.38, diffrax 0.7.2, equinox >=0.11.10. **Added 2026-07-02:** scipy >=1.16 (floor pin; used only inside the Optimizer Solver Layer for GMV's long-only-constrained path and CVaR Optimization's linear program).

**Package layout (Structural Seed):** `quantscenariobench.interface` (State-Space Interface, Scenario, TimeGrid), `quantscenariobench.models` (Black-Scholes, Heston, RoughBergomi), `quantscenariobench.solver` (internal Solver Layer, the only module importing diffrax), `quantscenariobench.api` (`simulate()`), `quantscenariobench.export` (Parquet/HF export), `quantscenariobench.testing` (conformance suite + dummy Market Model).

**Benchmark-layer package layout (added 2026-07-02):** `quantscenariobench.benchmark.interface` (`BaselineStrategy`/`ForecastOptimizer` ABCs, `PortfolioWeights`, `BenchmarkResult`), `quantscenariobench.benchmark.strategies` (`EqualWeight`, `GlobalMinimumVariance`, `CVaROptimization`), `quantscenariobench.benchmark.metrics` (Sharpe, Sortino, MaxDrawdown, FinalWealthFactor as pure `MetricFn`s), `quantscenariobench.benchmark.returns` (`derive_returns(scenario)`), `quantscenariobench.benchmark.solver` (internal Optimizer Solver Layer, the only benchmark module importing `scipy`), `quantscenariobench.benchmark.runner` (`run_benchmark()`), `quantscenariobench.benchmark.testing` (Portfolio Optimizer conformance suite + dummy `ForecastOptimizer`).

**Evaluation Results package layout (added 2026-07-03):** `quantscenariobench.benchmark.evaluation` — `EvaluationResult`, `to_evaluation_result()` (FR-31), local storage (FR-32), Hugging Face publishing (FR-33), Leaderboard aggregation (FR-34); depends on `quantscenariobench.benchmark.interface` only (AD-26).

**Architecture Decisions (ADs) governing implementation:**
- AD-1: State-Space Interface is an `equinox.Module` ABC.
- AD-2: `Scenario` is an `equinox.Module` with `observation`/`latent_state` dynamic and `metadata` static.
- AD-3: Randomness defaults to `diffrax.VirtualBrownianTree`; materialization is a separate, explicit opt-in path.
- AD-4: Solver Layer wraps `diffrax` exclusively, behind one fixed `_drift`/`_diffusion` signature; these are Solver-Layer-internal, not public API.
- AD-5: Dataset export is generic over the `Scenario` schema; never imports a concrete Market Model.
- AD-6: `equinox` is a project-wide pytree convention (not diffrax-only).
- AD-7: float64 (JAX x64) is the fixed v1 precision policy, enabled once in `quantscenariobench/__init__.py`.
- AD-8: Metadata's minimum guaranteed field set: `seed`, `prng_key_info`, `model_name`, `model_version`, `parameters`, `time_grid`, `n_paths`, `library_version`, `dataset_version`, `generated_at`.
- AD-9: One-way dependency direction (Models → Interface ← Solver/API/Export/Testing); `equinox` is project-wide, `diffrax` is solver-exclusive.
- AD-10: Correctness-check references are independently implemented, never borrowed from a bundled quant library. **Amended 2026-07-02:** also covers Metrics/Traditional Baselines correctness checks.
- AD-11: Public API stability follows semantic versioning, independent of dataset versioning. **Amended 2026-07-02:** also covers `BaselineStrategy`/`ForecastOptimizer`/`BenchmarkResult`.
- AD-12: `TimeGrid` is an explicit, ordered time-point sequence, not a generative spec.
- **AD-13** *(added 2026-07-02)*: Portfolio Optimizer Interface is an `equinox.Module` ABC split into `BaselineStrategy` (`allocate(historical_returns) -> PortfolioWeights`) and `ForecastOptimizer` (`allocate(historical_returns, forecast) -> PortfolioWeights`).
- **AD-14**: Optimizer Solver Layer (`quantscenariobench.benchmark.solver`) is the only module importing `scipy`; `GlobalMinimumVariance` takes an explicit `long_only: bool` (unconstrained = `jax.numpy.linalg`, constrained = `scipy.optimize.minimize`/SLSQP via the solver layer); `CVaROptimization` always uses `scipy.optimize.linprog` via the solver layer; solver non-convergence raises rather than returning degenerate weights.
- **AD-15**: `CVaROptimization.confidence_level` is a required, recorded constructor parameter (v1 default 0.95, confirmed by user).
- **AD-16**: Return-series derivation is exactly one function, `quantscenariobench.benchmark.returns.derive_returns(scenario)` — arithmetic return, per-`TimeGrid`-step (confirmed by user), `jax.numpy`/`jit`-native.
- **AD-17**: `BenchmarkResult` is a plain, JSON-native Python dataclass, not an `equinox.Module` (it is never re-traced through `jit`/`vmap`, unlike `Scenario`).
- **AD-18**: Metrics are pure `MetricFn`s (`jax.numpy`/`jit`-native) in an explicit ordered `Sequence` registry passed to the Runner (never a `dict`); duplicate `.name`s raise; Sharpe/Sortino return `0.0` on a zero denominator.
- **AD-19**: Benchmark-layer dependency direction mirrors AD-9 — `strategies` may import only `benchmark.interface`/`benchmark.solver`/`equinox`; `metrics`/`returns` depend on nothing benchmark-specific beyond `quantscenariobench.interface`; only `benchmark.runner` imports concrete strategies, and only as caller-supplied arguments; no benchmark module imports `quantscenariobench.models`/`quantscenariobench.solver`.
- **AD-20**: `PortfolioWeights` (`Float[Array, "n"]`) is a validated value type — entries sum to 1 within `1e-6`, all entries `>= 0` (long-only, universal for v1), `n` matches asset count — enforced at construction, not left to strategy discipline.
- **AD-21**: `ForecastOptimizer.forecast` is a fixed-shape point forecast (`Float[Array, "n"]`, one predicted return per asset) — distributional/quantile forecasts are out of scope for v1.
- **AD-22**: Multi-asset composition requires every constituent Scenario to share an identical `TimeGrid` (mismatch raises before return derivation); `Scenario.observation` must be a 1D, strictly-positive price series to be benchmark-layer-usable.
- **AD-23**: Strategy dispatch is by `isinstance(strategy, ForecastOptimizer)`; `run_benchmark()` takes explicit `historical_returns`/`evaluation_returns` arguments (no internal split inference); `allocate()` runs exactly once per call, weights held static across the evaluated window.
- **AD-24**: `BenchmarkResult`'s minimum guaranteed field set is fixed (mirrors AD-8): `strategy_name`, `strategy_parameters`, `metrics`, `asset_scenario_ids`, `time_grid_reference`, `library_version`, `generated_at`.
- **AD-25**: Benchmark layer is JAX-native by default (Metrics, returns, Runner, interface types, `EqualWeight`, GMV's unconstrained path); the Optimizer Solver Layer's `scipy` dependency is the sole, deliberately bounded, swappable exception — `qpax`/`linrax`/`MPAX` named as future JAX-native replacement candidates, not yet mature enough for v1.
- **AD-26** *(added 2026-07-03)*: `EvaluationResult` is a distinct, plain, JSON-native dataclass derived from `BenchmarkResult` via one pure function, `to_evaluation_result()` — `BenchmarkResult` (AD-17, AD-24) remains the sole runtime representation and is unchanged; `EvaluationResult` reshapes `metrics` into a list of `{name, value}` records (vs. `BenchmarkResult`'s flat dict) to match the Hugging Face `model-index` convention, and is the sole type local storage, HF publishing, and Leaderboard aggregation consume.

**Deferred in Architecture (not yet covered by any story — flagged so they aren't silently assumed done):** release/publish operational envelope (PyPI release process, HF Hub publish trigger/auth), runtime/compute environment (CPU/GPU/TPU), dataset generation/hosting cost at scale, Parquet row granularity, dataset versioning scheme specifics, Hugging Face org/namespace convention, conformance test harness mechanism (pytest vs. property-based), rBergomi statistical test suite specifics, open-source license choice. **Added 2026-07-02:** Portfolio Optimizer conformance test harness mechanism; real forecasting-model integrations (PatchTST, iTransformer, TimeMixer); short positions/non-long-only constraints; distributional/quantile forecasts; `PortfolioWeights`/solver error-type hierarchy; JAX-native replacement for the Optimizer Solver Layer (`qpax`/`linrax`/`MPAX`, revisit as they mature). **Added 2026-07-03:** hosted Leaderboard web UI/Space (PRD Feature 4.10 — `EvaluationResult` publishing and Leaderboard aggregation are now designed and covered by Epic 7; only the hosted UI itself remains deferred); Evaluation Results repo layout/namespace convention; Evaluation Results external/community submission and write-access model; Evaluation Results "verified" reproduction workflow.

### UX Design Requirements

Not applicable — no UI, no UX design document exists for this project.

## FR Coverage Map

| FR | Epic | Brief description |
|----|------|-------------------|
| FR-1 | Epic 1 | `simulate()` call signature, identical across all models |
| FR-2 | Epic 1 | Stable `Scenario` schema (`observation`, `latent_state`, `metadata`) |
| FR-3 | Epic 1 | Explicit `TimeGrid` object, non-uniform spacing supported |
| FR-4 | Epic 1 | Reproducibility via seed + `Metadata` provenance |
| FR-5 | Epic 1 | Opt-in `Randomness` materialization (separate path) |
| FR-6 | Epic 1 | Soft parameter validation (warning, never exception) |
| FR-7 | Epic 1 | `BlackScholes` model, correctness vs. closed-form reference |
| FR-8 | Epic 2 | `Heston` model, variance as `latent_state`, semi-closed-form correctness |
| FR-9 | Epic 2 | `RoughBergomi` model, volatility as `latent_state`, statistical correctness |
| FR-10 | Epic 1 | Public `MarketModel` ABC — zero changes to core on new model |
| FR-11 | Epic 1 | Conformance test suite + test-only dummy model proving interface independence |
| FR-12 | Epic 3 | Parquet export, columns mirror `Scenario` schema |
| FR-13 | Epic 3 | Per-model Hugging Face dataset, shared top-level schema |
| FR-14 | Epic 3 | Independent dataset versioning (decoupled from library version) |
| FR-15 | Epic 3 | Dataset card with all 6 required fields |
| FR-23 | Epic 4 | Public `BaselineStrategy` interface |
| FR-24 | Epic 4 | Public `ForecastOptimizer` interface (point forecast) |
| FR-25 | Epic 4 | Portfolio Optimizer conformance suite + dummy `ForecastOptimizer` |
| FR-16 | Epic 4 | Sharpe Ratio metric |
| FR-17 | Epic 4 | Sortino Ratio metric |
| FR-18 | Epic 4 | Maximum Drawdown metric |
| FR-19 | Epic 4 | Final Wealth Factor metric |
| FR-20 | Epic 5 | Equal Weight baseline |
| FR-21 | Epic 5 | Global Minimum Variance baseline (long-only) |
| FR-22 | Epic 5 | CVaR Optimization baseline (95% confidence default) |
| FR-26 | Epic 6 | Multi-asset composition from single-asset Scenarios, `TimeGrid`-aligned |
| FR-27 | Epic 6 | Benchmark Runner orchestration, static buy-and-hold allocation |
| FR-28 | Epic 6 | Return-series derivation from `Scenario.observation` |
| FR-29 | Epic 6 | JSON-serializable `BenchmarkResult` with fixed minimum fields |
| FR-30 | Epic 7 | `EvaluationResult` schema, derived from `BenchmarkResult` |
| FR-31 | Epic 7 | `BenchmarkResult` → `EvaluationResult` pure transform |
| FR-32 | Epic 7 | Local Evaluation Results file storage (timestamped, append-only) |
| FR-33 | Epic 7 | Hugging Face Evaluation Results publishing (shared repo + card) |
| FR-34 | Epic 7 | Leaderboard aggregation (generic reader + ranked table) |

## Epic List

### Epic 1: Package Foundation, Core Simulation API & Black-Scholes
A researcher can install the package, call `simulate(model=BlackScholes(...), time_grid=..., n_paths=..., seed=...)`, and receive a reproducible, correctly-shaped `Scenario`. The `MarketModel` interface is documented and a conformance test suite (including a test-only dummy model) proves the interface contract holds independently of any real financial model.
**FRs covered:** FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, FR-10, FR-11
**NFRs covered:** NFR-1, NFR-2, NFR-3 (Black-Scholes correctness check), NFR-4, NFR-5

### Epic 2: Heston & Rough Bergomi Models (Complete v1 Model Zoo)
A researcher can simulate with all three v1 Market Models — comparing behavior across closed-form (Black-Scholes), semi-closed-form (Heston), and no-closed-form (rBergomi) stochastic processes — each validated against correctness checks appropriate to its mathematical character. This proves the State-Space Interface generalizes across mathematically diverse models without changes to `simulate()` or the export pipeline.
**FRs covered:** FR-8, FR-9
**NFRs covered:** NFR-3 (Heston + rBergomi correctness checks)

### Epic 3: Benchmark Dataset Export & Publishing
A researcher can load a versioned QuantScenarioBench dataset directly from the Hugging Face Hub (via `datasets.load_dataset(...)`) without running any simulation code locally. The export pipeline is generic over the `Scenario` schema and publishes independently versioned, documented datasets for all three v1 Market Models.
**FRs covered:** FR-12, FR-13, FR-14, FR-15
**NFRs covered:** (AD-2, AD-5, AD-8, AD-11 govern this epic)

### Epic 4: Portfolio Optimizer Interface & Performance Metrics (Benchmark Foundation)
A contributor can implement a `BaselineStrategy` or `ForecastOptimizer` against a documented, `equinox.Module`-based interface, and any Portfolio Return series can be scored with the four v1 Metrics (Sharpe, Sortino, Maximum Drawdown, Final Wealth Factor). A test-only dummy `ForecastOptimizer` proves the interface holds independent of any real strategy — the direct analogue of Epic 1's State-Space Interface + conformance suite, this time for the benchmark layer.
**FRs covered:** FR-23, FR-24, FR-25, FR-16, FR-17, FR-18, FR-19
**NFRs covered:** NFR-2 (extended), NFR-3 (extended), NFR-4 (extended), NFR-6

### Epic 5: Traditional Baseline Strategies
A researcher can allocate a portfolio using three standardized, non-learned baselines — Equal Weight, Global Minimum Variance, and CVaR Optimization — each implementing the Epic 4 `BaselineStrategy` interface, giving every future forecasting/optimization model a concrete, reproducible bar to clear.
**FRs covered:** FR-20, FR-21, FR-22
**NFRs covered:** NFR-3 (extended)

### Epic 6: Benchmark Runner & Results
A researcher can assemble a multi-asset portfolio from several generated or HF-loaded Scenarios, run any Epic 5 baseline (or a custom Epic 4-conforming strategy) through `run_benchmark()`, and receive a structured, JSON-serializable `BenchmarkResult` — without writing any backtest plumbing. This is the feature that realizes UJ-4 and UJ-5 end-to-end.
**FRs covered:** FR-26, FR-27, FR-28, FR-29
**NFRs covered:** NFR-6

### Epic 7: Evaluation Results & Leaderboard
A researcher can take a `BenchmarkResult` from Epic 6, publish it as a versioned Evaluation Result to a shared Hugging Face dataset repo, and see it appear — ranked against every other published strategy/Benchmark Dataset combination — in an aggregated Leaderboard table, without writing any publishing, aggregation, or ranking code. This is the feature that realizes UJ-6 end-to-end. A hosted Leaderboard web UI (a Hugging Face Space or equivalent, PRD Feature 4.10) is an explicit future phase and is not part of this epic.
**FRs covered:** FR-30, FR-31, FR-32, FR-33, FR-34
**NFRs covered:** NFR-7

---

## Epic 1: Package Foundation, Core Simulation API & Black-Scholes

**Goal:** A researcher can install the package, call `simulate(model=BlackScholes(...), time_grid=..., n_paths=..., seed=...)`, and receive a reproducible, correctly-shaped `Scenario`. The `MarketModel` interface is documented and a conformance test suite (including a test-only dummy model) proves the interface contract holds independently of any real financial model.

**FRs covered:** FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, FR-10, FR-11
**NFRs covered:** NFR-1, NFR-2, NFR-3 (Black-Scholes correctness check), NFR-4, NFR-5

---

### Story 1.1: Package Scaffold & Development Environment

As a developer working on QuantScenarioBench,
I want a properly structured, installable Python package with pinned dependencies and x64 precision enabled,
So that the entire codebase starts from a consistent foundation that enforces the Architecture Spine's package layout and precision policy from the first line of code.

**Acceptance Criteria:**

**Given** the repository is cloned,
**When** `pip install -e ".[dev]"` is run,
**Then** the package installs without errors on Python >=3.11.

**Given** the package is installed,
**When** `import quantscenariobench` is executed,
**Then** `jax.config.x64_enabled` is `True` (x64 enabled per AD-7).

**Given** the package is installed,
**When** `import quantscenariobench` is executed,
**Then** the six sub-packages (`interface`, `models`, `solver`, `api`, `export`, `testing`) are all importable without errors.

**Given** `pyproject.toml`,
**When** inspected,
**Then** it specifies `python_requires = ">=3.11"`, `jax>=0.4.38`, `diffrax==0.7.2`, and `equinox>=0.11.10` as dependencies.

**Given** the entire codebase,
**When** searched for `jax_enable_x64`,
**Then** it appears exactly once, in `quantscenariobench/__init__.py`, and nowhere else.

---

### Story 1.2: State-Space Interface Core Types (MarketModel, Scenario, TimeGrid)

As a developer implementing a new Market Model,
I want a documented `MarketModel` abstract base class, a stable `Scenario` type, and an explicit `TimeGrid` type to implement against,
So that I know exactly what my model must produce and can verify conformance before integrating with the simulation core.

**Acceptance Criteria:**

**Given** `quantscenariobench.interface`,
**When** inspected,
**Then** it exports `MarketModel`, `Scenario`, `Metadata`, and `TimeGrid` as its public surface.

**Given** `MarketModel`,
**When** instantiated directly (without subclassing),
**Then** a `TypeError` is raised — it cannot be instantiated as an abstract class.

**Given** `MarketModel`,
**When** inspected,
**Then** it is a subclass of `equinox.Module` with abstract methods `_drift(self, t, state) -> PyTree` and `_diffusion(self, t, state) -> PyTree` enforced at construction time (AD-1, AD-4).

**Given** a concrete `MarketModel` subclass that omits `_drift` or `_diffusion`,
**When** instantiated,
**Then** an error is raised indicating the abstract method is not implemented.

**Given** `Scenario` constructed with `observation` and `latent_state` arrays and a `Metadata` instance,
**When** pytree-flattened via `jax.tree_util.tree_leaves(scenario)`,
**Then** `observation` and `latent_state` appear as leaves, and `metadata` does not — it is pytree aux_data only (AD-2).

**Given** `Metadata`,
**When** constructed,
**Then** it carries exactly these fields and no others: `seed`, `prng_key_info`, `model_name`, `model_version`, `parameters`, `time_grid`, `n_paths`, `library_version`, `dataset_version`, `generated_at` (AD-8).

**Given** `TimeGrid` constructed from a non-monotonic array,
**When** instantiated,
**Then** a `ValueError` is raised.

**Given** `TimeGrid` constructed from a non-uniform array (irregular spacing),
**When** instantiated,
**Then** it is a valid `TimeGrid` — non-uniform grids are supported (FR-3, AD-12).

**Given** any module in `quantscenariobench.models`, `quantscenariobench.solver`, `quantscenariobench.api`, `quantscenariobench.export`, or `quantscenariobench.testing`,
**When** its source is inspected for cross-module imports,
**Then** none imports from any sibling sub-package — each may only import from `quantscenariobench.interface` and approved third-party dependencies (AD-9).

---

### Story 1.3: Solver Layer (diffrax Integration)

As the simulation core,
I want a Solver Layer that translates any conforming `MarketModel`'s drift and diffusion into a simulated path via diffrax,
So that `simulate()` can remain model-agnostic and no Market Model ever needs to know diffrax exists.

**Acceptance Criteria:**

**Given** every Python module in the `quantscenariobench` package,
**When** searched for `import diffrax`,
**Then** the import appears only inside `quantscenariobench.solver` (AD-4, AD-9).

**Given** the solver's internal `solve_sde(model, time_grid, n_paths, key)` function called with a minimal conforming `MarketModel`, a `TimeGrid`, an integer `n_paths`, and a JAX PRNG key,
**When** executed,
**Then** it returns path arrays with a leading `n_paths` axis and one value per `TimeGrid` time point.

**Given** `solve_sde` called twice with identical arguments on the same backend,
**When** results are compared,
**Then** the returned path arrays are bit-identical (NFR-1, FR-4).

**Given** `solve_sde` called with the default (no `return_randomness`) path,
**When** profiled for memory allocation,
**Then** no full Brownian noise array is materialised — `diffrax.VirtualBrownianTree` is used and raw noise is not stored (AD-3).

**Given** a `TimeGrid` with non-uniform spacing passed to `solve_sde`,
**When** the output path is inspected,
**Then** the time axis of the returned arrays corresponds exactly to the explicit time points in the `TimeGrid`, not to a uniform re-sampling (AD-12, FR-3).

**Given** `solve_sde` called with `return_randomness=True`,
**When** compared to the default path,
**Then** it uses a separate, explicit construction path (not a conditional branch inside the default path) and returns Brownian increments alongside the simulated path (AD-3, FR-5).

---

### Story 1.4: `simulate()` Public Orchestrator

As a quantitative researcher,
I want to call `simulate(model, time_grid, n_paths, seed)` and receive a `Scenario` with reproducible `observation`, `latent_state`, and `metadata`,
So that I have a single, model-agnostic entry point that works identically regardless of which stochastic process I choose.

**Acceptance Criteria:**

**Given** `quantscenariobench.api.simulate` called with any conforming `MarketModel`, a `TimeGrid`, `n_paths`, and an integer `seed`,
**When** executed,
**Then** it returns a `Scenario` with exactly three top-level attributes: `observation`, `latent_state`, and `metadata` (FR-1, FR-2).

**Given** `simulate()` source code,
**When** inspected,
**Then** it contains no Market Model-specific branching — all model behaviour is dispatched through `_drift` and `_diffusion` on the model itself (FR-1, AD-4).

**Given** `simulate()` called twice with identical `model`, `time_grid`, `n_paths`, and `seed` on the same backend,
**When** the returned `Scenario` objects are compared,
**Then** `observation` and `latent_state` arrays are bit-identical (FR-4, NFR-1).

**Given** `simulate()` called with a `MarketModel` that has no latent process,
**When** the returned `Scenario` is inspected,
**Then** `latent_state` is present as an explicitly empty structure — never absent (FR-2).

**Given** `simulate()` called with `return_randomness=False` (the default),
**When** the returned `Scenario` is inspected,
**Then** no raw random draws are materialised or returned.

**Given** `simulate()` called with `return_randomness=True`,
**When** the returned `Randomness` is used to replay the simulation,
**Then** it reproduces the identical `observation` and `latent_state` arrays deterministically (FR-5).

**Given** `quantscenariobench.api` source code,
**When** inspected,
**Then** it imports from `quantscenariobench.interface` and `quantscenariobench.solver` only — never from `quantscenariobench.models` directly (AD-9).

**Given** `Scenario.metadata` on any returned Scenario,
**When** inspected,
**Then** all ten fields are present: `seed`, `prng_key_info`, `model_name`, `model_version`, `parameters`, `time_grid`, `n_paths`, `library_version`, `dataset_version`, `generated_at` (AD-8, FR-4).

---

### Story 1.5: Black-Scholes Market Model with Closed-Form Correctness Validation

As a quantitative researcher,
I want to simulate geometric Brownian motion paths using a `BlackScholes` model that conforms to the State-Space Interface,
So that I have a first working, numerically-validated end-to-end simulation and can verify the framework produces correct paths against the known closed-form Black-Scholes solution.

**Acceptance Criteria:**

**Given** `quantscenariobench.models.BlackScholes` instantiated with parameters `(mu, sigma, S0)`,
**When** inspected,
**Then** it is an `equinox.Module` subclass of `MarketModel` (AD-1, AD-6).

**Given** `BlackScholes._drift(t, state)` and `BlackScholes._diffusion(t, state)`,
**When** inspected,
**Then** they implement geometric Brownian motion dynamics (drift `mu * S`, diffusion `sigma * S`) correctly in JAX with no Python-level loops (NFR-2).

**Given** `simulate(model=BlackScholes(...), time_grid=..., n_paths=N, seed=42)`,
**When** the returned `Scenario` is inspected,
**Then** `latent_state` is an explicitly empty structure — Black-Scholes has no separate latent process (FR-7).

**Given** `simulate(model=BlackScholes(mu=mu, sigma=sigma, S0=S0), time_grid=TimeGrid([0, T]), n_paths=10_000, seed=0)`,
**When** `mean(log(S_T / S_0))` is computed over paths,
**Then** it is within 3 standard errors of the closed-form value `(mu - 0.5 * sigma**2) * T` (FR-7, NFR-3, AD-10).

**Given** the same simulation,
**When** `std(log(S_T / S_0))` is computed over paths,
**Then** it is within 3 standard errors of the closed-form value `sigma * sqrt(T)` (FR-7, NFR-3, AD-10).

**Given** `quantscenariobench.models.BlackScholes` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.interface` and `equinox` — never from `quantscenariobench.solver`, `quantscenariobench.api`, or `diffrax` (AD-9).

**Given** `BlackScholes` constructed with `sigma < 0` (a research-invalid parameter),
**When** constructed,
**Then** a `QuantScenarioBenchValidationWarning` is emitted (not an exception), and subsequent `simulate()` calls still complete and return a `Scenario` (FR-6).

**Given** the entire codebase,
**When** searched for soft validation warnings,
**Then** all validation warnings use exactly `QuantScenarioBenchValidationWarning` — no bare `UserWarning` and no model-specific warning subclasses exist anywhere (FR-6 Consistency Convention).

---

### Story 1.6: State-Space Interface Conformance Suite

As a developer contributing a new Market Model,
I want a reusable conformance test suite I can run against any `MarketModel` implementation,
So that I can verify my model satisfies the State-Space Interface — including Scenario shape, reproducibility, and validation behaviour — without modifying `simulate()` or the export pipeline.

**Acceptance Criteria:**

**Given** `quantscenariobench.testing` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.interface` and test tooling — never from `quantscenariobench.models`, `quantscenariobench.solver`, `quantscenariobench.export`, or `diffrax` (AD-9).

**Given** the conformance test suite run against `BlackScholes`,
**When** executed,
**Then** all conformance tests pass.

**Given** the conformance test suite run against the test-only dummy `MarketModel` defined inside `quantscenariobench.testing`,
**When** executed,
**Then** all conformance tests pass with zero changes to `simulate()`, `Scenario`, or the export pipeline source (FR-10, FR-11).

**Given** the test-only dummy `MarketModel`,
**When** the public package API is inspected,
**Then** it does not appear in `quantscenariobench.models` and is not exported from any non-testing module — it exists only inside `quantscenariobench.testing` (FR-11).

**Given** the conformance suite's reproducibility test running `simulate()` twice with the same arguments on the same backend,
**When** compared,
**Then** `observation` and `latent_state` arrays are bit-identical (FR-4, NFR-1).

**Given** the conformance suite's Scenario shape test,
**When** `simulate()` is called with any conforming model,
**Then** the returned `Scenario` always has `observation`, `latent_state` (possibly empty but never absent), and `metadata` as top-level attributes (FR-2).

**Given** the conformance suite's validation behaviour test when a model exposes a declared research constraint,
**When** the model is constructed with a parameter that violates the constraint,
**Then** a `QuantScenarioBenchValidationWarning` is emitted and no exception is raised (FR-6).

---

## Epic 2: Heston & Rough Bergomi Models (Complete v1 Model Zoo)

**Goal:** A researcher can simulate with all three v1 Market Models — comparing behavior across closed-form (Black-Scholes), semi-closed-form (Heston), and no-closed-form (rBergomi) stochastic processes — each validated against correctness checks appropriate to its mathematical character. This proves the State-Space Interface generalises across mathematically diverse models without changes to `simulate()` or the export pipeline.

**FRs covered:** FR-8, FR-9
**NFRs covered:** NFR-3 (Heston + rBergomi correctness checks)

---

### Story 2.1: Heston Stochastic Volatility Model with Semi-Closed-Form Correctness Validation

As a quantitative researcher,
I want to simulate correlated asset price and variance paths using a `Heston` model that conforms to the State-Space Interface,
So that I can compare stochastic volatility dynamics against Black-Scholes using exactly the same `simulate()` call and `Scenario` structure.

**Acceptance Criteria:**

**Given** `quantscenariobench.models.Heston` instantiated with parameters `(mu, kappa, theta, xi, rho, v0, S0)`,
**When** inspected,
**Then** it is an `equinox.Module` subclass of `MarketModel` (AD-1, AD-6).

**Given** `Heston._drift(t, state)` and `Heston._diffusion(t, state)`,
**When** inspected,
**Then** they implement the Heston SDE dynamics correctly in JAX — `state` encodes both the asset price and the variance process, and the correlation `rho` is applied between the two Brownian drivers (NFR-2, FR-8).

**Given** `simulate(model=Heston(...), time_grid=..., n_paths=N, seed=42)`,
**When** the returned `Scenario` is inspected,
**Then** `latent_state` contains the variance process paths with a shape consistent with `(n_paths, len(time_grid))` or equivalent (FR-8).

**Given** `simulate(model=Heston(...), ...)` and `simulate(model=BlackScholes(...), ...)`,
**When** the returned `Scenario` objects' top-level attributes are compared,
**Then** both have identical attribute names (`observation`, `latent_state`, `metadata`) and neither call modifies `simulate()`'s source (FR-2, FR-8, FR-10).

**Given** `simulate(model=Heston(...), ...)` with `N=10_000` paths and a liquid European call option configuration,
**When** the option price is estimated via Monte Carlo averaging over paths,
**Then** the estimate is within a defined numerical tolerance of the independently-implemented Heston semi-closed-form (characteristic-function-based) price — reference implemented within the package, not borrowed from a third-party quant library (FR-8, NFR-3, AD-10).

**Given** `Heston` constructed with Feller-violating parameters (`2 * kappa * theta < xi**2`),
**When** constructed,
**Then** a `QuantScenarioBenchValidationWarning` is emitted, no exception is raised, and a subsequent `simulate()` call completes and returns a valid `Scenario` (FR-6, FR-8).

**Given** the conformance test suite from Story 1.6 run against `Heston`,
**When** executed,
**Then** all conformance tests pass with zero changes to `simulate()`, `Scenario`, or the export pipeline source (FR-10, FR-11).

**Given** `quantscenariobench.models.Heston` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.interface` and `equinox` (AD-9).

---

### Story 2.2: Rough Bergomi Model with Statistical Correctness Validation

As a quantitative researcher,
I want to simulate rough volatility paths using a `RoughBergomi` model that conforms to the State-Space Interface,
So that I can compare non-Markovian, rough-volatility dynamics against Heston and Black-Scholes using the same `simulate()` call — even though rBergomi has no closed-form pricing reference.

**Acceptance Criteria:**

**Given** `quantscenariobench.models.RoughBergomi` instantiated with parameters `(H, eta, rho, xi0, S0)` (Hurst exponent, vol-of-vol, correlation, initial variance level, initial spot),
**When** inspected,
**Then** it is an `equinox.Module` subclass of `MarketModel` (AD-1, AD-6).

**Given** `RoughBergomi._drift(t, state)` and `RoughBergomi._diffusion(t, state)`,
**When** inspected,
**Then** they implement rBergomi dynamics in JAX — the volatility process is driven by fractional Brownian motion with Hurst exponent `H`, using a discretised Volterra representation or equivalent (NFR-2, FR-9).

**Given** `simulate(model=RoughBergomi(...), time_grid=..., n_paths=N, seed=42)`,
**When** the returned `Scenario` is inspected,
**Then** `latent_state` contains the rough volatility process paths with a shape consistent with `(n_paths, len(time_grid))` or equivalent (FR-9).

**Given** `simulate(model=RoughBergomi(...), ...)` and `simulate(model=BlackScholes(...), ...)`,
**When** the returned `Scenario` objects' top-level attributes are compared,
**Then** both have identical attribute names (`observation`, `latent_state`, `metadata`) and neither call modifies `simulate()`'s source (FR-2, FR-9, FR-10).

**Given** `simulate(model=RoughBergomi(H=0.5, ...), ...)` with `N=10_000` paths (H=0.5 is the Markovian boundary),
**When** the distribution of `log(S_T / S0)` is inspected,
**Then** the mean and variance are consistent with the expected Markovian limiting case within 3 standard errors (FR-9, NFR-3, AD-10).

**Given** `simulate(model=RoughBergomi(...), ...)` for a range of `H` values (`H < 0.5`),
**When** the short-maturity implied volatility skew steepness is measured across `H` values,
**Then** smaller `H` (rougher volatility) produces steeper skew — consistent with the known qualitative behavior of rough volatility models (FR-9, NFR-3, AD-10).

**Given** the conformance test suite from Story 1.6 run against `RoughBergomi`,
**When** executed,
**Then** all conformance tests pass with zero changes to `simulate()`, `Scenario`, or the export pipeline source (FR-10, FR-11).

**Given** `quantscenariobench.models.RoughBergomi` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.interface` and `equinox` (AD-9).

---

## Epic 3: Benchmark Dataset Export & Publishing

**Goal:** A researcher can load a versioned QuantScenarioBench dataset directly from the Hugging Face Hub via `datasets.load_dataset(...)` without running any simulation code locally. The export pipeline is generic over the `Scenario` schema and publishes independently versioned, documented datasets for all three v1 Market Models.

**FRs covered:** FR-12, FR-13, FR-14, FR-15
**NFRs covered:** (AD-2, AD-5, AD-8, AD-11 govern this epic)

---

### Story 3.1: Parquet Export of Scenario Batches

As a quantitative researcher,
I want to export a batch of Scenarios to Parquet files with columns that mirror the Scenario schema,
So that I can persist simulation results in a portable, standard format that downstream tools (pandas, Polars, Arrow, HuggingFace datasets) can consume without any custom deserialization logic.

**Acceptance Criteria:**

**Given** `quantscenariobench.export` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.interface` and approved third-party libraries — never from `quantscenariobench.models`, `quantscenariobench.solver`, or `quantscenariobench.testing` (AD-5, AD-9).

**Given** a list of `Scenario` objects produced by any v1 Market Model,
**When** `export_parquet(scenarios, path)` is called,
**Then** Parquet file(s) are written to `path` with columns derived by pytree-flattening `observation` and `latent_state` (dynamic fields) and including all ten Metadata fields as additional columns (AD-2, AD-5, AD-8, FR-12).

**Given** Parquet exported from a `BlackScholes` Scenario batch and from a `Heston` Scenario batch,
**When** their top-level column schemas are compared,
**Then** both share the same column names — differences are in content or shape, not in column names (FR-13).

**Given** Parquet written by `export_parquet`,
**When** read back via `pandas.read_parquet` or `pyarrow.parquet.read_table`,
**Then** the round-tripped `observation` and `latent_state` values are numerically identical to the originals (FR-12).

**Given** exported Parquet,
**When** the column list is inspected,
**Then** all ten Metadata fields appear as columns: `seed`, `model_name`, `model_version`, `parameters`, `time_grid`, `n_paths`, `library_version`, `dataset_version`, `generated_at`, `prng_key_info` (AD-8, FR-15).

---

### Story 3.2: Hugging Face Dataset Publishing & Dataset Cards

As a quantitative researcher,
I want to load a published QuantScenarioBench Benchmark Dataset directly from the Hugging Face Hub,
So that I can benchmark my model against standardised synthetic market scenarios without installing or running the simulation library locally.

**Acceptance Criteria:**

**Given** the three v1 Benchmark Datasets (Black-Scholes, Heston, rBergomi) published to the Hugging Face Hub,
**When** `datasets.load_dataset("<namespace>/<dataset-name>")` is called for any of them,
**Then** data loads successfully and conforms to the documented shared column schema from Story 3.1 (FR-13).

**Given** the three published datasets,
**When** their top-level column schemas are compared,
**Then** all three share the same column names — consistent with the Parquet schema established in Story 3.1 (FR-13).

**Given** a `dataset_version` value in any published dataset's Metadata,
**When** compared to the `library_version` field,
**Then** they are independent identifiers — bumping a dataset's generation parameters does not require a new library release, and a new library release does not force a dataset version bump (FR-14).

**Given** any published Benchmark Dataset on the Hugging Face Hub,
**When** its dataset card is inspected,
**Then** all six required fields are present: (1) column schema, (2) Market Model name and parameter values used to generate it, (3) `TimeGrid` and `n_paths` used, (4) library version, (5) dataset version identifier, (6) backend-scoped reproducibility caveat (cross-backend bit-identity not guaranteed) (FR-15).

**Given** `quantscenariobench.export` source code in the publishing path,
**When** inspected,
**Then** it imports only from `quantscenariobench.interface` — never from `quantscenariobench.models` — confirming the export pipeline is generic over the Scenario schema and agnostic of which concrete Market Model produced it (AD-5, FR-10).

---

## Epic 4: Portfolio Optimizer Interface & Performance Metrics (Benchmark Foundation)

**Goal:** A contributor can implement a `BaselineStrategy` or `ForecastOptimizer` against a documented, `equinox.Module`-based interface, and any Portfolio Return series can be scored with the four v1 Metrics. A test-only dummy `ForecastOptimizer` proves the interface holds independent of any real strategy — the benchmark layer's direct analogue of Epic 1's State-Space Interface + conformance suite.

**FRs covered:** FR-23, FR-24, FR-25, FR-16, FR-17, FR-18, FR-19
**NFRs covered:** NFR-2 (extended), NFR-3 (extended), NFR-4 (extended), NFR-6

---

### Story 4.1: Portfolio Optimizer Interface Core Types (`BaselineStrategy`, `ForecastOptimizer`, `PortfolioWeights`, `BenchmarkResult`)

As a developer implementing a new portfolio strategy,
I want a documented `BaselineStrategy` ABC, a `ForecastOptimizer` ABC, a validated `PortfolioWeights` type, and a `BenchmarkResult` type,
So that I know exactly what my strategy must produce, and consumers of `PortfolioWeights`/`BenchmarkResult` can rely on a fixed shape without per-strategy special-casing.

**Acceptance Criteria:**

**Given** `quantscenariobench.benchmark.interface`,
**When** inspected,
**Then** it exports `BaselineStrategy`, `ForecastOptimizer`, `PortfolioWeights`, and `BenchmarkResult` as its public surface.

**Given** `BaselineStrategy` or `ForecastOptimizer`,
**When** instantiated directly (without subclassing),
**Then** a `TypeError` is raised — neither can be instantiated as an abstract class.

**Given** `BaselineStrategy`,
**When** inspected,
**Then** it is an `equinox.Module` ABC subclass with one abstract method, `allocate(historical_returns: Float[Array, "t n"]) -> PortfolioWeights` (AD-13).

**Given** `ForecastOptimizer`,
**When** inspected,
**Then** it is an `equinox.Module` ABC subclass with one abstract method, `allocate(historical_returns: Float[Array, "t n"], forecast: Float[Array, "n"]) -> PortfolioWeights` (AD-13, AD-21).

**Given** a `PortfolioWeights` constructed from an array whose entries do not sum to `1.0` within `1e-6`,
**When** constructed,
**Then** it raises rather than silently normalizing or returning a malformed vector (AD-20).

**Given** a `PortfolioWeights` constructed from an array containing a negative entry,
**When** constructed,
**Then** it raises — non-negativity (long-only) is a v1-universal, type-level invariant, not a per-strategy convention (AD-20).

**Given** a `PortfolioWeights` constructed with `n` not matching the number of constituent assets in the call,
**When** constructed,
**Then** it raises (AD-20).

**Given** `BenchmarkResult`,
**When** inspected,
**Then** it is a plain immutable Python dataclass (`@dataclasses.dataclass(frozen=True)`), not an `equinox.Module`, with only JSON-native field types (`str`, `float`, `int`, `dict`, `list`) (AD-17).

**Given** any module in `quantscenariobench.benchmark.strategies`, `quantscenariobench.benchmark.metrics`, `quantscenariobench.benchmark.returns`, `quantscenariobench.benchmark.solver`, `quantscenariobench.benchmark.runner`, or `quantscenariobench.benchmark.testing`,
**When** its source is inspected for cross-module imports,
**Then** none imports `quantscenariobench.models` or `quantscenariobench.solver` (the scenario-generation Solver Layer) directly — the only connection to scenario generation is a `Scenario` flowing in via `quantscenariobench.interface` (AD-19).

---

### Story 4.2: Portfolio Optimizer Conformance Suite (incl. Test-Only Dummy `ForecastOptimizer`)

As a contributor implementing a new `BaselineStrategy` or `ForecastOptimizer`,
I want a reusable conformance test suite,
So that I can verify my strategy satisfies the Portfolio Optimizer Interface — including `PortfolioWeights` validity and deterministic output — before it is ever run through the Benchmark Runner.

**Acceptance Criteria:**

**Given** `quantscenariobench.benchmark.testing` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.benchmark.interface` and test tooling — never from `quantscenariobench.benchmark.strategies` or `quantscenariobench.benchmark.runner` (AD-19).

**Given** the test-only dummy `ForecastOptimizer` defined inside `quantscenariobench.benchmark.testing`,
**When** its `allocate(historical_returns, forecast)` method is called with valid inputs,
**Then** it returns a `PortfolioWeights` satisfying all of AD-20's invariants (FR-25).

**Given** the dummy `ForecastOptimizer`,
**When** the public package API is inspected,
**Then** it does not appear in `quantscenariobench.benchmark.strategies` and is not exported from any non-testing module — it exists only inside `quantscenariobench.benchmark.testing` (FR-25, mirrors FR-11's dummy Market Model treatment).

**Given** the conformance suite's ABC-enforcement test,
**When** a class subclasses `BaselineStrategy` or `ForecastOptimizer` without implementing `allocate`,
**Then** instantiation raises (AD-13).

**Given** the conformance suite's determinism test,
**When** any conforming strategy's `allocate()` is called twice with identical arguments,
**Then** it returns bit-identical `PortfolioWeights` both times.

**Given** the conformance suite's `PortfolioWeights` shape test,
**When** any conforming strategy's `allocate()` is called for an N-asset portfolio,
**Then** the returned `PortfolioWeights` has shape `(N,)`.

---

### Story 4.3: Portfolio Performance Metrics (Sharpe, Sortino, Maximum Drawdown, Final Wealth Factor)

As a researcher evaluating any portfolio strategy,
I want four standardized Metrics computed from a Portfolio Return series,
So that I can compare strategies on a common, reproducible basis regardless of which strategy produced the returns.

**Acceptance Criteria:**

**Given** `quantscenariobench.benchmark.metrics` source code,
**When** inspected,
**Then** every `MetricFn` is written entirely in `jax.numpy`, is `jit`-compatible, and never calls `scipy` or plain NumPy (AD-18, AD-25).

**Given** a hand-derived reference Portfolio Return series,
**When** Sharpe Ratio is computed (risk-free rate 0, no annualization),
**Then** it matches the reference value within floating-point tolerance (FR-16, NFR-3 extended, AD-10 amended).

**Given** a constant (zero-variance) Portfolio Return series,
**When** Sharpe Ratio is computed,
**Then** it returns `0.0` rather than raising or returning `NaN`/`inf` (FR-16, AD-18).

**Given** a hand-derived reference Portfolio Return series,
**When** Sortino Ratio is computed (risk-free rate 0, no annualization),
**Then** it matches the reference value within floating-point tolerance (FR-17, NFR-3 extended).

**Given** a Portfolio Return series with no negative returns,
**When** Sortino Ratio is computed,
**Then** it returns `0.0` rather than raising (FR-17, AD-18).

**Given** a hand-derived reference Portfolio Return series,
**When** Maximum Drawdown is computed,
**Then** it matches the reference value within floating-point tolerance (FR-18, NFR-3 extended).

**Given** a hand-derived reference Portfolio Return series,
**When** Final Wealth Factor is computed,
**Then** it matches the reference value within floating-point tolerance (FR-19, NFR-3 extended).

**Given** every `MetricFn`,
**When** inspected,
**Then** it carries a `.name: str` attribute and matches the fixed signature `Callable[[Float[Array, "t"]], float]` (AD-18).

**Given** the four v1 `MetricFn`s assembled into a registry passed to `run_benchmark()`,
**When** two entries share a `.name`,
**Then** the call raises rather than silently allowing one to shadow the other (AD-18).

**Given** the entire codebase's Metrics/Baselines correctness tests,
**When** inspected for their reference-value source,
**Then** none imports its expected value from `empyrical`, `quantstats`, `PyPortfolioOpt`, `Riskfolio-Lib`, or any other portfolio-analytics library — every reference is hand-derived or independently implemented (AD-10 amended).

---

## Epic 5: Traditional Baseline Strategies

**Goal:** A researcher can allocate a portfolio using three standardized, non-learned baselines — Equal Weight, Global Minimum Variance, and CVaR Optimization — each implementing the Epic 4 `BaselineStrategy` interface, giving every future forecasting/optimization model a concrete, reproducible bar to clear.

**FRs covered:** FR-20, FR-21, FR-22
**NFRs covered:** NFR-3 (extended)

---

### Story 5.1: Equal Weight Baseline

As a researcher benchmarking a portfolio,
I want an `EqualWeight` `BaselineStrategy` that allocates equally across all assets,
So that I have the simplest possible standardized comparison anchor, requiring no historical-data fitting.

**Acceptance Criteria:**

**Given** `quantscenariobench.benchmark.strategies.EqualWeight` instantiated,
**When** inspected,
**Then** it is an `equinox.Module` subclass of `BaselineStrategy` (AD-13, AD-6).

**Given** `EqualWeight.allocate(historical_returns)` called for an N-asset portfolio,
**When** executed,
**Then** every returned `PortfolioWeight` equals `1/N`, regardless of `historical_returns`' content (FR-20).

**Given** `EqualWeight`'s implementation,
**When** inspected,
**Then** it is written entirely in `jax.numpy` and never calls `quantscenariobench.benchmark.solver` (AD-25).

**Given** `quantscenariobench.benchmark.strategies.EqualWeight` source code,
**When** inspected,
**Then** it imports only from `quantscenariobench.benchmark.interface` and `equinox` (AD-19).

**Given** the Story 4.2 conformance suite run against `EqualWeight`,
**When** executed,
**Then** all conformance tests pass (FR-23, FR-25 cross-check).

---

### Story 5.2: Global Minimum Variance Baseline & the Optimizer Solver Layer

As a researcher benchmarking a portfolio,
I want a `GlobalMinimumVariance` `BaselineStrategy` that minimizes portfolio variance — via a closed-form path when unconstrained, and via a solver when long-only-constrained,
So that I have a standard risk-minimizing comparison anchor, with the project's first (deliberately bounded) non-JAX-native dependency.

**Acceptance Criteria:**

**Given** `quantscenariobench.benchmark.solver` source code,
**When** inspected,
**Then** it is the only benchmark-layer module that imports `scipy`, and it is responsible for converting a JAX array to NumPy on the way in and back to JAX on the way out (AD-14).

**Given** `GlobalMinimumVariance(long_only=False).allocate(historical_returns)`,
**When** executed,
**Then** it computes weights via `jax.numpy.linalg` (closed-form covariance inversion) with no call into `quantscenariobench.benchmark.solver` (AD-14, AD-25).

**Given** `GlobalMinimumVariance(long_only=True).allocate(historical_returns)`,
**When** executed,
**Then** it calls `quantscenariobench.benchmark.solver.solve_allocation(...)` (`scipy.optimize.minimize`, SLSQP) and returns a `PortfolioWeights` satisfying AD-20's non-negativity invariant (AD-14).

**Given** `GlobalMinimumVariance` source code,
**When** inspected,
**Then** it never imports `scipy` directly — only `quantscenariobench.benchmark.solver` (AD-14).

**Given** a `GlobalMinimumVariance` allocation on a given `historical_returns`,
**When** its portfolio variance is compared to `EqualWeight`'s on the same data,
**Then** `GlobalMinimumVariance`'s variance is no greater (FR-21 sanity property).

**Given** a `solve_allocation(...)` call that fails to converge,
**When** invoked from `GlobalMinimumVariance(long_only=True)`,
**Then** it raises a `QuantScenarioBenchSolverError` rather than returning a degenerate or unconverged weight vector (AD-14).

**Given** the Story 4.2 conformance suite run against both `GlobalMinimumVariance(long_only=True)` and `GlobalMinimumVariance(long_only=False)`,
**When** executed,
**Then** all conformance tests pass (FR-23, FR-25 cross-check).

---

### Story 5.3: CVaR Optimization Baseline

As a researcher benchmarking a portfolio,
I want a `CVaROptimization` `BaselineStrategy` that minimizes Conditional Value-at-Risk at a required, recorded confidence level,
So that I have a tail-risk-aware standardized comparison anchor with a reproducible, explicit confidence level.

**Acceptance Criteria:**

**Given** `CVaROptimization` constructed without a `confidence_level` argument,
**When** constructed,
**Then** it raises — `confidence_level` is a required constructor argument, never an internal hardcoded default (AD-15).

**Given** `CVaROptimization(confidence_level=0.95)` (the confirmed v1 default) `.allocate(historical_returns)`,
**When** executed,
**Then** it calls `quantscenariobench.benchmark.solver.solve_allocation(...)` (`scipy.optimize.linprog`, Rockafellar–Uryasev formulation) and returns a `PortfolioWeights` satisfying AD-20's invariants (FR-22, AD-14).

**Given** `CVaROptimization` source code,
**When** inspected,
**Then** it never imports `scipy` directly (AD-14).

**Given** a constructed `CVaROptimization(confidence_level=0.95)`,
**When** its recorded parameters are inspected,
**Then** `confidence_level=0.95` is present in the strategy's identity/parameters, so a later `BenchmarkResult` stays reproducible (FR-22, AD-15).

**Given** the correctness test for `CVaROptimization`,
**When** its weights are computed against a hand-derived reference CVaR-minimizing allocation,
**Then** they match within tolerance — not borrowed from a bundled portfolio-analytics library (AD-10 amended).

**Given** the Story 4.2 conformance suite run against `CVaROptimization`,
**When** executed,
**Then** all conformance tests pass (FR-23, FR-25 cross-check).

---

## Epic 6: Benchmark Runner & Results

**Goal:** A researcher can assemble a multi-asset portfolio from several generated or HF-loaded Scenarios, run any Epic 5 baseline (or a custom Epic 4-conforming strategy) through `run_benchmark()`, and receive a structured, JSON-serializable `BenchmarkResult` — without writing any backtest plumbing. This is the epic that realizes UJ-4 and UJ-5 end-to-end.

**FRs covered:** FR-26, FR-27, FR-28, FR-29
**NFRs covered:** NFR-6

---

### Story 6.1: Multi-Asset Composition & Return-Series Derivation

As a researcher assembling a portfolio,
I want the Benchmark Runner to compose N independently generated or loaded single-asset Scenarios into one aligned, multi-asset returns matrix,
So that I can evaluate a portfolio strategy across assets without hand-writing `TimeGrid` alignment or return-conversion logic.

**Acceptance Criteria:**

**Given** N Scenarios that all share an identical `TimeGrid`,
**When** passed to the Benchmark Runner's composition step,
**Then** they assemble into one `Float[Array, "t n"]` returns matrix, with no change to `simulate()`, a Market Model, or the export/load path (FR-26).

**Given** N Scenarios where at least one has a different `TimeGrid` (different length or time points) from the others,
**When** passed to the composition step,
**Then** it raises before any return derivation is attempted — no implicit padding, truncation, or resampling (AD-22).

**Given** a single Scenario,
**When** `quantscenariobench.benchmark.returns.derive_returns(scenario)` is called,
**Then** it returns simple/arithmetic period returns computed once per `TimeGrid` step (FR-28, AD-16).

**Given** two Scenarios with identical `observation` paths,
**When** `derive_returns` is called on each,
**Then** they produce identical return series (FR-28).

**Given** a Scenario from `simulate()` and a Scenario loaded from a published Benchmark Dataset with identical `observation` paths,
**When** `derive_returns` is called on each,
**Then** they produce identical return series — the same convention applies regardless of source (FR-28).

**Given** `derive_returns`,
**When** inspected,
**Then** it is written entirely in `jax.numpy` and is `jit`-compatible (AD-16, AD-25).

**Given** a Scenario whose `observation` is not a one-dimensional, strictly-positive price series,
**When** passed to `derive_returns`,
**Then** it is rejected as not benchmark-layer-usable (AD-22).

---

### Story 6.2: Benchmark Runner Orchestration & Extensibility Proof

As a researcher,
I want to call `run_benchmark()` with any `BaselineStrategy` or `ForecastOptimizer` and get portfolio returns/metrics computed via the same pipeline regardless of strategy,
So that I can add a new strategy later without ever touching the Runner's source.

**Acceptance Criteria:**

**Given** `run_benchmark()` called with a `BaselineStrategy` and explicit `historical_returns`/`evaluation_returns` arguments,
**When** executed,
**Then** it calls `strategy.allocate(historical_returns)` exactly once and applies the resulting `PortfolioWeights` unchanged across the full `evaluation_returns` window — no intra-run rebalancing (FR-27, AD-23).

**Given** `run_benchmark()` called with a `ForecastOptimizer` and a `forecast` argument,
**When** executed,
**Then** it dispatches via `isinstance(strategy, ForecastOptimizer)` and calls `strategy.allocate(historical_returns, forecast)` (AD-23).

**Given** `run_benchmark()` called with a `ForecastOptimizer` but no `forecast` argument,
**When** executed,
**Then** it raises rather than silently proceeding (AD-23).

**Given** `run_benchmark()` called with a `BaselineStrategy` and a caller-supplied `forecast` argument,
**When** executed,
**Then** it raises rather than silently ignoring the `forecast` (AD-23).

**Given** `run_benchmark()`'s source code,
**When** inspected,
**Then** it contains no strategy-specific branching beyond the `isinstance(ForecastOptimizer)` dispatch — strategy behavior lives entirely behind the Portfolio Optimizer Interface (FR-27, mirrors FR-1's guarantee for Market Models).

**Given** `run_benchmark()` called with the same strategy and same returns/scenarios twice,
**When** the two `BenchmarkResult`s are compared,
**Then** they are identical (FR-27, NFR-6).

**Given** `run_benchmark()` called with the Story 4.2 conformance suite's dummy `ForecastOptimizer`,
**When** executed,
**Then** it runs successfully through the full pipeline with zero changes to `run_benchmark()`'s source — the benchmark-layer analogue of SM-1 (FR-25, AD-19).

**Given** `run_benchmark()` called with each of the three Epic 5 Traditional Baselines in turn,
**When** executed,
**Then** all three run successfully through the identical pipeline with zero Runner changes (FR-23, cross-epic extensibility proof).

**Given** a metrics registry argument to `run_benchmark()`,
**When** executed,
**Then** it iterates over the registry generically, calling each `MetricFn` on the derived portfolio return series (AD-18).

---

### Story 6.3: JSON-Serializable `BenchmarkResult`

As a researcher running a benchmark,
I want a structured, JSON-serializable `BenchmarkResult` with a fixed minimum field set,
So that I can persist, compare, and reproduce benchmark runs across strategies and future leaderboards.

**Acceptance Criteria:**

**Given** a completed `run_benchmark()` call,
**When** the returned `BenchmarkResult` is inspected,
**Then** it carries at minimum: `strategy_name`, `strategy_parameters`, `metrics` (a flat `dict[str, float]` keyed by each `MetricFn.name`), `asset_scenario_ids`, `time_grid_reference`, `library_version`, and `generated_at` (FR-29, AD-24).

**Given** a `BenchmarkResult` missing any one of the fields above,
**When** reviewed,
**Then** it fails review — mirrors FR-15's dataset-card gate (FR-29).

**Given** a `BenchmarkResult`,
**When** passed through `json.dumps`/`json.loads` (or equivalent),
**Then** it round-trips without loss (FR-29, NFR-6).

**Given** a `BenchmarkResult`,
**When** inspected,
**Then** it is a plain immutable Python dataclass, not an `equinox.Module`, with only JSON-native field types (AD-17).

**Given** a `CVaROptimization` run with `confidence_level=0.95`,
**When** its `BenchmarkResult` is inspected,
**Then** `strategy_parameters` includes `confidence_level: 0.95` (FR-22, AD-15).

**Given** a multi-asset `run_benchmark()` call,
**When** its `BenchmarkResult` is inspected,
**Then** `asset_scenario_ids` identifies each constituent Scenario/dataset used, and `time_grid_reference` identifies the shared `TimeGrid` (AD-22, AD-24).

---

## Epic 7: Evaluation Results & Leaderboard

**Goal:** A researcher can take a `BenchmarkResult` produced by Epic 6's `run_benchmark()`, publish it as a versioned Evaluation Result to a shared Hugging Face dataset repo, and see it appear — ranked against every other published strategy/Benchmark Dataset combination — in an aggregated Leaderboard table, without writing any publishing, aggregation, or ranking code. This is the epic that realizes UJ-6 end-to-end. A hosted Leaderboard web UI (PRD Feature 4.10) is an explicit future phase, not part of this epic — see each story's Out of Scope.

**FRs covered:** FR-30, FR-31, FR-32, FR-33, FR-34
**NFRs covered:** NFR-7

---

### Story 7.1: `EvaluationResult` Schema

As a researcher who wants to compare benchmark runs across strategies and datasets,
I want a fixed, JSON-native `EvaluationResult` schema derived from `BenchmarkResult`,
So that every published result has the same shape, regardless of which strategy or Benchmark Dataset produced it.

**Acceptance Criteria:**

**Given** the `EvaluationResult` type,
**When** inspected,
**Then** it carries at minimum: `schema_version`, `result_id`, `strategy` (`name`, `parameters`), `benchmark_dataset` (`asset_scenario_ids`, `time_grid_reference`), `metrics`, `library_version`, and `generated_at` (FR-30, AD-26).

**Given** `EvaluationResult.metrics`,
**When** inspected,
**Then** it is an ordered list of `{name, value}` records — not `BenchmarkResult`'s flat `dict[str, float]` — matching the Hugging Face `model-index.results[].metrics[]` convention (FR-30, AD-26).

**Given** an `EvaluationResult` missing any one of the required fields above,
**When** reviewed,
**Then** it fails review, mirroring FR-29's and FR-15's review gates (FR-30).

**Given** an `EvaluationResult`,
**When** passed through `json.dumps`/`json.loads` (or equivalent),
**Then** it round-trips without loss (FR-30, NFR-7).

**Given** `EvaluationResult`,
**When** inspected,
**Then** it is a plain immutable Python dataclass, not an `equinox.Module`, with only JSON-native field types — the same posture AD-17 fixes for `BenchmarkResult` (AD-26).

**Given** `BenchmarkResult`'s own schema (FR-29, AD-24),
**When** `EvaluationResult` is added to the codebase,
**Then** `BenchmarkResult`'s fields, types, and behavior are unchanged — `EvaluationResult` is additive, never a replacement (AD-26).

---

### Story 7.2: `BenchmarkResult` → `EvaluationResult` Transformation

As a researcher who has just completed a `run_benchmark()` call,
I want a single function that converts my `BenchmarkResult` into an `EvaluationResult`,
So that I can publish a run without hand-writing any field-mapping or reshaping logic.

**Acceptance Criteria:**

**Given** a completed `BenchmarkResult`,
**When** `to_evaluation_result(result)` is called,
**Then** it returns an `EvaluationResult` populating every required field (FR-30) from the corresponding `BenchmarkResult` field — `strategy_name`→`strategy.name`, `strategy_parameters`→`strategy.parameters`, `asset_scenario_ids`/`time_grid_reference`→`benchmark_dataset.*`, `metrics`→the reshaped `{name, value}` list, `library_version`, `generated_at` (FR-31, AD-26).

**Given** `to_evaluation_result(result)` called twice on the same `BenchmarkResult`,
**When** the two `EvaluationResult`s are compared,
**Then** they are identical (determinism, mirroring FR-27's guarantee for `BenchmarkResult` itself) (FR-31).

**Given** `to_evaluation_result`'s source code,
**When** inspected,
**Then** it does not mutate or subclass `BenchmarkResult`, and requires zero changes to `run_benchmark()` or any Epic 6 module (FR-31, AD-26).

**Given** `quantscenariobench.benchmark.evaluation`'s import statements,
**When** inspected,
**Then** it imports only `quantscenariobench.benchmark.interface` (to read `BenchmarkResult`) — never `strategies`, `solver`, `metrics`, `returns`, or `runner` directly (mirrors AD-19's dependency posture).

---

### Story 7.3: Local Evaluation Results Storage

As a researcher accumulating benchmark runs over time,
I want to write each `EvaluationResult` to a local, organized file layout,
So that I can keep a full, append-only history of every run without overwriting past results.

**Acceptance Criteria:**

**Given** an `EvaluationResult` for a given Benchmark Dataset and strategy,
**When** written to local storage,
**Then** it is saved as a timestamped JSON file under a directory keyed by Benchmark Dataset and strategy name (e.g. `results/<dataset>/<strategy>/result_<timestamp>.json`) (FR-32).

**Given** two `EvaluationResult`s written for the same Benchmark Dataset/strategy combination,
**When** both writes complete,
**Then** two separate files exist — neither write overwrites the other (FR-32).

**Given** the local file layout produced by this story,
**When** compared to the Hugging Face upload path (Story 7.4),
**Then** the directory structure requires no reorganization to be uploaded as-is (FR-32).

---

### Story 7.4: Hugging Face Evaluation Results Publishing

As a researcher who wants their benchmark runs to be publicly comparable,
I want to publish an `EvaluationResult` to a shared Hugging Face Evaluation Results dataset repo,
So that anyone can see how my strategy performed alongside every other published result, without me hosting anything myself.

**Acceptance Criteria:**

**Given** one or more `EvaluationResult`s and a target Hugging Face dataset repo ID,
**When** published,
**Then** each is uploaded as its own file under the repo's results layout (mirroring Story 7.3's local organization), and none of the repo's previously published results is overwritten (FR-33).

**Given** a publish call to the shared Evaluation Results repo,
**When** it completes,
**Then** the repo's README/card is regenerated to reflect the current set of published results, mirroring `generate_dataset_card`'s role for Benchmark Datasets (FR-33, FR-15).

**Given** the Hugging Face publishing function's source code,
**When** inspected,
**Then** it consumes `EvaluationResult` exclusively — it never reads a `BenchmarkResult` directly (AD-26).

**Out of Scope (this story):** an external/community submission workflow or write-access model beyond the maintainer's own token-based publish (deferred; see Architecture Spine Deferred section).

---

### Story 7.5: Leaderboard Aggregation

As a researcher comparing strategies across Benchmark Datasets,
I want a generic reader that loads every published `EvaluationResult` and ranks them in one table,
So that I can see how strategies compare without writing any aggregation or ranking code myself.

**Acceptance Criteria:**

**Given** a Hugging Face Evaluation Results repo (or an equivalent local collection) containing `EvaluationResult`s for multiple strategies and Benchmark Datasets,
**When** the aggregation reader is called,
**Then** it returns a table with one row per strategy × Benchmark Dataset combination and one column per Metric name (FR-34).

**Given** a newly published `EvaluationResult` for a strategy/Benchmark Dataset combination not previously seen,
**When** the aggregation reader is re-run,
**Then** the new combination appears as a new row, with zero changes to the aggregation function's source (FR-34, the Leaderboard-layer analogue of AD-18's extensibility guarantee for Metrics).

**Given** the aggregation reader's return value,
**When** inspected,
**Then** it is a plain, headlessly-testable table/data structure with no dependency on any UI framework (FR-34).

**Given** the aggregation reader's source code,
**When** inspected,
**Then** it contains no strategy-specific or dataset-specific branching — it is generic purely because `EvaluationResult`'s schema (AD-26) is fixed (FR-34, mirrors AD-18/AD-19's genericity requirement).

**Out of Scope (this story):** a hosted, browsable Leaderboard web UI (a Hugging Face Space, Gradio/Streamlit app, or equivalent) — the aggregation this story produces is UI-agnostic by design; rendering it as a live page is PRD Feature 4.10, an explicit future phase (Epic 8, not yet planned).
