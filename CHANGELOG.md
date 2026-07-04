# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) for the
library itself. Dataset versions are tracked independently (see `dataset_version`
in each `Scenario`'s metadata).

## [1.2.1] - 2026-07-04

### Fixed

- **Leaderboard Space**: the Space now handles a missing, private, gated,
  or otherwise inaccessible Evaluation Results repository gracefully —
  and a repository that is reachable but has zero published results —
  instead of crashing. Both cases render an empty table with a clear,
  user-facing message. Previously, an unset or nonexistent
  `QSB_EVAL_RESULTS_REPO` (including the packaged default) caused the
  Space to fail on load with an unhandled `RepositoryNotFoundError`.
  Existing behavior when `QSB_EVAL_RESULTS_REPO` points to a valid,
  populated dataset repo is unchanged.

## [1.2.0] - 2026-07-04

### Added

- **Leaderboard Space** (Epic 8): a hosted, browsable Hugging Face Space
  (`spaces/leaderboard/`) that renders v1.1's Leaderboard aggregation as a
  live page — the Hugging Face Space / Gradio Leaderboard UI promised in
  the v1.1 Notes below.
  - Built with Gradio; a presentation layer only — it consumes
    `quantscenariobench.benchmark.evaluation`'s existing
    `EvaluationResult`/Leaderboard aggregation pipeline (`aggregate_evaluation_results`,
    `load_evaluation_results_from_hub`) as an ordinary library dependency
    and adds no aggregation, ranking, or data-model logic of its own.
  - Live table rendering, refreshed fresh on every session load so newly
    published results appear without redeploying the Space.
  - Sorting by any column (Metric, Strategy, or Benchmark Dataset), via
    Gradio's native `Dataframe` column-header sort.
  - Filtering by Benchmark Dataset, Strategy, and Metric — independently
    or combined — via `filter_leaderboard()`.
  - Space deployment configuration (`spaces/leaderboard/README.md`):
    Hugging Face Space metadata (SDK, entry point, suggested hardware)
    and a documented git-push-to-deploy path.

### Notes

- The Space is scoped deliberately narrow: a ranked, sortable, filterable
  table only. Advanced analytics, visualizations, historical/trend
  tracking, and strategy-to-strategy comparison tooling remain explicitly
  out of scope for this release.
- The Space's compute tier defaults to the free CPU tier (no model
  inference is performed) and its data currency is read-on-session-load,
  with no server-side cache or scheduler — both sized to the project's
  current traffic, not committed as permanent architecture. See
  `spaces/leaderboard/README.md` and the Architecture Spine's Deferred
  section for the upgrade path if that changes.
- The Hugging Face namespace for the Space and its Evaluation Results
  repo is not yet finalized — see the README's `QSB_EVAL_RESULTS_REPO`
  configuration section.

## [1.1.0] - 2026-07-03

### Added

- **Benchmark Core** (Epics 4–6): a portfolio-benchmarking layer built on the
  same conformance-suite pattern as the Market Model layer.
  - Portfolio Optimizer Interface (`BaselineStrategy`, `ForecastOptimizer`)
    and a validated `PortfolioWeights` type
    (`quantscenariobench.benchmark.interface`).
  - Four portfolio performance metrics — Sharpe Ratio, Sortino Ratio,
    Maximum Drawdown, Final Wealth Factor
    (`quantscenariobench.benchmark.metrics`).
  - Three traditional baseline strategies — `EqualWeight`,
    `GlobalMinimumVariance`, `CVaROptimization`
    (`quantscenariobench.benchmark.strategies`).
  - Return-series derivation and multi-asset composition from one or more
    Scenarios sharing a common `TimeGrid` (`derive_returns`,
    `compose_returns`).
  - `run_benchmark()` orchestrator producing a JSON-serializable
    `BenchmarkResult`.
  - Portfolio Optimizer conformance test suite for third-party strategy
    authors, mirroring the Market Model conformance suite.
- **Evaluation Results & Leaderboard pipeline** (Epic 7):
  - `EvaluationResult` schema — a fixed, JSON-native, versioned record
    derived from `BenchmarkResult`
    (`quantscenariobench.benchmark.evaluation`).
  - `to_evaluation_result()`, a pure `BenchmarkResult` → `EvaluationResult`
    transform.
  - Local Evaluation Results storage (`write_evaluation_result`) — one
    timestamped, append-only JSON file per run, organized by Benchmark
    Dataset and strategy.
  - Hugging Face Evaluation Results publishing (`publish_evaluation_results`,
    `generate_evaluation_results_card`) to a shared, append-only dataset
    repo with an auto-generated summary card.
  - Leaderboard aggregation (`aggregate_evaluation_results`,
    `load_evaluation_results`, `load_evaluation_results_from_hub`) — a
    generic reader that builds a ranked strategy × Benchmark Dataset table
    from every published `EvaluationResult`, with no UI dependency.

### Notes

- This release ships **leaderboard aggregation only**: a ranked table of
  rows you can load into pandas or print yourself. It does not include a
  hosted or public leaderboard UI. A Hugging Face Space (Gradio Leaderboard
  UI) is planned for **v1.2**.
- README restructured to distinguish current capabilities from planned
  (v1.2) work; see its new Roadmap section.

## [1.0.0] - 2026-07-01

Initial public release.

### Added

- Core state-space interface (`MarketModel`, `Scenario`, `TimeGrid`, `Metadata`)
  shared across all models, solver, export, and testing utilities.
- `simulate()` public orchestrator with reproducible seeding, full provenance
  metadata, and opt-in randomness materialization/replay.
- diffrax-backed solver layer for SDE integration.
- Three v1 market models:
  - **Black-Scholes** — Geometric Brownian Motion, validated against the
    closed-form pricing formula.
  - **Heston** — stochastic volatility, validated via semi-closed-form
    (Gil-Pelaez inversion) pricing.
  - **Rough Bergomi** — non-Markovian rough volatility (Volterra fBM),
    validated via statistical skew-monotonicity properties.
- State-space interface conformance test suite for third-party model authors.
- Parquet export (`export_parquet`) with a fixed 12-column schema shared
  across all models.
- Hugging Face Hub publishing (`publish_to_hub`) and dataset card generation
  (`generate_dataset_card`).
- Three published benchmark sample datasets on the Hugging Face Hub
  (Black-Scholes, Heston, Rough Bergomi) under the `QuantScenarioBench` org.
- Project README covering quick start, model reference, export/publish
  workflow, reproducibility guarantees, and custom model authoring.
