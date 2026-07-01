# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) for the
library itself. Dataset versions are tracked independently (see `dataset_version`
in each `Scenario`'s metadata).

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
