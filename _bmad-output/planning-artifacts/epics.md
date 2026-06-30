---
stepsCompleted: [1]
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

### NonFunctional Requirements

NFR-1: Determinism is backend-scoped — reproducibility (FR-4) holds for repeated runs on the same backend; cross-backend (CPU/GPU/TPU) bit-identity is not guaranteed.
NFR-2: All Market Model simulation logic is implemented in JAX (jit/vmap-compatible).
NFR-3: Every v1 Market Model has an automated correctness check appropriate to its mathematical character (closed-form, semi-closed-form, or statistical sanity check).
NFR-4: Backward-incompatible changes to `simulate()`, the `Scenario` schema, or the State-Space Interface require a major version bump (public API semantic versioning).
NFR-5: Language/runtime targets: Python >=3.11, jax >=0.4.38 (confirmed during Architecture, supersedes the PRD's original 3.10+ assumption).

### Additional Requirements (from Architecture Spine)

**No starter template** — this is a from-scratch Python package; the Architecture Spine's Structural Seed source tree is the initial package layout for Epic 1 Story 1, not a third-party starter.

**Stack (pinned in Architecture):** Python >=3.11, jax >=0.4.38, diffrax 0.7.2, equinox >=0.11.10.

**Package layout (Structural Seed):** `quantscenariobench.interface` (State-Space Interface, Scenario, TimeGrid), `quantscenariobench.models` (Black-Scholes, Heston, RoughBergomi), `quantscenariobench.solver` (internal Solver Layer, the only module importing diffrax), `quantscenariobench.api` (`simulate()`), `quantscenariobench.export` (Parquet/HF export), `quantscenariobench.testing` (conformance suite + dummy Market Model).

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
- AD-10: Correctness-check references are independently implemented, never borrowed from a bundled quant library.
- AD-11: Public API stability follows semantic versioning, independent of dataset versioning.
- AD-12: `TimeGrid` is an explicit, ordered time-point sequence, not a generative spec.

**Deferred in Architecture (not yet covered by any story — flagged so they aren't silently assumed done):** release/publish operational envelope (PyPI release process, HF Hub publish trigger/auth), runtime/compute environment (CPU/GPU/TPU), dataset generation/hosting cost at scale, Parquet row granularity, dataset versioning scheme specifics, Hugging Face org/namespace convention, conformance test harness mechanism (pytest vs. property-based), rBergomi statistical test suite specifics, open-source license choice.

### UX Design Requirements

Not applicable — no UI, no UX design document exists for this project.

### FR Coverage Map

{{requirements_coverage_map}}

## Epic List

{{epics_list}}
