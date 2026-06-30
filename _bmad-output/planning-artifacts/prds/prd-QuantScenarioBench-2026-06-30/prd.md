---
title: PRD - QuantScenarioBench
status: final
created: 2026-06-30
updated: 2026-06-30
---

# PRD: QuantScenarioBench

## 0. Document Purpose

This PRD defines the v1 functional and quality requirements for QuantScenarioBench, a finance-scoped, JAX-native Python framework for generating reproducible market scenarios and publishing them as versioned benchmark datasets. It builds directly on the finalized [Product Brief](../../briefs/brief-QuantScenarioBench-2026-06-30/brief.md) and its [Addendum](../../briefs/brief-QuantScenarioBench-2026-06-30/addendum.md) — read those first for the problem framing, landscape research, and rationale; this document does not repeat them. It is written for the author (also the primary user) and for downstream BMad workflows (architecture, epics/stories) that will derive implementation plans from it. Terms are Glossary-anchored (§3); features are grouped with Functional Requirements (FRs) nested under them and numbered globally; inferred content is tagged inline as `[ASSUMPTION]` and indexed in §9.

## 1. Vision

QuantScenarioBench gives a quantitative researcher one consistent way to generate market scenarios, regardless of which stochastic process produces them. A researcher writes the same `simulate()` call against Black-Scholes, Heston, or rough Bergomi, gets back a `Scenario` object with the same top-level shape every time, and never has to learn a model-specific simulation API to compare results across models.

The framework's defining bet is architectural, not algorithmic: every Market Model implements one State-Space Interface (`observation`, `latent_state`, `metadata`, with `randomness` optional), so the simulation core and the dataset export pipeline never need to change when a new model is added. v1 proves this contract with three models of meaningfully different mathematical character — Black-Scholes (closed-form), Heston (semi-closed-form), rough Bergomi (no closed form) — and ships the first versioned, Parquet-backed benchmark datasets to Hugging Face built on top of it.

This is a launch-grade specification for an intentionally small v1: the design must hold up to public scrutiny and to a second and third model being added later without rework, even though the first implementation stays narrow. As the brief states plainly, the bet here is execution and integration, not a novel algorithm — the stochastic models and the autodiff machinery already exist; the product is assembling them into one coherent, reusable, published toolkit.

Beyond v1, the same contract is meant to outlive the three models it ships with: a researcher who wants to contribute a new Market Model should be able to do so against the published State-Space Interface (Feature 4.3) without coordinating with the maintainer on simulation or export internals.

## 2. Target User

### 2.1 Jobs To Be Done

- Generate reproducible synthetic market scenarios to train or evaluate pricing models, hedging strategies, or risk estimators — without hand-building a simulator per project.
- Compare a downstream model's behavior across stochastic processes with different mathematical character (closed-form vs. semi-closed-form vs. no closed form) through one consistent interface.
- Consume a standardized, versioned benchmark dataset directly from Hugging Face, without running simulation code locally.
- Extend the framework with a new Market Model by implementing the State-Space Interface alone, without touching simulation or export internals.
- *(secondary, beyond v1)* Adopt QuantScenarioBench's published datasets and Market Models as a shared community benchmark, rather than each researcher publishing one-off synthetic data alongside an individual paper.

### 2.2 Non-Users (v1)

- Live or production trading/execution systems — QuantScenarioBench simulates from specified parameters; it does not calibrate to or connect with real market data.
- Anyone needing multi-asset or cross-asset (basket, correlated multi-underlying) scenario generation — v1 is single-asset only.
- Anyone who needs oracle labels (theoretical pricing, hedging deltas) shipped with the v1 datasets — acknowledged as a goal, not delivered in v1 (see [Open Questions & Risks](../../briefs/brief-QuantScenarioBench-2026-06-30/brief.md#open-questions--risks) in the brief).
- CLI-only or non-Python workflows — v1 ships a Python API only.

### 2.3 Key User Journeys

*Single-persona, API-first, capability-driven product — journeys are kept to one-line scope per the Lighter dial rather than full named-persona narratives.*

- **UJ-1.** A researcher configures a `Heston` Market Model and calls `simulate()` with a seed to get a reproducible `Scenario` for local model training or evaluation. Realizes Feature 4.1.
- **UJ-2.** A researcher loads a published QuantScenarioBench dataset directly from the Hugging Face Hub to benchmark a model, without installing or running the simulation library. Realizes Feature 4.4.
- **UJ-3.** A contributor adds a new Market Model by implementing the State-Space Interface, with no changes to `simulate()` or the export pipeline. Realizes Feature 4.3.

## 3. Glossary

- **Market Model** — A typed configuration object (a JAX PyTree dataclass, e.g. `BlackScholes(...)`, `Heston(...)`) describing a stochastic process, passed to `simulate()`. v1 models: Black-Scholes, Heston, rough Bergomi (rBergomi).
- **State-Space Interface** — The contract every Market Model implementation must satisfy to plug into `simulate()` and the export pipeline: it must produce `Observation`, may produce `Latent State`, always produces `Metadata`, and may optionally expose `Randomness`.
- **Scenario** — The object returned by `simulate()`. Stable top-level schema: `observation`, `latent_state`, `metadata`. Schema is identical in shape across all Market Models, even when a field is empty for a given model.
- **Observation** — The simulated output series a downstream consumer treats as "the market data" (e.g. an asset price path).
- **Latent State** — Model-internal stochastic state not directly observed (e.g. Heston's variance process). Model-specific; may be empty for models with no separate latent process (e.g. Black-Scholes).
- **Metadata** — The provenance record attached to a Scenario: seed/PRNG information plus enough generation context (Market Model identity and parameters, TimeGrid, library version) to make the Scenario reproducible on the same backend. `[ASSUMPTION: exact field list — see §9]`
- **Randomness** — The raw underlying random draws used during simulation. Conceptually part of the State-Space Interface; materialized in a returned Scenario only when explicitly requested.
- **TimeGrid** — An explicit object describing the simulation's time points, supporting non-uniform grids (not a `(start, stop, steps)` tuple).
- **Benchmark Dataset** — A published, versioned, Parquet-backed Hugging Face dataset of Scenarios for one Market Model.
- **Oracle Label** — A ground-truth value (theoretical price, hedging delta) attached to a Scenario for supervised benchmarking. Out of scope for v1; referenced here because it shapes the Metadata and Benchmark Dataset schemas going forward.

## 4. Features

### 4.1 Core Simulation API

**Description:** The researcher-facing entry point: a single `simulate()` call that works identically across all Market Models. Realizes UJ-1. This feature is the literal expression of the brief's primary success criterion — the simulation core must not change as Market Models are added (see Feature 4.3).

**Functional Requirements:**

#### FR-1: Single-call scenario simulation

A researcher can call `simulate(model=<MarketModel>, time_grid=<TimeGrid>, n_paths=<int>, seed=<int>) -> Scenario` for any v1 Market Model. Realizes UJ-1.

**Consequences (testable):**
- The call signature and return type are identical across Black-Scholes, Heston, and rBergomi.
- `simulate()` itself contains no Market Model-specific branching — model behavior lives entirely behind the State-Space Interface.

#### FR-2: Stable Scenario schema

Every `simulate()` call returns a `Scenario` exposing `observation`, `latent_state`, and `metadata` as top-level attributes, with identical shape and field names across all Market Models.

**Consequences (testable):**
- `latent_state` is present (possibly empty) on every Scenario, including Black-Scholes, which has no latent process.
- Adding a new Market Model never adds, removes, or renames a top-level Scenario field.

#### FR-3: Explicit TimeGrid

Simulation time points are specified via a `TimeGrid` object, not a `(start, stop, steps)` tuple, so non-uniform grids are representable without a different call shape.

**Consequences (testable):**
- A non-uniform `TimeGrid` (irregular spacing) produces a correctly time-indexed `Scenario` for every v1 Market Model.

#### FR-4: Reproducibility via Metadata

Given the same Market Model, parameters, TimeGrid, and seed, `simulate()` produces a deterministic `Scenario` on a given backend. `Metadata` carries the seed/PRNG provenance needed to describe how the Scenario was generated. `[ASSUMPTION: full Metadata field list not yet specified — see §9]`

**Consequences (testable):**
- Two `simulate()` calls with identical arguments on the same backend (CPU, or the same GPU/TPU configuration) produce bit-identical `observation` and `latent_state` arrays.
- Cross-backend bit-identity (e.g. CPU vs. GPU) is explicitly NOT guaranteed — see Cross-Cutting NFRs.

**Out of Scope:**
- Guaranteeing identical results across different JAX backends or hardware.

#### FR-5: Optional randomness materialization

A caller can request the underlying `Randomness` used to generate a Scenario; by default it is not materialized or returned. `[ASSUMPTION: opt-in mechanism, e.g. a `return_randomness` flag on `simulate()` — exact API not yet specified]`

**Consequences (testable):**
- Default `simulate()` calls do not allocate or return raw random draws.
- When requested, returned `Randomness` is sufficient to reconstruct the Scenario's `observation`/`latent_state` deterministically.

#### FR-6: Soft parameter validation

Market Model parameter construction validates research-meaningful constraints where practical (e.g. the Heston Feller condition) and emits a warning — never a hard exception — when violated.

**Consequences (testable):**
- Constructing a Heston model with Feller-violating parameters succeeds and raises a Python warning, not an exception.
- A researcher can deliberately explore constraint-violating parameter regions without being blocked.

**Feature-specific NFRs:**
- The State-Space Interface (Feature 4.3) is the only extension point `simulate()` depends on; no Market Model-specific logic lives in the core API surface.

### 4.2 v1 Market Model Zoo

**Description:** Three Market Models, chosen specifically to span different mathematical character — closed-form, semi-closed-form, and no-closed-form pricing — so the State-Space Interface is proven against real diversity, not three variations on the same theme.

**Functional Requirements:**

#### FR-7: Black-Scholes model

A `BlackScholes` Market Model conforms to the State-Space Interface, with no `latent_state` (or an explicitly empty one).

**Consequences (testable):**
- Simulated `observation` paths reproduce the closed-form Black-Scholes price within a defined numerical tolerance under Monte Carlo averaging (see Feature-specific NFR on reference implementations, below).

#### FR-8: Heston model

A `Heston` Market Model conforms to the State-Space Interface, exposing the variance process as `latent_state`.

**Consequences (testable):**
- Simulated paths are consistent with Heston's semi-closed-form (characteristic-function-based) pricing within a defined numerical tolerance (see Feature-specific NFR on reference implementations, below).
- Constructed with Feller-violating parameters per FR-6, simulation still completes and produces a valid Scenario.

#### FR-9: Rough Bergomi (rBergomi) model

A `RoughBergomi` Market Model conforms to the State-Space Interface, exposing its volatility process as `latent_state`.

**Consequences (testable):**
- Since rBergomi has no closed-form or semi-closed-form pricing reference (confirmed in the brief's addendum), correctness is validated via distributional/statistical sanity checks (e.g. known asymptotic or moment properties of rough volatility paths) rather than a closed-form comparison. A Markovian approximation (e.g. aBergomi, named in the brief's addendum) is a candidate technique for generating a faster-to-compute reference for these checks. `[ASSUMPTION: specific statistical test suite not yet defined; aBergomi noted as a candidate, not committed]`

**Out of Scope:**
- Oracle labels (theoretical price, hedging delta) for any v1 model — see brief §Scope.

**Feature-specific NFRs:**
- Every correctness check in this feature (FR-7, FR-8, FR-9) validates against an independently implemented reference, not against a general-purpose quant library taken as a bundled dependency — consistent with the brief's framing of this project as its own integration, not a thin wrapper around existing tools (e.g. `tf-quant-finance`).

### 4.3 State-Space Extensibility Contract

**Description:** The structural promise the whole product is built on: a new Market Model can be added by implementing the State-Space Interface alone. Realizes UJ-3. This feature exists to make the brief's primary success criterion testable, not just stated.

**Functional Requirements:**

#### FR-10: Public Market Model interface

A documented interface (e.g. a Python Protocol or abstract base class) specifies exactly what a Market Model must implement — its parameter dataclass shape and its simulation entry point — to satisfy the State-Space Interface and be usable by `simulate()` and the dataset export pipeline (Feature 4.4).

**Consequences (testable):**
- A new Market Model implementation requires zero changes to `simulate()`'s source or the export pipeline's source.

#### FR-11: Interface conformance test suite

A reusable test harness verifies that a given Market Model implementation satisfies the State-Space Interface (correct Scenario field shapes, reproducibility under FR-4, parameter validation behavior under FR-6). The suite includes a minimal dummy Market Model that exists only inside the test suite — never published, never part of the public framework surface — used specifically to prove the interface holds independent of any real financial model. `[ASSUMPTION: exact test harness mechanism — e.g. shared pytest fixtures or property-based tests — not yet specified]`

**Consequences (testable):**
- Running the conformance suite against any v1 Market Model (Black-Scholes, Heston, rBergomi) passes.
- The test-only dummy Market Model passes the conformance suite without any modification to `simulate()`, `Scenario`, or the export pipeline, proving the interface is satisfiable by an implementation the core was never written against.

**Out of Scope:**
- The dummy Market Model is not a shipped deliverable — it is not published as a Benchmark Dataset and is not part of the public framework API.

**Notes:** v1 deliverables stay focused on real financial models (Feature 4.2); extensibility is proven entirely through this test suite, not by shipping an artificial model as a product feature.

### 4.4 Benchmark Dataset Export & Publishing

**Description:** Turns a batch of Scenarios into a versioned, public artifact. Realizes UJ-2. One Hugging Face dataset per Market Model, sharing a common schema so cross-model comparison stays mechanical.

**Functional Requirements:**

#### FR-12: Parquet export

A batch of Scenarios for a given Market Model serializes to Parquet file(s) with columns mirroring the Scenario schema (`observation`, `latent_state`, `metadata`). `[ASSUMPTION: row granularity — one row per simulated path vs. one row per batch — not yet specified]`

**Consequences (testable):**
- Exported Parquet round-trips back into Scenario-equivalent data without loss.

#### FR-13: Per-model Hugging Face dataset publishing

Each v1 Market Model (Black-Scholes, Heston, rBergomi) publishes as its own Hugging Face dataset, all three sharing the same top-level column schema.

**Consequences (testable):**
- An external researcher can `datasets.load_dataset(...)` any of the three v1 datasets and get data in the documented shared schema.

#### FR-14: Independent dataset versioning

Published Benchmark Datasets carry their own version identifier, decoupled from the QuantScenarioBench library's release version. `[ASSUMPTION: versioning scheme — e.g. semver per dataset, tied to generation-parameter changes — not yet specified, see §9]`

**Consequences (testable):**
- The library can release a new version without forcing a new dataset version, and vice versa.

#### FR-15: Dataset documentation

Each published Benchmark Dataset includes a dataset card with, at minimum: the column schema, the Market Model name and parameter values used to generate it, the `TimeGrid` and `n_paths` used, the library version that produced it, and the dataset's own version identifier (FR-14). `[ASSUMPTION: this minimum field list, beyond "a dataset card exists," is inferred]`

**Consequences (testable):**
- Every published dataset's card contains all six fields listed above; a card missing any one of them fails review.

## 5. Non-Goals (Explicit)

- Calibrating Market Models to live or historical market data — v1 simulates from specified parameters only.
- Multi-asset, basket, or cross-asset correlated simulation — v1 is single-asset only.
- A CLI — v1 ships a Python API only, until a clear need for a CLI emerges.
- Oracle labels (theoretical pricing, hedging deltas) as a shipped dataset feature — acknowledged as the long-term direction (AD-through-paths, with Monte Carlo as an accepted interim), not committed to v1.
- GPU/TPU performance optimization as a goal in itself — JAX provides this where used, but it is not a v1 design target, and Success Metrics (§7) explicitly do not reward optimizing for it.
- Evaluating the BMAD + Claude Code AI development workflow — a real motivation for building this project (see brief addendum), but a development-process objective, not a product requirement.

## 6. MVP Scope

### 6.1 In Scope

- `simulate()` API, `Scenario` object, `TimeGrid` object (FR-1 through FR-5).
- Soft parameter validation (FR-6).
- Black-Scholes, Heston, and rough Bergomi Market Models (FR-7 through FR-9).
- The public State-Space Interface and its conformance test suite, including a test-only dummy Market Model used solely to prove extensibility (FR-10, FR-11).
- Parquet export and per-model Hugging Face dataset publishing, independently versioned, with dataset cards (FR-12 through FR-15).
- `[ASSUMPTION]` Documentation sufficient for an external researcher to generate or consume a dataset without reading the source: a README quickstart, an API reference, and one runnable example notebook per v1 Market Model (carried from the brief's Scope; this specific checklist is inferred).

### 6.2 Out of Scope for MVP

*Kept narrow deliberately: this is currently a one-person open-source effort, and a multi-model roadmap is ambitious against that bandwidth. Proving the interface on three models before expanding manages that risk rather than ignoring it.*

- SABR, jump-diffusion, and any further Market Model beyond the v1 three — deferred until the v1 interface proves itself.
- Oracle labels in published datasets — deferred; direction acknowledged (§ brief Solution), implementation not committed.
- CLI, multi-asset support, real-market calibration — see §5 Non-Goals.
- GPU/TPU performance tuning as a dedicated workstream.

## Cross-Cutting NFRs

- **Determinism is backend-scoped.** Reproducibility (FR-4) holds for repeated runs on the same backend; cross-backend (CPU/GPU/TPU) bit-identity is explicitly not guaranteed, consistent with the brief's Open Questions & Risks.
- **JAX-native computation.** All Market Model simulation logic is implemented in JAX (jit/vmap-compatible), consistent with the framework's core identity. The choice of underlying SDE-integration machinery (e.g. building on an existing JAX differential-equation solver vs. implementing one) is an Architecture-phase decision — see Open Question 8.
- **Numerical correctness testing.** Every v1 Market Model has an automated correctness check appropriate to its mathematical character (closed-form comparison, semi-closed-form comparison, or statistical sanity check) — see FR-7 through FR-9.
- **Public API stability policy.** Backward-incompatible changes to `simulate()`, the `Scenario` schema, or the State-Space Interface require a major version bump. `[ASSUMPTION: semantic versioning assumed; not explicitly confirmed]`
- **Language/runtime targets.** Python >=3.11, jax >=0.4.38 (confirmed during Architecture: driven by diffrax 0.7.2's minimum requirements — see [Architecture Spine](../../architecture/architecture-QuantScenarioBench-2026-06-30/ARCHITECTURE-SPINE.md) Stack table). Supersedes this PRD's original `[ASSUMPTION: Python 3.10+]`.

## 7. Success Metrics

**Primary**
- **SM-1**: API stability under extension — the test-only dummy Market Model (FR-11) passes the conformance suite with zero changes to `simulate()`, `Scenario`, or the export pipeline source. Validates FR-10, FR-11.
- **SM-2**: Reproducibility — a fixed-seed `simulate()` call produces a bit-identical Scenario on repeated runs on the same backend. Validates FR-4.
- **SM-3**: Dataset usability — all three v1 Benchmark Datasets load successfully via `datasets.load_dataset(...)` and conform to the documented shared schema. Validates FR-12, FR-13, FR-15.

**Secondary**
- **SM-4**: Model correctness — Black-Scholes and Heston Scenarios match their closed-form/semi-closed-form references within tolerance; rBergomi Scenarios pass their statistical sanity checks. Validates FR-7, FR-8, FR-9.
- **SM-5** *(post-v1, tracked not gated)*: External usage signals — stars, downloads, forks, citations — as evidence of adoption beyond internal use.

**Counter-metrics (do not optimize)**
- **SM-C1**: Raw simulation throughput/performance. Optimizing this at the expense of API simplicity or the State-Space Interface's stability would undermine SM-1 — counterbalances any temptation to hand-tune for speed during v1. Counterbalances SM-1.
- **SM-C2**: Model count. Adding Market Models faster than the interface can absorb them without modification would falsify SM-1 even while looking like progress. Counterbalances SM-1, SM-4.

## 8. Open Questions

1. Open-source license — not yet chosen (brief states "fully open source" but no specific license).
2. Exact `Metadata` field list (FR-4) — seed/PRNG info confirmed; full schema (model identity, parameter values, TimeGrid reference, library version, timestamp) not yet confirmed.
3. Dataset versioning scheme specifics (FR-14) — semver-per-dataset vs. content-hash vs. another scheme.
4. Parquet row granularity (FR-12) — one row per path vs. one row per batch/run.
5. Hugging Face organization/namespace and dataset naming convention (FR-13).
6. Whether CI/test-infrastructure requirements belong in this PRD as NFRs, or are entirely development-workflow scope (per the brief's exclusion of the AI-dev-workflow motivation from product success criteria) and therefore out of this document.
7. Dataset generation and hosting cost at scale — flagged as an unresolved risk in the brief; no budget or ceiling defined yet.
8. What QuantScenarioBench builds its SDE/path-integration machinery on — the brief's addendum names `diffrax` as the most likely foundation, and notes it would be additive rather than a wrapper around an existing common-API library like `tf-quant-finance`. This PRD deliberately leaves the choice to the Architecture phase, since it is an implementation decision, not a capability requirement — see [PRD addendum](addendum.md) for the full landscape context carried forward from the brief.

## 9. Assumptions Index

- §3 Glossary / FR-4 — exact `Metadata` field list not yet specified.
- FR-5 — randomness materialization opt-in mechanism (e.g. a flag) not yet specified.
- FR-9 — statistical/distributional test suite for rBergomi correctness not yet defined.
- FR-11 — conformance test harness mechanism (shared fixtures vs. property-based tests) not yet specified.
- FR-12 — Parquet row granularity not yet specified.
- FR-14 — dataset versioning scheme not yet specified.
- FR-15 — dataset card contents inferred from common HF practice, not explicitly specified.
- §6.1 — documentation/examples requirement carried over from the brief's own `[ASSUMPTION]`.
- Cross-Cutting NFRs / Public API stability policy — semantic versioning assumed, not explicitly confirmed.
- ~~Cross-Cutting NFRs / Language-runtime targets~~ — resolved during Architecture (Python >=3.11, jax >=0.4.38); no longer an open assumption.
