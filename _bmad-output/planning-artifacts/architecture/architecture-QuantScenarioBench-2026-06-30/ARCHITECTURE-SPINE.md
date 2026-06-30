---
name: 'QuantScenarioBench'
type: architecture-spine
purpose: build-substrate
altitude: initiative
paradigm: 'Strategy-pattern plugin core within a linear pipeline (Market Model -> Solver Layer -> Scenario -> generic Export)'
scope: 'QuantScenarioBench v1 - finance-scoped JAX market scenario simulation framework: the State-Space Interface, the simulation/solver boundary, and the dataset export boundary'
status: final
created: '2026-06-30'
updated: '2026-06-30'
binds: ['PRD Feature 4.1 Core Simulation API', 'PRD Feature 4.2 v1 Market Model Zoo', 'PRD Feature 4.3 State-Space Extensibility Contract', 'PRD Feature 4.4 Benchmark Dataset Export & Publishing']
sources: ['_bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/prd.md', '_bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/addendum.md']
companions: []
---

# Architecture Spine — QuantScenarioBench

## Design Paradigm

A linear pipeline with exactly one Strategy-pattern extension point. A Market Model is a typed configuration object satisfying the State-Space Interface (the Strategy); the Solver Layer is the single component that turns a Market Model's drift/diffusion into a simulated path via diffrax; `simulate()` is the orchestrator that wires Market Model + Solver Layer into a `Scenario`; Export is a generic consumer of the `Scenario` schema. No layer except Solver Layer knows diffrax exists; no layer except the caller of `simulate()` knows which concrete Market Model is in play.

Layer → namespace mapping:
- `quantscenariobench.interface` — the State-Space Interface (`MarketModel` ABC), `Scenario`, `TimeGrid`.
- `quantscenariobench.models` — concrete Market Models (`BlackScholes`, `Heston`, `RoughBergomi`), each an `equinox.Module` satisfying `MarketModel`.
- `quantscenariobench.solver` — the internal Solver Layer; the only module that imports `diffrax`.
- `quantscenariobench.api` — `simulate()`, the public orchestrator.
- `quantscenariobench.export` — Parquet/Hugging Face export, generic over `Scenario`.
- `quantscenariobench.testing` — the State-Space Interface conformance suite and its test-only dummy Market Model (FR-11); never imported by non-test code.

## Invariants & Rules

### AD-1 — State-Space Interface is an `equinox.Module` ABC

- **Binds:** every Market Model; `quantscenariobench.interface`, `quantscenariobench.models`.
- **Prevents:** structural-typing-only contracts that skip `eqx`'s native pytree/jit integration, and ad hoc per-model pytree registration.
- **Rule:** `MarketModel` is an `equinox.Module` subclass with abstract methods enforced at construction time. Every concrete Market Model subclasses `MarketModel`; none registers itself as a pytree by hand.

### AD-2 — `Scenario` is an `equinox.Module` with a fixed dynamic/static field split

- **Binds:** `simulate()`'s return type; `quantscenariobench.export`'s input type.
- **Prevents:** `metadata` (strings, ints, version identifiers) leaking into traced pytree leaves and breaking `jit`/`vmap`.
- **Rule:** `Scenario.observation` and `Scenario.latent_state` are dynamic (traced) `eqx.Module` fields. `Scenario.metadata` is `eqx.field(static=True)` — pytree aux_data, never a traced leaf.

### AD-3 — Randomness defaults to `diffrax.VirtualBrownianTree`; materialization is a separate path

- **Binds:** `quantscenariobench.solver`.
- **Prevents:** per-model divergence in how Brownian motion is constructed; a single code path silently paying full noise-array memory cost regardless of whether the caller wants it.
- **Rule:** The Solver Layer constructs Brownian motion via `diffrax.VirtualBrownianTree` by default (FR-4's reproducibility holds via seed/PRNG provenance, not via a stored noise path). When a caller requests materialized `Randomness` (FR-5), the Solver Layer takes an explicit, separate construction path — never a runtime branch inside the default path.

### AD-4 — Solver Layer wraps `diffrax` exclusively, behind one fixed drift/diffusion signature

- **Binds:** all Market Model implementations (FR-7, FR-8, FR-9); the State-Space Extensibility Contract (FR-10).
- **Prevents:** two Market Models independently choosing incompatible `TimeGrid`→`SaveAt` mappings, noise construction, or solver configuration that both "pass" yet produce inconsistent `Scenario` semantics; two Market Models exposing drift/diffusion under different call signatures; a Market Model's dynamics being called by anything other than the Solver Layer.
- **Rule:** `diffrax` is imported only inside `quantscenariobench.solver`. Every Market Model exposes its dynamics as `_drift(self, t, state) -> PyTree` and `_diffusion(self, t, state) -> PyTree` — an identical signature across all models, with no model-specific extra arguments (model parameters live on `self`, fixed at construction). The leading underscore marks these as Solver-Layer-internal: not part of the package's supported public surface, which is `simulate()` alone. A Market Model never constructs a `diffrax.Term`, `SaveAt`, or solver instance itself.

### AD-5 — Dataset export is generic over the `Scenario` schema

- **Binds:** Feature 4.4 (FR-12 through FR-15).
- **Prevents:** per-model export hooks reintroducing Market-Model-specific coupling into the export pipeline, which would violate FR-10's "zero changes to the export pipeline" guarantee.
- **Rule:** `quantscenariobench.export` derives Parquet columns by pytree-flattening a `Scenario` (per AD-2's dynamic/static split). It imports `quantscenariobench.interface` only — never a concrete Market Model from `quantscenariobench.models`.

### AD-6 — `equinox` is a project-wide pytree convention, not a diffrax-only dependency

- **Binds:** every pytree-bearing type in the project (Market Model configs, `Scenario`).
- **Prevents:** two different pytree mechanisms (hand-rolled `tree_util` registration vs. `eqx.Module`) coexisting in the same codebase.
- **Rule:** Every JAX-PyTree-typed dataclass in the project — Market Model parameter classes and `Scenario` alike — is an `equinox.Module`. None uses `jax.tree_util.register_pytree_node_class` directly.

### AD-7 — float64 (JAX x64) is the fixed v1 precision policy

- **Binds:** all Market Model simulation; FR-7/FR-8 correctness-tolerance checks; `quantscenariobench.solver`.
- **Prevents:** per-model float32/float64 divergence making cross-model comparison or closed-form tolerance checks meaningless; x64 silently never getting enabled because a caller imported a submodule directly instead of going through `quantscenariobench.api`.
- **Rule:** `jax.config.update("jax_enable_x64", True)` is called exactly once, as top-level code in `quantscenariobench/__init__.py`. Python import semantics guarantee the package `__init__` runs before any submodule's code, regardless of whether the caller imports `.api`, `.models`, or any other submodule directly — so this cannot be bypassed by import path. No Market Model or Solver Layer code overrides dtype per-call.

### AD-8 — Metadata's minimum guaranteed field set is fixed

- **Binds:** `Scenario.metadata`; dataset card generation (FR-15); resolves PRD Open Question 2.
- **Prevents:** two Market Models or two dataset exports independently choosing different provenance fields or representations, breaking FR-15's "every card has these fields" guarantee or Export's generic flattening (AD-5).
- **Rule:** `Scenario.metadata` always carries, at minimum: `seed`, `prng_key_info`, `model_name`, `model_version`, `parameters`, `time_grid`, `n_paths`, `library_version`, `dataset_version`, `generated_at`. `parameters` is always the Market Model's own parameter `eqx.Module` instance — never a hand-rolled dict or other ad hoc representation — so Export's pytree-flattening (AD-5) sees one consistent shape regardless of which Market Model produced it. A Market Model may not omit any field; it may not add a different name for any of them.

### AD-9 — Dependency direction is one-way: Models → Interface ← Solver/API/Export/Testing

- **Binds:** all modules (`quantscenariobench.*`).
- **Prevents:** a Market Model importing the Solver Layer, the public API, or Export directly — which would let a model bypass the State-Space Interface as the sole integration point; the conformance test suite (`testing`) importing concrete models or Export and so testing something other than the interface contract itself.
- **Rule:** see diagram. `equinox` is a project-wide third-party dependency (per AD-6) and may be imported by `interface`, `models`, and `solver` alike — it is not solver-exclusive. `diffrax` remains solver-exclusive: only `quantscenariobench.solver` imports it. Beyond that: a Market Model module may import only `quantscenariobench.interface` (+ `equinox`). `quantscenariobench.solver` may import `quantscenariobench.interface` (+ `equinox`, `diffrax`). `quantscenariobench.export` may import only `quantscenariobench.interface` (+ `equinox`) — never `models`, `solver`, or `testing`. `quantscenariobench.testing` may import only `quantscenariobench.interface` (+ `equinox`, test tooling) — never `models` or `export`; its dummy Market Model is defined inside `testing` itself, not borrowed from `models`. Only `quantscenariobench.api` may import concrete Market Models, and only as caller-supplied arguments — it never hardcodes a model name.

### AD-10 — Correctness-check references are independently implemented, never borrowed from a bundled quant library

- **Binds:** FR-7, FR-8, FR-9 correctness checks; `quantscenariobench.testing`.
- **Prevents:** a correctness check silently depending on `tf-quant-finance`, `QuantLib`, or another general-purpose quant library as its reference implementation — making the project's own pricing/sensitivity logic an unverified pass-through rather than an independent implementation (the brief's "not a thin wrapper" framing, addendum point 2).
- **Rule:** Every closed-form, semi-closed-form, or statistical reference value used to validate a Market Model (Black-Scholes analytic price, Heston characteristic-function price, rBergomi statistical/aBergomi-based sanity check) is implemented within QuantScenarioBench's own code. No correctness check imports a pricing formula from a third-party quant library as its source of truth.

### AD-11 — Public API stability follows semantic versioning, independent of dataset versioning

- **Binds:** `simulate()`, `Scenario`, the State-Space Interface (`MarketModel`, `TimeGrid`); resolves the PRD's Cross-Cutting NFR "Public API stability policy."
- **Prevents:** a breaking change to the public API shipping in a non-major library release; conflating library version bumps with dataset version bumps (FR-14 already fixes datasets as independently versioned — this AD is the library-side mirror of that same discipline).
- **Rule:** Any backward-incompatible change to `simulate()`'s signature, `Scenario`'s field set, or the `MarketModel`/`TimeGrid` contract requires a major version bump of the `quantscenariobench` package, tracked separately from any `dataset_version` value in `Metadata` (AD-8).

### AD-12 — `TimeGrid` is an explicit, ordered time-point sequence, not a generative spec

- **Binds:** FR-3; every Market Model's `_drift`/`_diffusion` signature (AD-4) and the Solver Layer's `TimeGrid`→`SaveAt` mapping (AD-4).
- **Prevents:** one Market Model accepting `TimeGrid` as a `(start, stop, steps)`-equivalent spec it expands internally, while another expects pre-expanded points — silently disagreeing on what `TimeGrid` "is."
- **Rule:** `TimeGrid` always carries an explicit, already-ordered array of time points (supporting non-uniform spacing per FR-3); no Market Model or the Solver Layer accepts or produces an alternate `(start, stop, steps)` representation.

```mermaid
graph LR
  models["models (BlackScholes, Heston, RoughBergomi)"] --> interface["interface (MarketModel, Scenario, TimeGrid)"]
  solver["solver (wraps diffrax)"] --> interface
  api["api (simulate)"] --> interface
  api --> solver
  export["export (Parquet/HF, generic)"] --> interface
  testing["testing (conformance suite + dummy model)"] --> interface
  caller(["caller code"]) --> api
  caller --> models
  caller --> export
```

## Consistency Conventions

| Concern | Convention |
| --- | --- |
| Naming (entities, files, interfaces) | Market Model classes are PascalCase nouns matching Glossary terms exactly (`BlackScholes`, `Heston`, `RoughBergomi`); the contract type is `MarketModel`; the public entrypoint is `simulate`; the Solver Layer's internal entrypoint is `solve_sde`. |
| Metadata field names | Fixed per AD-8: `seed`, `prng_key_info`, `model_name`, `model_version`, `parameters`, `time_grid`, `library_version`, `dataset_version`, `generated_at`. No Market Model introduces a synonym for any of these. |
| Soft validation (FR-6) | A single warning class (e.g. `QuantScenarioBenchValidationWarning`) is used for every research-constraint violation (Feller condition and future equivalents) across all Market Models — never a bare `UserWarning`, never a model-specific subclass. |
| State & cross-cutting (precision, randomness) | float64 fixed per AD-7; randomness construction lives only in the Solver Layer per AD-3/AD-4; no module outside `quantscenariobench.solver` touches a JAX `PRNGKey` split for simulation purposes. |

## Stack

| Name | Version |
| --- | --- |
| Python | >=3.11 (raised from PRD's [ASSUMPTION] of >=3.10 — driven by diffrax's minimum; PRD needs reconciling) |
| jax | >=0.4.38 (diffrax 0.7.2's pin) |
| diffrax | 0.7.2 |
| equinox | >=0.11.10 (diffrax 0.7.2's pin) |

## Structural Seed

```text
quantscenariobench/
  interface/       # MarketModel ABC, Scenario, TimeGrid — the only module every other module may depend on
  models/           # BlackScholes, Heston, RoughBergomi — each depends on interface only
  solver/           # internal Solver Layer — the only module that imports diffrax
  api/              # simulate() — depends on interface + solver; accepts any conforming Market Model
  export/           # Parquet / Hugging Face publishing — depends on interface only, generic over Scenario
  testing/          # State-Space Interface conformance suite + test-only dummy Market Model (FR-11)
```

## Capability → Architecture Map

| Capability / Area | Lives in | Governed by |
| --- | --- | --- |
| Feature 4.1 Core Simulation API (FR-1–FR-6) | `quantscenariobench.api`, `quantscenariobench.interface` | AD-1, AD-2, AD-3, AD-7, AD-11, AD-12 |
| Feature 4.2 v1 Market Model Zoo (FR-7–FR-9) | `quantscenariobench.models` | AD-1, AD-4, AD-6, AD-7, AD-10 |
| Feature 4.3 State-Space Extensibility Contract (FR-10–FR-11) | `quantscenariobench.interface`, `quantscenariobench.testing` | AD-1, AD-4, AD-9, AD-10 |
| Feature 4.4 Benchmark Dataset Export & Publishing (FR-12–FR-15) | `quantscenariobench.export` | AD-2, AD-5, AD-8, AD-11 |

## Deferred

- **Release/publish operational envelope** — PyPI release process for the library, and what triggers/authenticates a Hugging Face dataset publish. Not coached in this session (stated priority was the State-Space Interface and extension points); a real structural dimension, intentionally left open rather than silently decided.
- **Runtime/compute environment** — whether `simulate()` and dataset generation are expected to run on CPU, GPU, or TPU, and any associated infra/provider choice. GPU/TPU performance is explicitly a non-goal for v1 (PRD §5), but the baseline execution environment itself was not architected and is a real gap in the operational envelope, not just a performance question.
- **Dataset generation and hosting cost at scale** (PRD Open Question 7) — no budget, ceiling, or architectural mitigation (e.g. sampling fewer paths, sharding) has been decided.
- **Parquet row granularity** (PRD Open Question 4, FR-12) — AD-5 fixes that export is column-generic via pytree-flattening, but row granularity (one row per simulated path vs. one row per batch) was not decided here; one row per path is the natural lean for benchmark-dataset usability, not yet committed.
- **Dataset versioning scheme specifics** (PRD Open Question 3, FR-14) — AD-8 fixes that a `dataset_version` field exists in Metadata; the scheme that produces its value (semver-per-dataset vs. content-hash vs. other) is undecided.
- **Hugging Face organization/namespace and dataset naming convention** (PRD Open Question 5) — undecided.
- **Conformance test harness mechanism** (PRD FR-11 `[ASSUMPTION]`) — pytest-based fixtures are the natural default given the Python/pytest ecosystem, but not committed; property-based testing (e.g. `hypothesis`) remains an open alternative.
- **rBergomi statistical correctness test suite specifics** (PRD FR-9 `[ASSUMPTION]`) — aBergomi noted as a candidate reference technique in the PRD addendum; not architected here.
- **SABR, jump-diffusion, and any Market Model beyond the v1 three** — out of v1 scope per PRD; AD-1/AD-4/AD-9 are designed to make adding them a `quantscenariobench.models`-only change, but no second extensibility test beyond the FR-11 dummy model has been run.
- **Oracle label computation (AD-through-paths, Monte Carlo interim)** — out of v1 scope per PRD; AD-7's float64 policy and AD-4's Solver Layer boundary are compatible with adding this later, but no design work has been done on it.
- **Multi-asset / cross-asset simulation, CLI, real-market calibration** — out of v1 scope per PRD; not architected.
- **Open-source license** (PRD Open Question 1) — undecided.
- **CI/test-infrastructure scope** (PRD Open Question 6) — whether this belongs to product NFRs or pure dev-workflow scope is still unresolved; not architected either way.
