---
title: PRD - QuantScenarioBench
status: final
created: 2026-06-30
updated: 2026-07-03
---

# PRD: QuantScenarioBench

## 0. Document Purpose

This PRD defines the v1 functional and quality requirements for QuantScenarioBench, a finance-scoped, JAX-native Python framework for generating reproducible market scenarios, publishing them as versioned benchmark datasets, benchmarking portfolio strategies against those scenarios through a standardized metrics, baseline, and optimizer layer, and — as of this update — publishing those benchmark runs as versioned Evaluation Results that aggregate into a public Leaderboard. It builds directly on the finalized [Product Brief](../../briefs/brief-QuantScenarioBench-2026-06-30/brief.md) and its [Addendum](../../briefs/brief-QuantScenarioBench-2026-06-30/addendum.md) — read those first for the problem framing, landscape research, and rationale; this document does not repeat them. It is written for the author (also the primary user) and for downstream BMad workflows (architecture, epics/stories) that will derive implementation plans from it. Terms are Glossary-anchored (§3); features are grouped with Functional Requirements (FRs) nested under them and numbered globally; inferred content is tagged inline as `[ASSUMPTION]` and indexed in §9.

**Note on "Benchmark."** This PRD now uses "Benchmark" in two related but distinct senses: **Benchmark Dataset** (Feature 4.4, pre-existing) — a published, versioned dataset of Scenarios — and **Benchmark Runner** / **Benchmark Result** (Feature 4.8, new in the 2026-07-02 update) — the portfolio-strategy evaluation pipeline and its output. A Benchmark Runner consumes Scenarios (generated or from a Benchmark Dataset) as one of its inputs; the two capabilities are complementary, not the same thing.

**Note on "Benchmark Result" vs. "Evaluation Result."** *(added 2026-07-03)* A **Benchmark Result** (Feature 4.8) is the in-memory, runtime output of one `run_benchmark()` call — it is never itself published. An **Evaluation Result** (Feature 4.9, new in this update) is the published, canonical representation derived from a `BenchmarkResult` for the Hugging Face Hub and the Leaderboard — see AD-26. Every Evaluation Result comes from exactly one Benchmark Result; a Benchmark Result does not require publishing to be useful locally.

## 1. Vision

QuantScenarioBench gives a quantitative researcher one consistent way to generate market scenarios, regardless of which stochastic process produces them. A researcher writes the same `simulate()` call against Black-Scholes, Heston, or rough Bergomi, gets back a `Scenario` object with the same top-level shape every time, and never has to learn a model-specific simulation API to compare results across models.

The framework's defining bet is architectural, not algorithmic: every Market Model implements one State-Space Interface (`observation`, `latent_state`, `metadata`, with `randomness` optional), so the simulation core and the dataset export pipeline never need to change when a new model is added. v1 proves this contract with three models of meaningfully different mathematical character — Black-Scholes (closed-form), Heston (semi-closed-form), rough Bergomi (no closed form) — and ships the first versioned, Parquet-backed benchmark datasets to Hugging Face built on top of it.

This is a launch-grade specification for an intentionally small v1: the design must hold up to public scrutiny and to a second and third model being added later without rework, even though the first implementation stays narrow. As the brief states plainly, the bet here is execution and integration, not a novel algorithm — the stochastic models and the autodiff machinery already exist; the product is assembling them into one coherent, reusable, published toolkit.

Beyond v1, the same contract is meant to outlive the three models it ships with: a researcher who wants to contribute a new Market Model should be able to do so against the published State-Space Interface (Feature 4.3) without coordinating with the maintainer on simulation or export internals.

**Scope expansion (2026-07-02 update): from scenario generator to benchmarking framework.** v1 as originally scoped stopped at publishing scenario data. That update extended the vision one layer further: a researcher should be able to take those same scenarios — generated locally via `simulate()` or loaded from a published Benchmark Dataset — and evaluate a portfolio strategy against them using standardized, reproducible metrics, without hand-rolling backtest plumbing for every project. The same architectural bet that governs Market Models now governs portfolio strategies: a Portfolio Optimizer Interface (Feature 4.7) that any strategy — a fixed Traditional Baseline (Feature 4.6) or, longer-term, a learned forecasting model (e.g. PatchTST, iTransformer, TimeMixer) — implements once, so the Benchmark Runner (Feature 4.8) never needs strategy-specific logic. That was a deliberate scope expansion beyond the original v1 boundary, not a natural extension of scenario generation. It also knowingly superseded a line in the brief's own Vision, which framed future growth as strengthening the reference dataset zoo rather than redefining the framework itself — that update was exactly that redefinition, made with the user's explicit direction.

**Platform expansion (this update, 2026-07-03): from benchmarking framework to Hugging Face-native evaluation platform.** This update extends the vision a further layer: a `BenchmarkResult` should not be a dead-end local artifact. A researcher can publish it, in a fixed schema, as an Evaluation Result to a shared Hugging Face dataset repo, and see it aggregated — ranked against every other published strategy/dataset combination — on a public Leaderboard. This directly resolves the Non-Goal the 2026-07-02 update deliberately left in place ("hosting or publishing a benchmark leaderboard" — see the reversal in §5) and realizes the brief's own Platform expansion of its Vision (added 2026-07-03). Consistent with that brief update, this PRD scopes the *hosted Leaderboard UI* (a Hugging Face Space or equivalent, Feature 4.10) as an explicit future phase — Evaluation Results publishing and Leaderboard aggregation (Feature 4.9) are this update's committed scope; the Space is not part of this epic (see §5, §6.2).

## 2. Target User

### 2.1 Jobs To Be Done

- Generate reproducible synthetic market scenarios to train or evaluate pricing models, hedging strategies, or risk estimators — without hand-building a simulator per project.
- Compare a downstream model's behavior across stochastic processes with different mathematical character (closed-form vs. semi-closed-form vs. no closed form) through one consistent interface.
- Consume a standardized, versioned benchmark dataset directly from Hugging Face, without running simulation code locally.
- Extend the framework with a new Market Model by implementing the State-Space Interface alone, without touching simulation or export internals.
- Evaluate and compare portfolio allocation strategies — Traditional Baselines or a custom optimizer implementing the Portfolio Optimizer Interface — against generated or published scenarios, using standardized metrics, without hand-building backtest/evaluation plumbing per project. `[new — this update]`
- Plug a future forecasting model (e.g. a return-prediction network) into the benchmark pipeline as a `ForecastOptimizer`, and compare it against the Traditional Baselines on the same metrics and scenarios, without modifying the Benchmark Runner. `[new — 2026-07-02 update]`
- Publish a `BenchmarkResult` as a versioned Evaluation Result to a shared Hugging Face dataset repo, and see it aggregated into a ranked Leaderboard table alongside every other published strategy/dataset combination, without hand-building publishing, aggregation, or ranking code. `[new — this update]`
- *(secondary, beyond v1)* Adopt QuantScenarioBench's published datasets and Market Models as a shared community benchmark, rather than each researcher publishing one-off synthetic data alongside an individual paper.

### 2.2 Non-Users (v1)

- Live or production trading/execution systems — QuantScenarioBench simulates from specified parameters; it does not calibrate to or connect with real market data.
- Anyone needing correlated multi-asset scenario *generation* — a single `simulate()` call still produces one asset's path per Market Model; there is no joint/cross-asset stochastic process in v1. `[updated — this update]` Multi-asset *portfolio evaluation* is now in scope (Feature 4.6–4.8) by composing multiple independently generated or loaded single-asset Scenarios into a portfolio — see Feature 4.8 and the `[ASSUMPTION]` on FR-26.
- Anyone who needs oracle labels (theoretical pricing, hedging deltas) shipped with the v1 datasets — acknowledged as a goal, not delivered in v1 (see [Open Questions & Risks](../../briefs/brief-QuantScenarioBench-2026-06-30/brief.md#open-questions--risks) in the brief).
- Anyone needing a shipped, working forecasting model (e.g. PatchTST, iTransformer, TimeMixer) — v1 ships the `ForecastOptimizer` interface only, proven via a test-only dummy optimizer; no real forecasting model is a v1 deliverable. `[new — 2026-07-02 update]`
- Anyone needing a hosted, browsable Leaderboard *web UI* — Evaluation Results publishing and Leaderboard aggregation (Feature 4.9) are v1 deliverables; a Hugging Face Space (or equivalent) rendering them as a live page is a future phase (Feature 4.10), not part of this update. `[reversed — this update]` (Previously: "Anyone needing a hosted, public leaderboard — v1 produces a structured, JSON-serializable `BenchmarkResult` per run, suitable for future leaderboard use, but does not host or publish a leaderboard itself." That Non-User statement no longer holds for publishing/aggregation, which are now in scope — only the hosted UI remains out of this update.)
- CLI-only or non-Python workflows — v1 ships a Python API only.

### 2.3 Key User Journeys

*Single-persona, API-first, capability-driven product — journeys are kept to one-line scope per the Lighter dial rather than full named-persona narratives.*

- **UJ-1.** A researcher configures a `Heston` Market Model and calls `simulate()` with a seed to get a reproducible `Scenario` for local model training or evaluation. Realizes Feature 4.1.
- **UJ-2.** A researcher loads a published QuantScenarioBench dataset directly from the Hugging Face Hub to benchmark a model, without installing or running the simulation library. Realizes Feature 4.4.
- **UJ-3.** A contributor adds a new Market Model by implementing the State-Space Interface, with no changes to `simulate()` or the export pipeline. Realizes Feature 4.3.
- **UJ-4.** A researcher assembles a multi-asset portfolio from several generated or HF-loaded Scenarios, runs the Benchmark Runner with the Equal Weight, GMV, and CVaR Traditional Baselines, and gets back a `BenchmarkResult` per strategy — comparable via Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Final Wealth Factor — without writing any backtest code. Realizes Features 4.5–4.8. `[new — 2026-07-02 update]`
- **UJ-5.** A contributor implements a `ForecastOptimizer` wrapping a custom forecasting model, runs it through the same Benchmark Runner used for the Traditional Baselines, and gets a directly comparable `BenchmarkResult` with no changes to the Runner. Realizes Features 4.7 and 4.8. `[new — 2026-07-02 update]`
- **UJ-6.** A researcher publishes a `BenchmarkResult` from a completed run as an Evaluation Result to the shared QuantScenarioBench Evaluation Results repo on the Hugging Face Hub, and finds it correctly ranked, alongside every other published strategy/dataset combination, in the aggregated Leaderboard table. Realizes Feature 4.9. `[new — this update]`

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
- **Portfolio Weights** — A vector of allocation fractions across the assets in a portfolio (summing to 1, subject to any strategy-specific constraints, e.g. long-only), produced by a Traditional Baseline or a custom optimizer. `[new — this update]`
- **Portfolio Return** — The return series of a portfolio over time, computed by applying Portfolio Weights to the return series of its constituent assets. `[new — this update]`
- **Traditional Baseline** (or **Baseline Strategy**) — A fixed, non-learned portfolio allocation method (v1: Equal Weight, Global Minimum Variance, CVaR Optimization) that implements the `BaselineStrategy` half of the Portfolio Optimizer Interface, serving as a standardized comparison anchor for future forecasting/optimization models. `[new — this update]`
- **Forecast Optimizer** — A portfolio allocation method that additionally consumes a forecast (e.g. predicted returns from a forecasting model) alongside historical returns to produce Portfolio Weights, implementing the `ForecastOptimizer` half of the Portfolio Optimizer Interface. No concrete Forecast Optimizer ships in v1 — see Feature 4.7. `[new — this update]`
- **Portfolio Optimizer Interface** — The collective contract (`BaselineStrategy`, `ForecastOptimizer`) a portfolio allocation method must satisfy to plug into the Benchmark Runner without the Runner needing strategy-specific logic. Architecturally the portfolio-strategy analogue of the State-Space Interface. `[new — this update]`
- **Benchmark Runner** — The orchestration component that takes returns/Scenarios and a Portfolio Optimizer Interface implementation through the full pipeline (returns/scenarios → strategy → weights → portfolio returns → metrics) and produces a Benchmark Result. `[new — this update]`
- **Benchmark Result** — The structured, JSON-serializable output of one Benchmark Runner run: the evaluated strategy's identity, its Metrics values, and enough provenance to make the run reproducible and comparable to other runs (see FR-29). Distinct from a **Benchmark Dataset** (a published Scenario dataset) — see §0 note. `[new — 2026-07-02 update]`
- **Evaluation Result** — The published, canonical representation of one `BenchmarkResult`, derived from it via a fixed transform (see AD-26) and written to a shared Hugging Face Evaluation Results repo. Distinct from a Benchmark Result (in-memory, runtime, never itself published) — see §0 note. `[new — this update]`
- **Leaderboard** — The aggregated view over every published Evaluation Result: a ranked table (rows = strategy × Benchmark Dataset, columns = Metrics) built by reading the shared Evaluation Results repo. v1 (this update) ships the aggregation itself, not a hosted, browsable UI — see Feature 4.9/4.10. `[new — this update]`

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

### 4.5 Portfolio Performance Metrics

**Description:** *(new — this update)* The standardized measures every strategy is scored on. Computed from a Portfolio Return series only — the Metrics have no knowledge of how the returns were produced (Traditional Baseline, Forecast Optimizer, or otherwise). Realizes UJ-4, UJ-5.

**Functional Requirements:**

#### FR-16: Sharpe Ratio

Given a Portfolio Return series, compute its Sharpe Ratio using a risk-free rate of 0 and no annualization (v1 convention, confirmed by user).

**Consequences (testable):**
- Sharpe Ratio computed against a hand-derived reference return series matches within floating-point tolerance.
- A constant (zero-variance) Portfolio Return series returns a defined sentinel value (proposed default: `0.0`) rather than raising a division-by-zero exception. `[ASSUMPTION: proposed default, not yet confirmed by user]`

#### FR-17: Sortino Ratio

Given a Portfolio Return series, compute its Sortino Ratio using a risk-free rate of 0 (downside-deviation target of 0) and no annualization.

**Consequences (testable):**
- Sortino Ratio computed against a hand-derived reference return series matches within floating-point tolerance.
- A Portfolio Return series with no negative returns returns a defined sentinel value (proposed default: `0.0`) rather than raising, mirroring FR-16's zero-variance handling. `[ASSUMPTION: proposed default, not yet confirmed by user]`

#### FR-18: Maximum Drawdown

Given a Portfolio Return series, compute its Maximum Drawdown (the largest peak-to-trough decline in cumulative portfolio value over the series).

**Consequences (testable):**
- Maximum Drawdown computed against a hand-derived reference return series matches within floating-point tolerance.

#### FR-19: Final Wealth Factor

Given a Portfolio Return series, compute its Final Wealth Factor (the cumulative growth multiple of an initial unit investment over the full series).

**Consequences (testable):**
- Final Wealth Factor computed against a hand-derived reference return series matches within floating-point tolerance.

**Feature-specific NFRs:**
- All four v1 metrics are pure functions of a Portfolio Return series (no hidden state, no dependency on which strategy or Market Model produced it), so new metrics can be added later without touching the Benchmark Runner or Portfolio Optimizer Interface.
- Each metric has an automated correctness check against an independently hand-derived reference value, consistent with the correctness-testing convention established for Market Models (Feature 4.2).

### 4.6 Traditional Baseline Strategies

**Description:** *(new — this update)* Three fixed, non-learned allocation methods that give every future forecasting/optimization model something concrete to beat. Each implements the `BaselineStrategy` half of the Portfolio Optimizer Interface (Feature 4.7). Realizes UJ-4.

**Functional Requirements:**

#### FR-20: Equal Weight baseline

An `EqualWeight` `BaselineStrategy` allocates equal Portfolio Weights across all assets in the portfolio, independent of historical returns.

**Consequences (testable):**
- For an N-asset portfolio, every Portfolio Weight equals `1/N`.

#### FR-21: Global Minimum Variance (GMV) baseline

A `GlobalMinimumVariance` `BaselineStrategy` computes Portfolio Weights that minimize portfolio variance given the historical returns' covariance structure. `[ASSUMPTION: solver/dependency (e.g. closed-form via covariance inversion vs. a QP solver) not yet specified — Architecture-phase decision, see §8 Open Questions]`

**Consequences (testable):**
- Produced Portfolio Weights sum to 1.
- Produced Portfolio Weights achieve variance no greater than the Equal Weight baseline's on the same historical returns (sanity property, not a numerical-tolerance check).

#### FR-22: CVaR Optimization baseline

A `CVaROptimization` `BaselineStrategy` computes Portfolio Weights that minimize Conditional Value-at-Risk (CVaR) at a defined confidence level, given the historical returns. Confidence level default confirmed by the user, 2026-07-02: 95%. `[ASSUMPTION: solver/dependency not yet specified — Architecture-phase decision, see §8 Open Questions]`

**Consequences (testable):**
- Produced Portfolio Weights sum to 1.
- The confidence level used is recorded in the strategy's identity/parameters so a Benchmark Result is reproducible (see FR-29).

**Out of Scope:**
- Any baseline beyond these three (e.g. risk parity, maximum Sharpe/tangency portfolio) — deferred until the v1 three prove out the Portfolio Optimizer Interface, mirroring the Market Model Zoo's (Feature 4.2) narrow-v1-then-extend posture.

**Feature-specific NFRs:**
- All three baselines are reference implementations validated independently, not sourced from a bundled portfolio-optimization library taken as a black box — consistent with the correctness-testing convention in Feature 4.2's NFR.

### 4.7 Portfolio Optimizer Extensibility Contract

**Description:** *(new — this update)* The structural promise the benchmark layer is built on, mirroring Feature 4.3's role for Market Models: a new portfolio strategy — baseline or forecast-driven — can be added by implementing the Portfolio Optimizer Interface alone, with zero changes to the Benchmark Runner. Realizes UJ-5.

**Functional Requirements:**

#### FR-23: Public `BaselineStrategy` interface

A documented interface specifies exactly what a `BaselineStrategy` must implement: consuming historical returns and producing Portfolio Weights (`historical_returns → weights`), usable by the Benchmark Runner (Feature 4.8) without Runner changes.

**Consequences (testable):**
- All three v1 Traditional Baselines (FR-20–FR-22) conform to this interface.
- A new `BaselineStrategy` implementation requires zero changes to the Benchmark Runner's source.

#### FR-24: Public `ForecastOptimizer` interface

A documented interface specifies exactly what a `ForecastOptimizer` must implement: consuming historical returns plus an externally supplied forecast and producing Portfolio Weights (`historical_returns + forecast → weights`), usable by the Benchmark Runner without Runner changes. No concrete `ForecastOptimizer` implementation (e.g. wrapping PatchTST, iTransformer, or TimeMixer) is a v1 deliverable — see §6.2.

**Consequences (testable):**
- A `ForecastOptimizer` implementation requires zero changes to the Benchmark Runner's source.

#### FR-25: Portfolio Optimizer conformance test suite

A reusable test harness verifies that a given `BaselineStrategy` or `ForecastOptimizer` implementation satisfies the Portfolio Optimizer Interface (correct Portfolio Weight shape/normalization, deterministic output for deterministic input). The suite includes a minimal test-only dummy `ForecastOptimizer` that exists only inside the test suite — never published, never part of the public framework surface — used specifically to prove the interface holds independent of any real forecasting model, mirroring FR-11's dummy Market Model. `[ASSUMPTION: exact test harness mechanism not yet specified, same open point as FR-11]`

**Consequences (testable):**
- Running the conformance suite against any v1 Traditional Baseline passes.
- The test-only dummy `ForecastOptimizer` passes the conformance suite and runs successfully through the Benchmark Runner with no modification to the Runner's source, proving the interface is satisfiable by an implementation the Runner was never written against.

**Out of Scope:**
- The dummy `ForecastOptimizer` is not a shipped deliverable — not published, not part of the public API, mirroring FR-11's dummy Market Model treatment.
- Any real forecasting-model integration (PatchTST, iTransformer, TimeMixer, or otherwise) — named here only as the motivating future use case for this contract, per the user's guidance; not a v1 deliverable.

### 4.8 Benchmark Runner & Results

**Description:** *(new — this update)* Turns scenarios/returns plus a Portfolio Optimizer Interface implementation into a comparable, reproducible result. Realizes UJ-4 and UJ-5. One Benchmark Runner call, one `BenchmarkResult`, regardless of which strategy was evaluated — the same architectural posture as `simulate()` (Feature 4.1) being agnostic to which Market Model it's given.

**Functional Requirements:**

#### FR-26: Multi-asset portfolio composition from single-asset scenarios

The Benchmark Runner accepts a portfolio's constituent assets as multiple single-asset return series, each sourced independently from either a `simulate()`-generated Scenario or a Scenario loaded from a published Benchmark Dataset (Feature 4.4), and assembles them into one multi-asset returns matrix for strategy evaluation. Multi-asset portfolios are composed by combining independently generated/loaded single-asset Scenarios; no correlated multi-asset generation is added to the simulation core (Feature 4.1–4.3 remain single-asset per `simulate()` call, per the updated Non-Goal in §5). Confirmed by the user on 2026-07-02; this remains the layer's central design bet and is worth flagging to Architecture, but is no longer an open question.

**Consequences (testable):**
- A portfolio of N assets can be assembled from N independently generated or loaded single-asset Scenarios (or a mix of both sources) without any change to `simulate()`, a Market Model, or the export/load path.

#### FR-27: End-to-end orchestration

The Benchmark Runner executes the full pipeline for a given strategy: `returns/scenarios → strategy → portfolio weights → portfolio returns → metrics → BenchmarkResult`, calling the strategy's `BaselineStrategy`/`ForecastOptimizer` interface (Feature 4.7) and all four v1 metrics (Feature 4.5). When the input is a Scenario rather than a raw returns array, returns are derived per FR-28's conversion convention. Confirmed by the user on 2026-07-02: a strategy is fit once per Benchmark Runner call, on a historical/training return window preceding the evaluated window (both drawn from the same Scenario), and its resulting Portfolio Weights are held constant (buy-and-hold) over the full evaluated window; v1 does not rebalance mid-evaluation.

**Consequences (testable):**
- Running the same strategy against the same returns/scenarios twice produces an identical `BenchmarkResult` (determinism, mirroring FR-4's reproducibility guarantee for `simulate()`).
- The Runner contains no strategy-specific branching — strategy behavior lives entirely behind the Portfolio Optimizer Interface (Feature 4.7), mirroring FR-1's equivalent guarantee for Market Models.
- A strategy's Portfolio Weights, once computed, do not change within a single evaluated window (no intra-run rebalancing in v1).

#### FR-28: Return series derivation from Scenario observation

The Benchmark Runner derives a return series from a Scenario's `observation` (price path) before passing it to a strategy or metric, using one consistent convention across all v1 Market Models, Traditional Baselines, and data sources (`simulate()` output or a loaded Benchmark Dataset). Confirmed by the user, 2026-07-02: simple/arithmetic period return (not log return), sampled once per `TimeGrid` step.

**Consequences (testable):**
- The same return-derivation convention applies whether `observation` comes from a `simulate()` call or a loaded Benchmark Dataset.
- Two Scenarios with identical `observation` paths always produce identical return series.

#### FR-29: JSON-serializable Benchmark Result

Each Benchmark Runner run produces a `BenchmarkResult` that serializes losslessly to JSON, containing at minimum: the evaluated strategy's identity and parameters, the four v1 Metrics values (FR-16–FR-19), and enough provenance (constituent assets/Scenarios used, timestamp) to reproduce the run. `[ASSUMPTION: this minimum field list is inferred, mirroring FR-15's dataset-card minimum-fields pattern; exact schema not yet specified — see §8 Open Questions]`

**Consequences (testable):**
- A `BenchmarkResult` round-trips through `json.dumps`/`json.loads` (or equivalent) without loss.
- Every field in the minimum list above is present; a result missing any one of them fails review, mirroring FR-15's dataset-card review gate.

**Out of Scope:**
- Publishing a `BenchmarkResult` anywhere beyond the local process — see Feature 4.9, new in this update, for that capability. `[updated — this update]` (Previously: "Publishing or hosting `BenchmarkResult`s as a leaderboard — v1 produces the JSON artifact only; hosting is acknowledged as a future direction, not committed." Publishing and Leaderboard aggregation are now in scope; only the hosted UI remains a future direction — see Feature 4.9/4.10, §5, §6.2.)

**Feature-specific NFRs:**
- The Benchmark Runner is generic over the Portfolio Optimizer Interface, the same way the export pipeline (Feature 4.4, AD-5) is generic over the Scenario schema — it never imports a concrete strategy.

### 4.9 Evaluation Results & Leaderboard

**Description:** *(new — this update)* Turns a `BenchmarkResult` from a dead-end local artifact into a published, comparable one. Realizes UJ-6. Mirrors Feature 4.4's role for Scenarios (Parquet + dataset card + Hugging Face publish) one layer up: `EvaluationResult` is the canonical published representation of a `BenchmarkResult` (AD-26), and the Leaderboard is a generic aggregation over every published `EvaluationResult` — never a hosted UI in this update (that is Feature 4.10, a future phase; see §5, §6.2).

**Functional Requirements:**

#### FR-30: `EvaluationResult` schema

A fixed, JSON-native `EvaluationResult` schema, derived from `BenchmarkResult` (AD-26), carries at minimum the strategy's identity and parameters, the constituent Benchmark Dataset/Scenario identity, its Metrics (as an ordered list of `{name, value}` records — not `BenchmarkResult`'s flat dict, matching the Hugging Face `model-index.results[].metrics[]` convention consumed by Hub rendering and by Leaderboard aggregation), and publication provenance (`schema_version`, `result_id`, `library_version`, `generated_at`). `[ASSUMPTION: exact optional-field set — e.g. environment/backend info, tags, a `verified` flag — inferred from Hugging Face ecosystem convention, not yet confirmed by user; see §9]`

**Consequences (testable):**
- An `EvaluationResult` round-trips through `json.dumps`/`json.loads` without loss, the same guarantee FR-29 gives `BenchmarkResult`.
- Every field in the minimum list above is present; a result missing any one of them fails review, mirroring FR-29's and FR-15's review gates.
- `EvaluationResult`'s `metrics` field is a list of `{name, value}` records; a producer that instead emits `BenchmarkResult`'s flat dict shape fails review.

#### FR-31: `BenchmarkResult` → `EvaluationResult` transformation

A single, pure function converts a `BenchmarkResult` into an `EvaluationResult` (AD-26). The conversion never mutates or subclasses `BenchmarkResult`, and requires no changes to `run_benchmark()` or the Benchmark Runner (Feature 4.8) — it operates only on a `BenchmarkResult` already produced.

**Consequences (testable):**
- Calling the transform twice on the same `BenchmarkResult` produces identical `EvaluationResult`s (determinism, mirroring FR-27's guarantee for `BenchmarkResult` itself).
- `BenchmarkResult`'s own schema (FR-29, AD-24) is unchanged by this feature — the transform is additive, not a replacement.

#### FR-32: Local Evaluation Results storage

An `EvaluationResult` can be written to a local file layout organized by Benchmark Dataset and strategy (one JSON file per run, timestamped, never overwritten), mirroring the Parquet-data/rendered-card separation already established for Benchmark Dataset export (Feature 4.4). `[ASSUMPTION: exact directory/filename convention not yet specified — see §9]`

**Consequences (testable):**
- Writing two `EvaluationResult`s for the same strategy/dataset combination produces two files, not one overwritten file — an append-only history is preserved.
- The local layout requires no restructuring to become the layout uploaded in FR-33 — the same directory tree walks onto the Hub unchanged.

#### FR-33: Hugging Face Evaluation Results publishing

An `EvaluationResult` (or a batch of them) can be published to a shared Hugging Face dataset repo dedicated to Evaluation Results — distinct from the per-Market-Model Benchmark Dataset repos of Feature 4.4 — with an auto-generated README/card summarizing its contents, mirroring `generate_dataset_card`'s role for Benchmark Datasets (FR-15).

**Consequences (testable):**
- Publishing an `EvaluationResult` to the shared repo does not overwrite any previously published result — the repo accumulates an append-only history, consistent with FR-32's local convention.
- The shared repo's README reflects the current set of published results after each publish.

#### FR-34: Leaderboard aggregation

A generic reader loads every published `EvaluationResult` from the shared Evaluation Results repo (or a local equivalent) and builds a ranked table — rows keyed by strategy × Benchmark Dataset, columns keyed by Metric name — with no strategy-specific or dataset-specific branching, the same generic posture Feature 4.5's Metrics and Feature 4.8's Runner already hold.

**Consequences (testable):**
- Adding a newly published `EvaluationResult` for a strategy/dataset combination not previously seen adds a new row to the aggregated table with zero code changes.
- The aggregation function returns a plain, headlessly-testable table/data structure — it has no dependency on any UI framework (see Out of Scope).

**Out of Scope:**
- A hosted, browsable Leaderboard web UI (a Hugging Face Space, Gradio/Streamlit app, or equivalent) — the aggregation in FR-34 is UI-agnostic by design; rendering it as a live page is Feature 4.10, an explicit future phase, not part of this update.
- An external/community submission workflow or write-access model for the shared Evaluation Results repo — v1 publishing (FR-33) is maintainer-driven; who else, if anyone, can publish is undecided (see §8 Open Questions).
- An automated "verified" re-run/reproduction workflow — `EvaluationResult`'s optional `verified` field (FR-30) may exist as a value, but no process re-runs and confirms it in this update.
- Historical/trend views (e.g. a strategy's metric value over successive library versions) — the Leaderboard is a current-snapshot table in this update, not a time series.

**Feature-specific NFRs:**
- `EvaluationResult` production (FR-30, FR-31) and Leaderboard aggregation (FR-34) are pure, side-effect-free functions over already-produced `BenchmarkResult`/`EvaluationResult` data — neither requires a network call, mirroring how Metrics (Feature 4.5) are pure functions of a Portfolio Return series.

## 5. Non-Goals (Explicit)

- Calibrating Market Models to live or historical market data — v1 simulates from specified parameters only.
- Correlated multi-asset scenario *generation* — `simulate()` and every Market Model (Feature 4.1–4.3) remain single-asset per call; there is no joint stochastic process across assets in v1. `[narrowed — this update]` (Previously stated as a blanket "multi-asset ... v1 is single-asset only." Multi-asset *portfolio evaluation* is now in scope — see Feature 4.6–4.8 — by composing independently generated/loaded single-asset Scenarios, not by generating correlated paths.)
- A CLI — v1 ships a Python API only, until a clear need for a CLI emerges.
- Oracle labels (theoretical pricing, hedging deltas) as a shipped dataset feature — acknowledged as the long-term direction (AD-through-paths, with Monte Carlo as an accepted interim), not committed to v1.
- GPU/TPU performance optimization as a goal in itself — JAX provides this where used, but it is not a v1 design target, and Success Metrics (§7) explicitly do not reward optimizing for it.
- Evaluating the BMAD + Claude Code AI development workflow — a real motivation for building this project (see brief addendum), but a development-process objective, not a product requirement.
- Shipping a real forecasting-model integration (PatchTST, iTransformer, TimeMixer, or otherwise) — v1 ships the `ForecastOptimizer` interface only, proven via a test-only dummy optimizer (FR-25). `[new — 2026-07-02 update]`
- Hosting a Leaderboard *web UI* (a Hugging Face Space, Gradio/Streamlit app, or equivalent) — `[reversed — this update]` (Previously: "Hosting or publishing a benchmark leaderboard — v1 produces a structured, JSON-serializable `BenchmarkResult` per run (FR-29); leaderboard hosting is a post-v1 direction." Publishing `BenchmarkResult`s as versioned Evaluation Results and aggregating them into a ranked Leaderboard table are now in scope — Feature 4.9, FR-30 through FR-34. Only the hosted, browsable UI remains out of scope, deferred to Feature 4.10, a future phase — not because it's a lesser priority, but because it is a live, ongoing-cost service rather than a one-time publishing artifact, a distinction worth keeping explicit for a solo-maintainer project (see brief Open Questions & Risks).)

## 6. MVP Scope

### 6.1 In Scope

- `simulate()` API, `Scenario` object, `TimeGrid` object (FR-1 through FR-5).
- Soft parameter validation (FR-6).
- Black-Scholes, Heston, and rough Bergomi Market Models (FR-7 through FR-9).
- The public State-Space Interface and its conformance test suite, including a test-only dummy Market Model used solely to prove extensibility (FR-10, FR-11).
- Parquet export and per-model Hugging Face dataset publishing, independently versioned, with dataset cards (FR-12 through FR-15).
- `[ASSUMPTION]` Documentation sufficient for an external researcher to generate or consume a dataset without reading the source: a README quickstart, an API reference, and one runnable example notebook per v1 Market Model (carried from the brief's Scope; this specific checklist is inferred).
- Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Final Wealth Factor metrics (FR-16 through FR-19). `[new — 2026-07-02 update]`
- Equal Weight, Global Minimum Variance, and CVaR Optimization Traditional Baselines (FR-20 through FR-22). `[new — 2026-07-02 update]`
- The public Portfolio Optimizer Interface (`BaselineStrategy`, `ForecastOptimizer`) and its conformance test suite, including a test-only dummy `ForecastOptimizer` used solely to prove extensibility (FR-23 through FR-25). `[new — 2026-07-02 update]`
- The Benchmark Runner, multi-asset portfolio composition from single-asset Scenarios, and the JSON-serializable `BenchmarkResult` (FR-26 through FR-29). `[new — 2026-07-02 update]`
- The `EvaluationResult` schema, the `BenchmarkResult` → `EvaluationResult` transform, local Evaluation Results storage, Hugging Face Evaluation Results publishing, and Leaderboard aggregation (FR-30 through FR-34). `[new — this update]`

### 6.2 Out of Scope for MVP

*Kept narrow deliberately: this is currently a one-person open-source effort, and a multi-model roadmap is ambitious against that bandwidth. Proving the interface on three models before expanding manages that risk rather than ignoring it.*

- SABR, jump-diffusion, and any further Market Model beyond the v1 three — deferred until the v1 interface proves itself.
- Oracle labels in published datasets — deferred; direction acknowledged (§ brief Solution), implementation not committed.
- CLI, correlated multi-asset scenario *generation*, real-market calibration — see §5 Non-Goals.
- GPU/TPU performance tuning as a dedicated workstream.
- Any Traditional Baseline beyond the v1 three, and any shipped `ForecastOptimizer` implementation (PatchTST, iTransformer, TimeMixer, or otherwise) — deferred until the v1 Portfolio Optimizer Interface proves itself, same posture as the Market Model Zoo. `[new — 2026-07-02 update]`
- A hosted, browsable Leaderboard web UI (Feature 4.10) — Evaluation Results publishing and Leaderboard aggregation (Feature 4.9) are in scope this update; the Space/UI itself is an explicit future phase. `[reversed — this update]` (Previously: "A hosted/public benchmark leaderboard — v1 produces the `BenchmarkResult` JSON artifact only.")
- An external/community submission workflow for the shared Evaluation Results repo, an automated "verified" reproduction workflow, and historical/trend Leaderboard views — see Feature 4.9's Out of Scope. `[new — this update]`

## Cross-Cutting NFRs

- **Determinism is backend-scoped.** Reproducibility (FR-4) holds for repeated runs on the same backend; cross-backend (CPU/GPU/TPU) bit-identity is explicitly not guaranteed, consistent with the brief's Open Questions & Risks.
- **JAX-native computation.** All Market Model simulation logic is implemented in JAX (jit/vmap-compatible), consistent with the framework's core identity. The choice of underlying SDE-integration machinery (e.g. building on an existing JAX differential-equation solver vs. implementing one) is an Architecture-phase decision — see Open Question 8. This guarantee is scoped to Market Model simulation logic (Features 4.1–4.3); it does not extend to the Traditional Baselines' solvers (Feature 4.6) — GMV/CVaR Optimization may depend on a non-JAX numerical solver (Open Question 10) without violating this guarantee. `[new — this update]`
- **Numerical correctness testing.** Every v1 Market Model has an automated correctness check appropriate to its mathematical character (closed-form comparison, semi-closed-form comparison, or statistical sanity check) — see FR-7 through FR-9. The same convention extends to the v1 Metrics (FR-16–FR-19) and Traditional Baselines (FR-20–FR-22): each is checked against an independently derived reference value. `[extended — this update]`
- **Public API stability policy.** Backward-incompatible changes to `simulate()`, the `Scenario` schema, or the State-Space Interface require a major version bump. `[ASSUMPTION: semantic versioning assumed; not explicitly confirmed]` This policy extends to the Portfolio Optimizer Interface and the `BenchmarkResult` schema. `[extended — this update]`
- **Language/runtime targets.** Python >=3.11, jax >=0.4.38 (confirmed during Architecture: driven by diffrax 0.7.2's minimum requirements — see [Architecture Spine](../../architecture/architecture-QuantScenarioBench-2026-06-30/ARCHITECTURE-SPINE.md) Stack table). Supersedes this PRD's original `[ASSUMPTION: Python 3.10+]`.
- **Benchmark Result reproducibility and serializability.** A `BenchmarkResult` must be losslessly JSON-serializable (FR-29) and deterministic for identical strategy/returns input (FR-27), the portfolio-benchmarking analogue of FR-4's reproducibility guarantee for `simulate()`. `[new — 2026-07-02 update]`
- **Evaluation Result reproducibility and serializability.** *(new — this update)* An `EvaluationResult` must be losslessly JSON-serializable (FR-30) and deterministic for an identical input `BenchmarkResult` (FR-31) — the publication-layer analogue of the Benchmark Result guarantee above, one layer up.

## 7. Success Metrics

**Primary**
- **SM-1**: API stability under extension — the test-only dummy Market Model (FR-11) passes the conformance suite with zero changes to `simulate()`, `Scenario`, or the export pipeline source. Validates FR-10, FR-11.
- **SM-2**: Reproducibility — a fixed-seed `simulate()` call produces a bit-identical Scenario on repeated runs on the same backend. Validates FR-4.
- **SM-3**: Dataset usability — all three v1 Benchmark Datasets load successfully via `datasets.load_dataset(...)` and conform to the documented shared schema. Validates FR-12, FR-13, FR-15.
- **SM-6** *(new — 2026-07-02 update)*: Portfolio Optimizer Interface stability under extension — the test-only dummy `ForecastOptimizer` (FR-25) passes the conformance suite and runs through the Benchmark Runner with zero changes to the Runner's source. Validates FR-23–FR-25, the benchmark-layer analogue of SM-1.

**Secondary**
- **SM-4**: Model correctness — Black-Scholes and Heston Scenarios match their closed-form/semi-closed-form references within tolerance; rBergomi Scenarios pass their statistical sanity checks. Validates FR-7, FR-8, FR-9.
- **SM-5** *(post-v1, tracked not gated)*: External usage signals — stars, downloads, forks, citations — as evidence of adoption beyond internal use.
- **SM-7** *(new — 2026-07-02 update)*: Metrics and baseline correctness — all four v1 Metrics and all three Traditional Baselines match independently hand-derived reference values within tolerance. Validates FR-16–FR-22.
- **SM-8** *(new — 2026-07-02 update)*: Benchmark Result usability — a `BenchmarkResult` round-trips through JSON serialization without loss and contains every required field (FR-29). Validates FR-26–FR-29.
- **SM-9** *(new — this update)*: Evaluation Results and Leaderboard correctness — an `EvaluationResult` round-trips through JSON serialization without loss and contains every required field (FR-30); Leaderboard aggregation (FR-34) correctly adds a new row for a newly published strategy/dataset combination with zero code changes. Validates FR-30–FR-34.

**Counter-metrics (do not optimize)**
- **SM-C1**: Raw simulation throughput/performance. Optimizing this at the expense of API simplicity or the State-Space Interface's stability would undermine SM-1 — counterbalances any temptation to hand-tune for speed during v1. Counterbalances SM-1.
- **SM-C2**: Model count. Adding Market Models faster than the interface can absorb them without modification would falsify SM-1 even while looking like progress. Counterbalances SM-1, SM-4.
- **SM-C3** *(new — this update)*: Baseline strategy backtested performance. The Traditional Baselines exist as fixed, standardized comparison anchors — tuning them to maximize their own backtested return/Sharpe would defeat their purpose as a stable reference point for evaluating future forecasting models. Counterbalances SM-7.

## 8. Open Questions

1. Open-source license — not yet chosen (brief states "fully open source" but no specific license).
2. Exact `Metadata` field list (FR-4) — seed/PRNG info confirmed; full schema (model identity, parameter values, TimeGrid reference, library version, timestamp) not yet confirmed.
3. Dataset versioning scheme specifics (FR-14) — semver-per-dataset vs. content-hash vs. another scheme.
4. Parquet row granularity (FR-12) — one row per path vs. one row per batch/run.
5. Hugging Face organization/namespace and dataset naming convention (FR-13).
6. Whether CI/test-infrastructure requirements belong in this PRD as NFRs, or are entirely development-workflow scope (per the brief's exclusion of the AI-dev-workflow motivation from product success criteria) and therefore out of this document.
7. Dataset generation and hosting cost at scale — flagged as an unresolved risk in the brief; no budget or ceiling defined yet.
8. What QuantScenarioBench builds its SDE/path-integration machinery on — the brief's addendum names `diffrax` as the most likely foundation, and notes it would be additive rather than a wrapper around an existing common-API library like `tf-quant-finance`. This PRD deliberately leaves the choice to the Architecture phase, since it is an implementation decision, not a capability requirement — see [PRD addendum](addendum.md) for the full landscape context carried forward from the brief.
9. ~~Whether multi-asset portfolios should be composed from independently generated/loaded single-asset Scenarios, or the simulation core should gain true correlated multi-asset generation.~~ — **Resolved, confirmed by user 2026-07-02**: composed from independently generated/loaded single-asset Scenarios (FR-26); `simulate()` stays single-asset. Still the layer's central design bet, worth flagging to Architecture, but no longer open.
10. GMV and CVaR Optimization solver/dependency choice (e.g. closed-form covariance inversion, `scipy.optimize`, a QP library such as `cvxpy`) — deferred to Architecture, mirroring Open Question 8's treatment of the SDE-solver choice. Bundled with this is a product-facing modeling decision, not just an implementation one: whether v1 Portfolio Weights are constrained long-only (no negative weights) or allow short positions — this affects what a "weight" can be, not only which solver computes it.
11. ~~CVaR Optimization confidence level default~~ — **Resolved, confirmed by user 2026-07-02** (FR-22): 95%.
12. ~~Portfolio allocation timing~~ — **Resolved, confirmed by user 2026-07-02** (FR-27): a strategy is fit once per Benchmark Runner call and its Portfolio Weights held constant (buy-and-hold) over the full evaluated window; v1 does not rebalance at points along the `TimeGrid`.
13. ~~Historical-returns window sourcing~~ — **Resolved, confirmed by user 2026-07-02** (FR-27): the "historical returns" used to fit a strategy are a lookback window preceding the window being evaluated, both drawn from the same Scenario (not a separate held-out Scenario).
14. Exact `BenchmarkResult` JSON schema beyond FR-29's stated minimum field list — not yet specified.
15. Portfolio Optimizer Interface's exact typing mechanism (e.g. an `equinox.Module` ABC mirroring AD-1's treatment of `MarketModel`, vs. a plain Python Protocol) — deferred to Architecture.
16. ~~Return-series derivation convention from `Scenario.observation`~~ — **Resolved, confirmed by user 2026-07-02** (FR-28): simple/arithmetic return, sampled once per `TimeGrid` step.
17. *(new — this update)* `EvaluationResult`'s full optional-field set (FR-30) — required fields mirror `BenchmarkResult`'s (AD-24, AD-26); optional fields (environment/backend info, tags, a `verified` flag) are proposed by analogy to Hugging Face `model-index` convention, not yet confirmed by user.
18. *(new — this update)* Whether the shared Evaluation Results repo (FR-33) is one repo for all Benchmark Datasets/strategies, or split per Benchmark Dataset — this PRD assumes one shared repo (needed for cross-dataset Leaderboard comparison, FR-34) but the exact repo-naming/namespace convention is undecided, mirroring Open Question 5's unresolved Hugging Face namespace question for Benchmark Datasets.
19. *(new — this update)* Who, if anyone besides the maintainer, can publish an Evaluation Result to the shared repo (write-access/auth model) — explicitly out of scope for this update (Feature 4.9 Out of Scope) but flagged as a real open question for the eventual Feature 4.10 phase.

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
- ~~FR-26~~ — resolved: multi-asset portfolios are composed from independently generated/loaded single-asset Scenarios, confirmed by user 2026-07-02 (see Open Question 9); no longer an open assumption, though still the layer's central design bet.
- FR-16/FR-17 — zero-variance / no-downside edge-case behavior for Sharpe/Sortino not yet specified; a defined sentinel value (e.g. 0.0) rather than raising is the likely v1 default, but not yet confirmed.
- FR-21 — GMV solver/dependency not yet specified (Open Question 10).
- ~~FR-22 — CVaR confidence level~~ — resolved: 95%, confirmed by user 2026-07-02 (Open Question 11). Solver/dependency and long-only-vs-short constraint still not specified at the PRD level (Open Question 10, deferred to Architecture).
- FR-25 — Portfolio Optimizer conformance test harness mechanism not yet specified, same open point as FR-11.
- ~~FR-27~~ — resolved: allocation timing and historical-returns window sourcing (static buy-and-hold, same-Scenario lookback window) confirmed by user 2026-07-02 (see Open Questions 12, 13); no longer an open assumption.
- ~~FR-28 — return-series derivation convention~~ — resolved: simple/arithmetic return, per-`TimeGrid`-step, confirmed by user 2026-07-02 (Open Question 16).
- FR-29 — `BenchmarkResult` minimum field list inferred from FR-15's dataset-card pattern, not explicitly specified (Open Question 14).
- FR-30 — `EvaluationResult`'s optional-field set inferred from Hugging Face `model-index` convention, not yet confirmed by user (Open Question 17).
- FR-32 — exact local file/directory naming convention for Evaluation Results not yet specified.
- FR-33 — shared-repo-vs-per-dataset-repo layout and Hugging Face namespace convention for Evaluation Results not yet specified (Open Question 18).
