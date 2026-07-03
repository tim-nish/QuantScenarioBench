---
title: Product Brief - QuantScenarioBench
status: final
created: 2026-06-30
updated: 2026-07-03
---

# Product Brief: QuantScenarioBench

## Executive Summary

QuantScenarioBench is a finance-scoped, open-source, JAX-native Python framework for generating reproducible market scenarios across multiple stochastic models. The framework is the core product: every model — starting with Black-Scholes, Heston, and rough Bergomi (rBergomi), and later SABR and jump-diffusion — is expressed through one common state-space interface: `randomness`, `latent_state`, `observation`, `metadata`. That shared shape is the project's core bet: it lets a researcher swap the underlying stochastic process without rewriting the simulation harness, and lets the same export pipeline turn any model's output into a versioned, reusable artifact.

Publishing those scenarios as benchmark datasets on Hugging Face for quantitative finance research is a primary deliverable built on top of that framework — downstream of it, not a separate product.

To the best of our knowledge, no existing library combines JAX-native automatic differentiation, a model-family-spanning common API, and published reusable benchmark datasets with ground-truth ("oracle") labels. The closest adjacent tools — `diffrax` (a general-purpose differentiable SDE solver with no finance models or state-space convention) and `tf-quant-finance` (a comparable common-API idea, but TensorFlow-based and not packaged as datasets) — leave this combination open. QuantScenarioBench is scoped to fill it, deliberately starting narrow: a working v1 framework and one published dataset, not a finished model zoo.

## The Problem

Quant researchers who want to benchmark models — pricing networks, hedging agents, risk estimators — against synthetic market data today stitch together model-specific simulators by hand: a Black-Scholes path generator here, a bespoke Heston or rough-volatility implementation there, each with its own API, its own randomness handling, and no shared schema. None of this is typically published as a reusable, versioned artifact, so every project re-derives both the simulation code and any "ground truth" (theoretical price, hedging ratio) it needs to evaluate against. There is no equivalent, for market scenario generation, of the standardized benchmark datasets that other ML subfields take for granted.

This also means there is no straightforward way to compare a model across stochastic processes with meaningfully different mathematical character — Black-Scholes and Heston admit closed-form or semi-closed-form pricing, while rough-volatility models like rBergomi do not, forcing researchers toward Monte Carlo or worse, toward skipping ground-truth comparisons entirely.

## The Solution

A Python/JAX framework that implements each stochastic market model behind one common state-space interface (`randomness`, `latent_state`, `observation`, `metadata`), generates reproducible scenarios from it, and exports them as benchmark datasets to the Hugging Face Hub. "Reproducible" means a fixed seed deterministically reproduces a scenario on a given backend [ASSUMPTION — scope discussed under Open Questions & Risks].

Longer term, published datasets carry oracle labels — theoretical pricing and "perfect" hedging deltas — for supervised learning and benchmarking. The brief intentionally does not prescribe how those labels get computed: for models with closed-form or semi-closed-form pricing (Black-Scholes, Heston), classical formulas apply directly; where no closed form exists (rBergomi and similar), the long-term direction is automatic differentiation through the simulated paths themselves — a technique JAX is unusually well suited for and that the literature treats as a credible substitute for closed-form sensitivities. Monte Carlo-based labels are an acceptable interim oracle where appropriate. This is a design space the project will resolve as it goes, not a commitment made in this brief.

## What Makes This Different

To the best of our knowledge, the project's case isn't a single feature — it's a combination nothing else currently offers:

- **JAX-native, not adapted.** `diffrax` already provides differentiable, GPU-capable SDE solving — QuantScenarioBench is expected to build on it rather than reimplement SDE integration [ASSUMPTION], but adds the finance-specific layer diffrax doesn't have: named models, a shared state-space schema, and a dataset export path.
- **One API across models with different mathematical character.** `tf-quant-finance` proves the "common interface across stochastic processes" idea works, but it's TensorFlow, and it stops at simulation — it doesn't publish anything. QuantScenarioBench carries the same idea into JAX and through to a published artifact.
- **Datasets as a deliverable, not just code.** To the best of our knowledge, nothing comparable currently publishes maintained, versioned market-scenario benchmark datasets on Hugging Face; what exists there today is raw historical OHLCV data, not synthetic scenarios with oracle labels.

Honestly stated: this is an execution and integration bet, not a novel-algorithm bet. The underlying stochastic models and the autodiff machinery are known; the value is in assembling them into one coherent, reusable, published toolkit that doesn't currently exist.

## Who This Serves

**Primary**: quantitative researchers — including the author — who need synthetic market scenarios with known ground truth to develop or evaluate pricing models, hedging strategies, or risk estimators, and who currently have to build that infrastructure themselves each time.

**Secondary**: the broader open-source quant/ML research community, who could adopt published QuantScenarioBench datasets as a shared benchmark rather than each publishing one-off synthetic data alongside individual papers.

## Success Criteria

[ASSUMPTION — none of these were specified explicitly; review and adjust]

The primary success criterion is API stability under extension: the common state-space interface must remain stable as new market models are added, with a new model implementable purely by conforming to the interface — no changes to the simulation core or the dataset export pipeline required. Black-Scholes, Heston, and rBergomi are the initial deliverables that prove the API works; they are not themselves the definition of success. The real test is the next model added after them (the first candidates being SABR or jump-diffusion).

Supporting criteria:
- A fixed-seed scenario run is reproducible on a given backend.
- The v1 benchmark dataset (see Scope) is actually published and usable by an external researcher.
- Each of the initial three models is independently testable against known properties of that model (e.g. Black-Scholes scenarios reproduce the closed-form price within numerical tolerance).
- Beyond v1: external usage signals (stars, downloads, forks, citations) as evidence the framework and published datasets are actually adopted by other researchers, not just used internally.

## Scope

**In scope for v1:**
- Common state-space simulation API (`randomness`, `latent_state`, `observation`, `metadata`).
- Black-Scholes, Heston, and rBergomi implementations against that API.
- Reproducible (seeded) scenario generation.
- An export pipeline that publishes generated scenarios as a Hugging Face dataset.
- At least one versioned, reproducible benchmark dataset published as a v1 deliverable in its own right, not merely a pipeline smoke test.
- [ASSUMPTION] Documentation and runnable examples sufficient for an external researcher to generate and use a dataset without reading the source.

**Explicitly out of scope for v1:**
- SABR, jump-diffusion, and other future models — planned, not built, until the v1 API proves itself on the first three.
- Oracle labels (theoretical pricing, hedging deltas) as a shipped dataset feature — acknowledged as a goal, with AD-through-paths as the long-term direction and Monte Carlo as an accepted interim, but the implementation and rollout are deliberately left open beyond this brief.
- Calibration to live or historical market data — v1 simulates from specified model parameters; fitting those parameters to real markets is a separate, later concern. [ASSUMPTION]
- Performance/scale optimization for GPU or TPU — JAX gives this for free where it's used, but it is not a v1 design goal in itself. [ASSUMPTION]
- Evaluating the BMAD + Claude Code AI development workflow — a real motivation for doing this project, but a development-process objective, not a product objective, and deliberately excluded from this brief's success criteria.

## Vision

If this works, QuantScenarioBench becomes the reference implementation for finance-scoped state-space simulation: the framework other researchers reach for, or build on, when they need a stochastic market model expressed as `randomness` / `latent_state` / `observation` / `metadata`. The model zoo grows past the initial three (SABR, jump-diffusion, and beyond, plus community-contributed models against the same state-space interface), and reference datasets — built on top of the framework, not separate from it — accumulate real usage as evidence the standardization actually took hold. Oracle labels mature from Monte Carlo interim to AD-derived ground truth across the model set, strengthening those reference datasets over time rather than redefining the framework itself.

**Platform expansion (added 2026-07-03):** The long-term vision extends one layer further than the framework-plus-datasets picture above. QuantScenarioBench's endpoint is not just a source of published scenario data, but a Hugging Face-native platform for *evaluating* portfolio strategies against that data: every `BenchmarkResult` produced by the benchmark layer (see the PRD's Feature 4.8) is publishable, in a fixed schema, to a shared Hugging Face dataset repo, and those published records accumulate into a public Leaderboard comparing strategies — and, eventually, forecasting models — on the same standardized metrics and datasets. This is the same "standardization accumulates real usage as evidence" bet already stated above for reference datasets, applied one layer up to evaluation itself, not a separate ambition. A hosted Leaderboard *web UI* (a Hugging Face Space or equivalent) is part of this same long-term direction but is deliberately treated as a later phase, not a near-term commitment — see the PRD for the current phase boundary.

## Open Questions & Risks

- **Oracle label fidelity is genuinely unresolved.** AD-through-paths for models without closed-form pricing is literature-supported but not yet proven out in this codebase; Monte Carlo interim labels carry their own variance/bias tradeoffs. This brief deliberately leaves the resolution to implementation, not because it's unimportant, but because it isn't decided yet.
- **Reproducibility across JAX backends is a known sharp edge.** JAX/XLA numerics aren't guaranteed bit-identical across CPU/GPU/TPU; "reproducible" needs a precise, scoped definition before it becomes a tested claim.
- **Solo-maintainer bandwidth against a multi-model roadmap.** The three-model, then-expanding scope is ambitious for what is currently a one-person open-source effort — the narrow, API-first v1 scope above exists specifically to manage that risk rather than front-load the full model zoo.
- **Dataset generation and hosting cost at scale** is unestimated — large benchmark datasets (many scenarios × many models) may have non-trivial compute and storage costs once published.
- **Leaderboard hosting is an ongoing operational commitment, not a one-time deliverable.** *(added 2026-07-03)* A published Evaluation Results repo is a bounded, one-time-per-run publishing cost, similar to dataset publishing above; a hosted Leaderboard *Space* is a live service someone has to keep running and paying for. This brief's solo-maintainer-bandwidth risk (above) applies to that ongoing cost specifically, which is why the Space is scoped as a later phase rather than bundled with Evaluation Results publishing.
