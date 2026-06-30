---
title: Addendum - QuantScenarioBench
status: final
created: 2026-06-30
updated: 2026-06-30
---

# Addendum: QuantScenarioBench

Supporting depth that informed the brief but doesn't belong in a 1-2 page executive document. Useful input for the downstream PRD/architecture work.

## Landscape research (full digest)

**Comparable/adjacent libraries**

- **diffrax** (Patrick Kidger) — JAX-native, autodiff- and GPU-capable solver for ODEs/SDEs/CDEs, including Stochastic Runge-Kutta methods and adjoint-based backprop through integration. General-purpose numerical toolkit, not finance-specific: no built-in Black-Scholes/Heston/rBergomi models, no state-space convention, no dataset publishing. Most likely foundation layer for QuantScenarioBench's actual path simulation rather than something to duplicate.
- **tf-quant-finance** (Google) — TensorFlow-based, has a real common-API pattern (`ItoProcess` base class with `drift_fn`/`volatility_fn`, Euler-scheme sampling) covering GBM-type diffusions and others. Same architectural idea as the proposed state-space API, but TF-based rather than JAX, and not packaged as benchmark datasets.
- **jaxfin** (paolodelia99) — small, less mature JAX library for exotic option pricing; no evidence of a unified multi-model state-space API or rough-volatility support.
- **QuantLib / py_vollib** — QuantLib supports many models but as separate engines rather than a unified API; not JAX, not differentiable. py_vollib (and a newer "fast-vollib" JAX/PyTorch port) is narrowly Black-Scholes-family IV/Greeks, no stochastic-vol or rough-vol support.

**Hugging Face dataset landscape**

Search surfaced real historical market data (e.g. `mito0o852/OHLCV-1m`, `123olp/binance-futures-ohlcv`) but no popular, maintained synthetic-scenario or option-pricing benchmark dataset. Related synthetic-market work exists only as research artifacts attached to papers (GAN/diffusion option-market simulators — e.g. "Deep Hedging: Learning to Simulate Equity Option Markets," "Beyond Monte Carlo" diffusion-based pricing) rather than published, reusable HF datasets.

**rBergomi pricing**

Confirmed: rBergomi has no closed-form or semi-closed-form pricing (unlike Black-Scholes' analytic formula or Heston's characteristic-function/Fourier approach) — its non-Markovian, fractional driver rules out classical methods, so Monte Carlo is the literature standard. Approaches for ground-truth sensitivities without closed forms: pathwise/likelihood-ratio MC estimators (pathwise generally preferred when applicable), Markovian approximations (e.g. aBergomi) to speed up MC, and increasingly autodiff through simulated paths — exactly what JAX/diffrax provide natively, and the direction the brief commits to long-term.

Sources: diffrax (github.com/patrick-kidger/diffrax), tf-quant-finance (github.com/google/tf-quant-finance), jaxfin (github.com/paolodelia99/jaxfin), fast-vollib (arxiv.org/html/2604.27210v1), Markovian approximation of rBergomi for MC pricing (arxiv.org/pdf/2007.02113), hierarchical sparse grids/QMC for rBergomi (arxiv.org/pdf/1812.08533), mito0o852/OHLCV-1m (huggingface.co/datasets/mito0o852/OHLCV-1m), "Deep Hedging: Learning to Simulate Equity Option Markets" (arxiv.org/pdf/1911.01700).

## Why not just extend an existing library

Considered and set aside during the brief conversation:

- **Extending tf-quant-finance** instead of building new: rejected because the project is explicitly JAX-scoped (motivation #1 in the original framing), and tf-quant-finance's architecture is tied to TensorFlow idioms, not a thin wrapper swap.
- **Treating diffrax as sufficient on its own**: rejected because diffrax solves the numerical-integration problem but has no opinion on finance-domain modeling (named stochastic models, the state-space schema, dataset export) — the value QuantScenarioBench adds is exactly the layer diffrax doesn't have.

## Development-process motivation (out of brief scope)

The third original motivation — using this project to evaluate an AI-driven development workflow (BMAD + Claude Code) end to end, from requirements through stories, implementation, and CI — was explicitly excluded from the brief's success criteria per the author's own framing: it's a development objective, not a product objective. Recorded here for downstream context (e.g. if a PRD or process retrospective wants to reference why this project was chosen as the workflow's test case), not as something the product itself should be measured against.
