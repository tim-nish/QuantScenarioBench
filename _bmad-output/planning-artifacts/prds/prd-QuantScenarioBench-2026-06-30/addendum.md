---
title: Addendum - QuantScenarioBench PRD
status: final
created: 2026-06-30
updated: 2026-06-30
---

# Addendum: QuantScenarioBench PRD

Technical-foundation context carried forward from the [Product Brief addendum](../../briefs/brief-QuantScenarioBench-2026-06-30/addendum.md) that the PRD deliberately does not resolve (capabilities, not implementation — see PRD §0 Document Purpose). Intended as direct input to the Architecture phase, not as a constraint this PRD imposes.

## SDE / path-integration foundation

The brief's addendum identified `diffrax` (Patrick Kidger) as the most likely foundation layer for QuantScenarioBench's actual path simulation: it is JAX-native, autodiff- and GPU-capable, and already solves the numerical-integration problem (Stochastic Runge-Kutta methods, adjoint-based backprop) — but has no opinion on finance-domain modeling. QuantScenarioBench's value, per that reasoning, is the layer diffrax doesn't have: named Market Models, the state-space schema (PRD §3 Glossary), and the dataset export path (PRD Feature 4.4) — not reimplementing SDE integration.

This PRD treats "build on diffrax" as the leading candidate, not a commitment (PRD Open Question 8). Whatever the Architecture phase decides, it should produce the same observable behavior described by the PRD's FRs (FR-7 through FR-9, FR-4's reproducibility guarantee) — the PRD's requirements do not assume any particular solver internally.

## Why not tf-quant-finance as a base

`tf-quant-finance` (Google) already has a comparable "common API across stochastic processes" pattern (`ItoProcess` base class). It was considered and rejected as a foundation during the brief conversation: it is TensorFlow-based, not JAX, and its architecture is tied to TensorFlow idioms rather than being a thin wrapper swap. This is why FR-7 and FR-8's correctness checks specify an independently implemented reference rather than reuse of a general-purpose quant library's pricing formulas as a bundled dependency — the project's stated bet is its own integration of model + schema + dataset pipeline, not assembly from an existing common-API library.

## Candidate approach for rBergomi ground-truth checks

FR-9 leaves the rBergomi statistical/distributional test suite as an open design point (`[ASSUMPTION]`). The brief's addendum names a specific candidate worth Architecture's attention: Markovian approximations of rough Bergomi (e.g. aBergomi), which trade some model fidelity for tractable, faster-to-compute reference values usable as a Monte Carlo ground truth in tests — distinct from the long-term AD-through-paths direction for *oracle labels* (which is a published-dataset concern, out of v1 scope per the PRD), this is purely about what the v1 test suite checks simulated paths against.
