---
title: Input Reconciliation — PRD Addendum vs. Architecture Spine
created: 2026-06-30
---

# Reconciliation: PRD Addendum -> Architecture Spine

Source: `prds/prd-QuantScenarioBench-2026-06-30/addendum.md`
Target: `architecture-QuantScenarioBench-2026-06-30/ARCHITECTURE-SPINE.md`

The addendum carries forward three technical-foundation points from the product brief, explicitly flagged as open questions for Architecture to resolve (PRD Open Question 8, FR-9's `[ASSUMPTION]`). Each is checked below.

## Point 1 — diffrax as SDE/path-integration foundation

**Addendum claim:** `diffrax` is the leading candidate foundation (JAX-native, autodiff/GPU-capable, handles SRK methods + adjoint backprop); QuantScenarioBench's value-add is the layer diffrax doesn't have (Market Models, state-space schema, export), not reimplementing SDE integration. Treated as "leading candidate, not commitment" (PRD Open Question 8).

**Spine engagement:** Fully engaged and committed.
- Design Paradigm: "the Solver Layer is the single component that turns a Market Model's drift/diffusion into a simulated path via diffrax"
- AD-4 commits `diffrax` as the exclusive solver dependency, isolated to `quantscenariobench.solver`
- AD-3 specifies `diffrax.VirtualBrownianTree` as the default randomness mechanism
- Stack table pins `diffrax 0.7.2`, `jax >=0.4.38`, `equinox >=0.11.10` (diffrax's transitive pins)
- AD-9 dependency diagram shows `solver` as the only consumer of the third-party library

**Verdict:** No gap. The spine resolves Open Question 8 explicitly and traces the "thin layer over diffrax, not a reimplementation" framing through AD-4 and AD-9's import-boundary rules.

---

## Point 2 — Why tf-quant-finance was rejected as a base (independently-implemented reference requirement)

**Addendum claim:** `tf-quant-finance`'s `ItoProcess` pattern was considered and rejected because it's TensorFlow-based, not a thin wrapper-swap candidate. The addendum states this is *specifically why* FR-7 and FR-8's correctness checks should validate against an **independently implemented reference**, not against a bundled general-purpose quant library's pricing formulas. This requirement is also stated directly in the PRD itself (prd.md line 155: "Every correctness check in this feature (FR-7, FR-8, FR-9) validates against an independently implemented reference, not against a general-purpose quant library taken as a bundled dependency").

**Spine engagement:** Not engaged — silent gap.
- The spine never mentions `tf-quant-finance` anywhere.
- The Capability -> Architecture Map binds FR-7–FR-9 only to AD-1 (eqx.Module ABC), AD-4 (diffrax-only solver), AD-6 (equinox pytree convention), AD-7 (float64 policy). None of these ADs addresses *where correctness-reference values come from* (independently implemented vs. reused from a bundled library).
- The Deferred section lists "rBergomi statistical correctness test suite specifics" (covers FR-9 only) but does not mention the broader FR-7/FR-8 "independently-implemented-reference, no bundled quant library" constraint that the PRD states applies to all three FRs.
- This is a real, PRD-stated correctness/testing-architecture constraint with no corresponding AD, Deferred entry, or even a passing mention — it was carried forward by the addendum specifically so Architecture would engage with it, and the spine does not.

**Verdict:** Gap. The spine should either (a) add an AD or extend AD-4's scope to state that reference values for FR-7/FR-8/FR-9 correctness checks must come from independently implemented formulas/methods (not from importing tf-quant-finance or an equivalent bundled library as a dependency), or (b) explicitly add this to Deferred if genuinely not architected yet — currently it is neither.

---

## Point 3 — aBergomi as candidate rBergomi reference technique

**Addendum claim:** Names aBergomi (Markovian approximation of rough Bergomi) as a candidate technique for generating tractable Monte Carlo ground-truth reference values for FR-9's distributional test suite, distinct from the long-term AD-through-paths oracle-label direction (out of v1 scope).

**Spine engagement:** Acknowledged via deferral, not architected — consistent with the addendum's own framing (FR-9 is explicitly `[ASSUMPTION]`, not a committed design point).
- Deferred section: "rBergomi statistical correctness test suite specifics (PRD FR-9 `[ASSUMPTION]`) — aBergomi noted as a candidate reference technique in the PRD addendum; not architected here."
- This correctly distinguishes the deferred v1 test-suite question from the also-correctly-excluded oracle-label/AD-through-paths concern (spine's Deferred also separately lists "Oracle label computation (AD-through-paths, Monte Carlo interim) — out of v1 scope").

**Verdict:** No gap. The spine explicitly surfaces aBergomi by name and correctly scopes it as deferred rather than silently dropping it.

---

## Summary

| Addendum point | Spine engagement | Status |
| --- | --- | --- |
| diffrax as foundation | Committed (Design Paradigm, AD-3, AD-4, AD-9, Stack) | OK |
| tf-quant-finance rejection -> independently-implemented reference requirement (FR-7/FR-8/FR-9) | Not mentioned anywhere; not bound in Capability Map; not in Deferred | **Gap** |
| aBergomi as FR-9 candidate reference technique | Named explicitly in Deferred, correctly scoped vs. oracle-label concern | OK |

**One gap found:** the spine adopted diffrax and surfaced aBergomi, but silently dropped the addendum's tf-quant-finance-derived requirement that FR-7/FR-8/FR-9 correctness checks must validate against independently implemented references rather than a bundled general-purpose quant library — a PRD-stated constraint (prd.md line 155) with no corresponding AD or Deferred entry.
