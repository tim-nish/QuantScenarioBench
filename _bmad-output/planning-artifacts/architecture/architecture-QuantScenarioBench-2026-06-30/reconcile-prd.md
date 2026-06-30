---
title: PRD <-> Architecture Spine Reconciliation
purpose: input-reconciliation
created: '2026-06-30'
inputs:
  - _bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/prd.md
  - _bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/addendum.md
  - _bmad-output/planning-artifacts/architecture/architecture-QuantScenarioBench-2026-06-30/ARCHITECTURE-SPINE.md
---

# PRD <-> Architecture Spine Reconciliation

## Method

Walked every PRD FR (FR-1 through FR-15), every Cross-Cutting NFR, and every Open Question against the
spine's AD-1 through AD-9, Capability -> Architecture Map, Consistency Conventions, and Deferred section.
A finding is reported only if: (a) a PRD FR/NFR has no architectural home — not governed by any AD, not
listed in the Capability -> Architecture Map, and not explicitly named in Deferred; or (b) an AD's Rule and
a PRD FR's testable consequence cannot both be true simultaneously.

Items explicitly carried into Deferred (Parquet row granularity, dataset versioning scheme, HF
namespace/naming, conformance harness mechanism, rBergomi statistical test specifics, license, CI scope,
SABR/jump-diffusion/oracle-labels/multi-asset/CLI) are treated as correctly handled, not gaps — the spine
is allowed to punt explicitly.

## Findings

### Gap 1 — "Public API stability policy" Cross-Cutting NFR has no architectural home

**PRD:** Cross-Cutting NFRs states: "Backward-incompatible changes to `simulate()`, the `Scenario` schema,
or the State-Space Interface require a major version bump." `[ASSUMPTION: semantic versioning assumed]`

**Spine:** No AD addresses *library/API* versioning policy. AD-8 and the Deferred section's "Dataset
versioning scheme specifics" entry both address only the *dataset's* `dataset_version` field (FR-14) —
a distinct concept from the library's own release/semver policy governing `simulate()`/`Scenario`/the
State-Space Interface itself. Searched all 9 ADs, the Capability -> Architecture Map, and Deferred: no
mention of library semver, breaking-change policy, or interface-stability versioning exists anywhere in
the spine.

**Verdict:** Real gap. This NFR is silently dropped, not deferred. It would take one line in Deferred (or
a tenth AD) to fix — e.g. "Library API versioning policy (semver enforcement mechanism for
`simulate()`/`Scenario`/State-Space Interface) — not architected this session."

### Gap 2 — AD-8's fixed Metadata field set omits `n_paths`, which FR-15 requires on every dataset card

**PRD:** FR-15 requires every dataset card to contain, at minimum, six fields: column schema, Market Model
name + parameter values, **TimeGrid and `n_paths` used**, library version, dataset version identifier.
Testable consequence: "a card missing any one of them fails review."

**Spine:** AD-8 states it "resolves PRD Open Question 2" and explicitly "Binds: ... dataset card generation
(FR-15)" — i.e., AD-8 is offered as the architectural mechanism that satisfies FR-15. AD-8's Rule enumerates
the fixed field set: `seed, prng_key_info, model_name, model_version, parameters, time_grid,
library_version, dataset_version, generated_at`. The Consistency Conventions table repeats this same
9-field list verbatim. **`n_paths` is absent from both.**

**Verdict:** Real contradiction, not just an omission. AD-8 explicitly claims to be the mechanism that
makes FR-15's "every card has these fields" guarantee hold, and explicitly closes the field list ("A
Market Model may not omit any of these; it may not add a different name for any of them" — read together
with the Consistency Conventions row repeating the exact same closed list with no `n_paths` synonym), yet
the closed list cannot produce a card containing `n_paths`. If AD-8's metadata is the sole source for card
generation, FR-15 as stated cannot pass review. Either AD-8's field list needs `n_paths` added, or the
spine needs to state explicitly that `n_paths` is sourced from the export-batch context rather than from
per-Scenario `Metadata` (which would also resolve it, but is not currently stated anywhere).

## Non-Findings (checked, no gap)

- **FR-6 (soft validation)** — not governed by a dedicated AD, but is covered by the Consistency
  Conventions table ("Soft validation (FR-6)" row, specifying a single shared warning class) and listed
  under Feature 4.1 in the Capability -> Architecture Map. Architectural home exists.
- **FR-3 (non-uniform TimeGrid)** — covered: `TimeGrid` lives in `interface`; AD-4 centralizes the
  TimeGrid -> `SaveAt` mapping inside the Solver Layer specifically to prevent per-model inconsistency.
- **Cross-Cutting NFR "Language/runtime targets"** — spine's Stack table raises Python to `>=3.11` from the
  PRD's `[ASSUMPTION]` of `>=3.10`, citing diffrax's minimum, and self-flags "PRD needs reconciling." This
  is a disclosed refinement of an explicit PRD `[ASSUMPTION]`, not a silent contradiction — acceptable
  architecture-phase behavior, not a gap.
- **FR-4/FR-5 (reproducibility, randomness materialization)** — AD-3 directly governs both; no conflict
  between "VirtualBrownianTree by default" and "separate path when materialization requested."
  AD-2/AD-6/AD-7 jointly support bit-identical reproducibility on a fixed backend/precision.
- **FR-10/FR-11 (extensibility contract, conformance suite, dummy model never published)** — AD-1, AD-4,
  AD-9 plus the `quantscenariobench.testing` namespace boundary ("never imported by non-test code")
  together satisfy both the zero-source-change guarantee and the "dummy model never shipped" constraint.
- **Numerical correctness NFR (FR-7/FR-8/FR-9 reference-implementation independence)** — Feature-specific
  NFR under 4.2 in the PRD; spine's Deferred section explicitly defers rBergomi test specifics, and AD-7
  (fixed float64) plus AD-4 (single Solver Layer) supply the structural precondition for closed-form/
  semi-closed-form tolerance checks to be meaningful. No contradiction.
- All other FRs (FR-1, FR-2, FR-7, FR-8, FR-9, FR-12, FR-13, FR-14) have explicit entries in the
  Capability -> Architecture Map and at least one governing AD.

## Summary

2 real gaps found:
1. Library API stability/semver policy (Cross-Cutting NFR) has no architectural home anywhere in the spine.
2. AD-8's closed Metadata field list, which is explicitly offered as resolving FR-15, omits `n_paths` —
   a field FR-15 requires on every dataset card — making the AD-8 Rule and the FR-15 consequence
   incompatible as currently written.
