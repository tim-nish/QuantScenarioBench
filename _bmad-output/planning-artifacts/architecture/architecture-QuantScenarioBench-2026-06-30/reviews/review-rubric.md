---
review: rubric-walker
target: ARCHITECTURE-SPINE.md
date: 2026-06-30
---

# Rubric Review — ARCHITECTURE-SPINE.md (QuantScenarioBench)

**Overall verdict:** The spine is structurally sound and most ADs are genuinely enforceable, but it ships one self-contradictory rule (AD-9 vs AD-1/AD-6 on `equinox` import scope) and silently drops one PRD-flagged open question (dataset generation/hosting cost at scale), so it is not yet finalize-ready as written.

---

## Findings

### CRITICAL — AD-9's import rule contradicts AD-1 and AD-6

**Location:** AD-9 — Dependency direction is one-way (rule text, line ~84)

AD-9's Rule states: *"A Market Model module may import only `quantscenariobench.interface`. `quantscenariobench.solver`, `quantscenariobench.api`, and `quantscenariobench.export` may import `quantscenariobench.interface` and (for `solver` only) third-party `diffrax`/`equinox`."*

This restricts third-party `equinox` imports to the `solver` module exclusively. But:
- AD-1 requires `MarketModel` (in `interface`) to be an `equinox.Module` subclass — `interface` must `import equinox`.
- AD-6 requires every Market Model parameter class (in `models`) to be an `equinox.Module` — `models` must `import equinox` directly to subclass it.

As literally written, AD-9 forbids exactly the imports AD-1 and AD-6 mandate. A contributor implementing a new Market Model and reading AD-9 in isolation would not know whether `import equinox` is permitted in `models/`. This is not a vague rule — it is precise, but precisely wrong, and two independently-built Market Models could resolve the ambiguity differently (one importing equinox per AD-6, another avoiding it out of literal AD-9 compliance and hand-rolling a non-eqx class), which is exactly the kind of incompatible divergence AD-6 exists to prevent.

**Suggested fix:** Reword AD-9's rule to scope the "third-party" restriction to `diffrax` only: *"`quantscenariobench.solver` is the only module that may import `diffrax`. `equinox` may be imported by `interface`, `models`, and `solver` per AD-1/AD-2/AD-6, since it is a project-wide pytree convention, not a diffrax-only dependency."* Also update the Naming/Stack cross-reference so AD-4 (diffrax-only-in-solver) and AD-9 (general dependency direction) aren't read as one merged, conflicting rule.

---

### HIGH — PRD Open Question 7 (dataset generation/hosting cost at scale) is missing entirely, including from Deferred

**Location:** Deferred section; compare to PRD §8 Open Question 7

The PRD explicitly flags as an unresolved risk: *"Dataset generation and hosting cost at scale — flagged as an unresolved risk in the brief; no budget or ceiling defined yet"* (PRD Open Question 7). The spine's Deferred section enumerates PRD Open Questions 1, 3, 4, 5, and 6 explicitly (with PRD-OQ cross-references), and resolves OQ-2 and OQ-8 via ADs — but OQ-7 appears nowhere in the spine, not even as a flagged-but-undecided item.

This is a structural dimension the initiative altitude owns: compute/storage cost at scale plausibly interacts with AD-3 (VirtualBrownianTree vs. materialized randomness — a real cost/memory tradeoff already named in AD-3's own Prevents clause) and with the deferred Parquet row-granularity decision (AD-5/Deferred). Leaving it completely silent — rather than at least flagged per the checklist's standard (cf. how "Release/publish operational envelope" was correctly flagged) — risks two downstream implementers making incompatible assumptions about dataset scale (e.g. one assumes thousands of paths feasible locally, another assumes a cloud budget exists).

**Suggested fix:** Add a Deferred line: *"Dataset generation/hosting cost at scale (PRD Open Question 7) — not addressed in this session; interacts with AD-3's randomness-materialization tradeoff and the deferred Parquet row-granularity decision. Flagged, not architected."*

---

### MEDIUM — Deployment/runtime operational envelope is only half-flagged

**Location:** Deferred section, "Release/publish operational envelope" item

The Deferred section flags the *publish/release* envelope (PyPI process, HF auth/triggers) but says nothing about the *runtime* environment the library itself targets: is `simulate()` assumed to run on a single local machine (CPU/GPU) only, or does v1 anticipate any CI/cloud compute path for generating the benchmark datasets at the volume Feature 4.4 implies? The Cross-Cutting NFR on backend-scoped determinism (cited in the PRD) touches this but the spine doesn't restate or bind it. Given the checklist's explicit instruction to at least flag deployment/environment dimensions, the current item only covers the publishing half, not the execution/compute half.

**Suggested fix:** Either fold an execution-environment line into the existing Deferred bullet ("...and the compute environment(s) `simulate()`/dataset generation are assumed to run in — local-only vs. CI/cloud — not addressed here") or add a sibling Deferred bullet for it.

---

### LOW — Capability → Architecture Map omits explicit FR-1/FR-2/FR-3 → AD bindings

**Location:** Capability → Architecture Map table, Feature 4.1 row

The Feature 4.1 row binds FR-1–FR-6 in aggregate to AD-1/AD-2/AD-3/AD-7, but FR-1 (call signature stability) and FR-3 (explicit TimeGrid, no tuple) aren't individually traceable to a specific AD's Rule the way FR-4 (AD-3), FR-6 (Conventions table), and FR-2 (AD-2) are. `TimeGrid` is named in the Layer→namespace mapping and the Structural Seed but has no AD of its own constraining its shape (e.g., that it must support non-uniform spacing, per FR-3's explicit consequence). This is a minor traceability gap, not a contradiction — FR-1/FR-3 are arguably satisfied by the `interface` module's existence — but a story-level implementer has no Rule to point to if someone proposes a `(start, stop, steps)` tuple shortcut for `TimeGrid`.

**Suggested fix:** Either add a short AD (or extend AD-1/AD-2) explicitly constraining `TimeGrid` to be an object supporting non-uniform spacing, or add an explicit note in the Conventions table cross-referencing FR-3 the way FR-6 is handled.

---

### LOW — `parameters` field name vs. AD-8 wording is a minor naming-precision gap

**Location:** AD-8 Rule vs. Consistency Conventions table (Metadata field names row)

AD-8's Rule prose says metadata carries "Market Model name and version-stamped parameter values" while the Conventions table's fixed field-name list uses a single field `parameters` (alongside `model_name`, `model_version`). The Rule's "version-stamped parameter values" phrasing could be read as the *parameters themselves* needing per-field version stamps, versus the table's flatter reading (one `parameters` blob, version captured separately via `model_version`). Low ambiguity, but two implementers could format the `parameters` field differently (flat dict vs. nested with embedded version tags) based on which wording they anchor to.

**Suggested fix:** Align AD-8's Rule prose to the Conventions table's exact field list and confirm `parameters` is a flat snapshot, with versioning carried solely by the separate `model_version` field.

---

## Items checked and found acceptable (no finding)

- **Mermaid diagram (AD-9):** Valid `graph LR` syntax, 7 named nodes, 9 edges, renders a real one-way dependency constraint (models/solver/api/export/testing all point into `interface`; only `api` depends on `solver`; `caller` is the only node pointing at `models` directly). Not a placeholder.
- **Stack table versions:** Python >=3.11, jax >=0.4.38, diffrax 0.7.2, equinox >=0.11.10 — all match the memlog's "(version)" entry verified 2026-06-30, and an independent web check confirms diffrax 0.7.2 (released Feb 2026) is still current as of this review date. Not stale-training-data-derived.
- **AD-1 through AD-8 individually:** each names a concrete enforcement mechanism (subclassing requirement, field-level `static=True`, single-call-path construction, single-import-site restriction, fixed field list, single warning class) rather than aspirational language — these are genuinely checkable in code review, not vague.
- **Deferred section, non-load-bearing items:** SABR/jump-diffusion future models, oracle labels, multi-asset/CLI/calibration, license choice, CI/test-infra scope, conformance-harness mechanism (pytest vs. hypothesis), rBergomi statistical suite specifics, dataset versioning scheme, Parquet row granularity, HF namespace — none of these, if left undecided through implementation, allow two independently-built v1 units (the three named Market Models, `simulate()`, `export`) to diverge *incompatibly*, because AD-1 through AD-9 already fix the structural contract surface those decisions sit behind. Correctly deferred.
