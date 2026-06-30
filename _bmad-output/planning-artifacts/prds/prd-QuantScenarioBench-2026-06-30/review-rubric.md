# PRD Quality Review — QuantScenarioBench

## Overall verdict

This PRD is decision-ready and unusually honest about what it doesn't know yet — the `[ASSUMPTION]` and Open Questions machinery is used correctly rather than decoratively, and the FRs are written so an engineer could build conformance tests directly from them. The main risks are mechanical, not strategic: a missing FR-12 breaks ID contiguity, and a handful of FRs (FR-16, dataset card contents; the "documentation sufficient" line in §6.1) lean on adjectives a strict reading of Done-ness would reject. Shape fit is correct — this is a capability spec for a single-operator tool, and it resists the temptation to pad itself with persona/UJ theater.

## Decision-readiness — strong

The PRD states real trade-offs and lets them stand rather than smoothing them away. FR-6's soft-validation choice ("never a hard exception") is stated as a decision with a stated reason (researchers exploring constraint-violating regions), not hedged as "where appropriate." The Cross-Cutting NFR on backend-scoped determinism explicitly says what is *not* guaranteed (cross-backend bit-identity) rather than letting "reproducible" do unexamined work. The counter-metrics (SM-C1, SM-C2) are genuine tensions, not safe checkpoints: SM-C2 ("model count... would falsify SM-1 even while looking like progress") names a real way the project could look successful while failing its own thesis — that's the rubric's red-flag test passed, not failed.

Open Questions (§8) are actually open: license choice, Metadata field list, versioning scheme, Parquet granularity, HF namespace, and hosting cost are all unresolved with no answer smuggled into the next sentence. Q6 (whether CI/test-infra belongs in this PRD) is a live scoping tension the PRD declines to resolve for itself, which is the right call to surface rather than silently decide.

### Findings
- **low** Open Questions are dense relative to "launch-grade" framing (§8, 7 items) — Note: appropriate at this stage per the rubric's stakes-relative test (this is pre-architecture, not a green-light-to-build-tomorrow doc), but worth the author's awareness that several of these (license, HF namespace, cost ceiling) block actual publishing and should close before Hugging Face upload, not before architecture. *Fix:* none needed for this gate; flag for the author to track which Open Questions block which downstream milestone.

## Substance over theater — strong

No persona theater — there is exactly one user role, stated once, not multiplied into named personas to look thorough. No innovation-section padding — novelty claims live in the brief, and the PRD itself doesn't re-litigate "what makes this different," it just builds the contract. NFRs in the Cross-Cutting section are specific, not boilerplate: "Determinism is backend-scoped," "Public API stability policy [requires] a major version bump" are product-specific commitments, not "the system must be reliable." The Vision statement (§1) is concrete enough that it could not swap into another PRD unchanged — it names the actual contract fields (`observation`, `latent_state`, `metadata`) and the actual three models, which is the opposite of vision theater.

### Findings
None — this dimension is clean.

## Strategic coherence — strong

The thesis is explicit and singular: the state-space contract (§1) is "architectural, not algorithmic," and every feature traces back to proving that contract holds under extension. Feature 4.3 (State-Space Extensibility Contract) states outright that it "exists to make the brief's primary success criterion testable, not just stated" — that's a PRD naming its own thesis-to-feature linkage rather than leaving it implicit. SM-1 (the dummy-model conformance test) is the load-bearing metric and it validates the thesis directly, not adoption or activity. SM-5 (stars/downloads) is correctly demoted to secondary and explicitly "tracked not gated," avoiding the DAU/MAU-style false proxy the rubric warns about. MVP scope is a coherent "problem-solving" capability-spec scope: every in-scope item (§6.1) traces to an FR that serves the contract-proving thesis; nothing reads as "easy first."

### Findings
None — this dimension is clean.

## Done-ness clarity — adequate

Most FRs are strong on this axis: FR-1 through FR-15 mostly carry a "Consequences (testable)" block with concrete, checkable conditions ("bit-identical... arrays," "zero changes to `simulate()`'s source," "round-trips... without loss"). This is the dimension the rubric says to be most unforgiving on, and most of the PRD earns that scrutiny. But a few spots fall back on adjectives or under-specify the testable surface:

- FR-16's own consequence — "a researcher unfamiliar with QuantScenarioBench can determine... how to interpret the columns" — is a usability claim with no bound (how is this checked? a checklist of required card fields? a comprehension test?). It's tagged `[ASSUMPTION]` for content but the *testability* of the consequence itself isn't tightened by that tag.
- §6.1's "Documentation and runnable examples sufficient for an external researcher to generate or consume a dataset without reading the source" is a textbook instance of the rubric's "reasonable performance / user-friendly" red flag — "sufficient" is never bounded (word count? example count? a specific scenario the docs must walk through?).
- FR-9's correctness consequence depends on a test suite that is itself `[ASSUMPTION]`-tagged as undefined ("specific statistical test suite not yet defined") — so the FR's testable consequence is currently a placeholder for a testable consequence, not one yet. This is honestly flagged (good scope-honesty), but it does mean Done-ness for FR-9 is not yet achievable from the PRD alone.

### Findings
- **medium** FR-16 dataset-card consequence is unbounded ("can determine... how to interpret") (§4.4, FR-16) — no checklist or required-field list makes this testable as written. *Fix:* either replace with a concrete required-fields checklist (schema, model+params, TimeGrid, n_paths, library version — already named in the FR body) stated as an acceptance list, or explicitly defer the bound to the architecture/story stage with a `[NOTE FOR PM]`.
- **low** §6.1 documentation requirement uses "sufficient... without reading the source" with no bound (§6.1) — unfalsifiable as stated. *Fix:* tie to a concrete artifact (e.g. "a runnable example notebook per Market Model" or "quickstart that round-trips one full simulate→export→load cycle").
- **low** FR-9's testability is currently deferred to an undefined test suite (§4.2, FR-9) — correctly `[ASSUMPTION]`-tagged, but worth a `[NOTE FOR PM]` since it's the only v1 model whose correctness bar isn't yet defined even in principle. *Fix:* no PRD change required before this gate; flag as a pre-architecture blocker since story creation can't write FR-9 acceptance tests without it.

## Scope honesty — strong

Non-Goals (§5) does real work — six explicit exclusions, each with a one-line reason, including the self-aware exclusion of "Evaluating the BMAD + Claude Code AI development workflow" as a non-goal because it's a process objective, not a product requirement (a distinction many PRDs blur). The `[ASSUMPTION]` tags (8 of them) are all genuinely inferred content the author didn't directly confirm — schema field lists, opt-in mechanisms, versioning schemes, test harness shape — not decorative hedging. §6.2 "Out of Scope for MVP" de-scopes SABR/jump-diffusion and oracle labels honestly, with the deferral reasoning visible ("deferred until the v1 interface proves itself") rather than silent omission. Open-items density (7 Open Questions + 8 Assumptions) is proportionate to a pre-architecture PRD that explicitly says it's "launch-grade design, incremental implementation," not a green-light-to-build document — consistent with the rubric's stakes-relative calibration.

### Findings
None — this dimension is clean.

## Downstream usability — adequate

This PRD does feed downstream BMad workflows (§0 says so explicitly: "written for... downstream BMad workflows (architecture, epics/stories)"), so this dimension matters more than it would for a pure standalone doc. The Glossary (§3) is comprehensive and the FRs largely use its terms consistently (Market Model, Scenario, State-Space Interface, TimeGrid all recur identically). UJs each name a protagonist ("A researcher...", "A contributor...") and each is tied to a Feature via "Realizes Feature X.X" — no floating UJs.

The one real defect is mechanical but significant for downstream extraction: **FR-12 does not exist** — Feature 4.3's FRs go FR-10, FR-11, then Feature 4.4 starts at FR-13. If this is intentional (e.g., an FR was cut during drafting), it should say so; if it's a slip, it will confuse anyone doing FR-to-story traceability later, since a downstream workflow scanning for FR-12 will reasonably wonder if a section got lost in editing.

### Findings
- **medium** FR ID gap: FR-12 is missing between FR-11 (§4.3) and FR-13 (§4.4) — no FR-12 appears anywhere in the document, and nothing notes the gap as intentional. *Fix:* either renumber FR-13 onward to close the gap, or add a one-line note ("FR-12 reserved/removed during drafting") so downstream story creation doesn't treat it as a missing requirement.

## Shape fit — strong

This is correctly shaped as a single-operator capability spec, not forced into a consumer-product mold. The PRD says so explicitly (§2.3: "Single-persona, API-first, capability-driven product — journeys are kept to one-line scope per the Lighter dial rather than full named-persona narratives") — that's the PRD naming its own shape-fit choice rather than leaving the reviewer to infer whether thin UJs are a gap or a deliberate dial setting. Three UJs at one line each is proportionate: enough to anchor each major Feature to a concrete use, not so much that it pads a tool with no second stakeholder. SMs are appropriately operational (API-stability, reproducibility, dataset-load-success) rather than user-facing engagement metrics, which is correct for this product type. No over-formalization (no invented secondary personas), no under-formalization (the contract itself — the actual product — gets full FR-level rigor where it matters).

### Findings
None — this dimension is clean.

## Mechanical notes

- **FR ID gap**: FR-12 is absent (FR-11 → FR-13, see Downstream usability finding above). This is the one mechanical issue with real downstream cost.
- **Glossary consistency**: Terms are used consistently in their canonical casing throughout (Market Model, Scenario, State-Space Interface, TimeGrid, Benchmark Dataset, Oracle Label) — no drift into lowercase/plural variants spotted in a full pass of §3 against §4–§9.
- **Assumptions Index roundtrip**: All 8 inline `[ASSUMPTION]` tags (§3/FR-4 Metadata, FR-5 randomness opt-in, FR-9 statistical suite, FR-11 harness mechanism, FR-13 row granularity, FR-15 versioning, FR-16 card contents, §6.1 docs) are indexed in §9, and every §9 entry has a corresponding inline tag. Clean roundtrip — no orphans either direction. (Two further `[ASSUMPTION]` tags appear in Cross-Cutting NFRs — semver policy and Python/JAX version targets — that are *not* listed in §9; minor index gap.)
- **UJ protagonist naming**: All three UJs (§2.3) name a protagonist ("A researcher...", "A researcher...", "A contributor...") and carry enough context inline to stand alone. Clean.
- **Cross-references**: "Realizes Feature X.X" and "Realizes UJ-N" links all resolve to sections that exist. FR-to-SM "Validates FR-N" links in §7 all resolve correctly.

### Findings
- **low** Two `[ASSUMPTION]` tags in Cross-Cutting NFRs (semantic-versioning policy; Python/JAX version targets, end of document) are not listed in the §9 Assumptions Index. *Fix:* add two lines to §9 for completeness, or fold Cross-Cutting NFRs into the §9 scan scope explicitly.
