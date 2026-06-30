---
title: Input Reconciliation — brief.md vs PRD
created: 2026-06-30
---

# Input Reconciliation: brief.md → prd.md

**Source input:** `_bmad-output/planning-artifacts/briefs/brief-QuantScenarioBench-2026-06-30/brief.md`
**PRD checked:** `_bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/prd.md`

Note: the brief also has a sibling `addendum.md` which the PRD cites as a joint source ("It builds directly on the finalized Product Brief and its Addendum"). This reconciliation is scoped to `brief.md` only, per the task instruction. Items below are things asserted/emphasized in `brief.md` specifically.

## Method

Read both documents in full. Looked for: (a) qualitative framing, tone, and "why" statements in the brief that the PRD's FR-structure could flatten or silently drop; (b) explicit brief content (named tools, named risks, named audiences) with no PRD trace; (c) places where the PRD's scope is narrower than the brief's without an explicit, deliberate call-out.

## Gaps Found

### Gap 1 — `diffrax` build-on relationship is dropped entirely

The brief states a specific architectural assumption under "What Makes This Different": *"`diffrax` already provides differentiable, GPU-capable SDE solving — QuantScenarioBench is expected to build on it rather than reimplement SDE integration `[ASSUMPTION]`, but adds the finance-specific layer diffrax doesn't have."* This is a load-bearing technical assumption about *how* the SDE integration itself gets built (reuse diffrax vs. write a from-scratch integrator) — it is not just competitive-landscape color.

The PRD has zero mention of `diffrax` anywhere (confirmed via search — no hits in prd.md). The Cross-Cutting NFRs section states "All Market Model simulation logic is implemented in JAX (jit/vmap-compatible)" but never addresses whether SDE integration is hand-rolled or delegated to diffrax. This is a real architectural decision the brief flagged as an open assumption, and the PRD neither carries it forward, confirms it, nor explicitly defers it to the architecture doc. Given this PRD explicitly says it's written to feed downstream architecture work, silently dropping a brief-level technical assumption about the core simulation dependency is a gap, not a deliberate narrowing — there's no "see architecture doc" pointer to make the deferral explicit.

**Recommendation:** Either carry the diffrax assumption into the PRD (e.g., as an Open Question or Cross-Cutting NFR — "SDE integration is expected to build on `diffrax` rather than be reimplemented from scratch `[ASSUMPTION]`"), or explicitly note it's deferred to the architecture phase.

### Gap 2 — "Execution/integration bet, not novel-algorithm bet" framing is lost

The brief has a deliberate, named framing under "What Makes This Different": *"Honestly stated: this is an execution and integration bet, not a novel-algorithm bet. The underlying stochastic models and the autodiff machinery are known; the value is in assembling them into one coherent, reusable, published toolkit that doesn't currently exist."*

This is a "why" statement that sets expectations about where the project's risk and value actually live (integration/packaging, not research novelty) — it's the kind of framing that should shape how reviewers, contributors, or downstream architecture decisions judge tradeoffs (e.g., "don't over-invest in inventing new techniques; the win is in coherent assembly"). The PRD's Vision (§1) states the architectural bet ("The framework's defining bet is architectural, not algorithmic") which partially echoes this, but drops the explicit "execution/integration bet" framing and the comparison to `tf-quant-finance`/`diffrax` that grounded it. The PRD's Vision is a reasonable rephrasing, but it loses the explicit disclaimer that no part of this is a novel-algorithm contribution — useful context for downstream readers (e.g., architecture, code review) who might otherwise look for algorithmic novelty as a goal.

**Severity:** Minor/moderate — the substance (architecture > algorithm) survives, but the explicit framing and competitive grounding (`tf-quant-finance` as the closest analog, `diffrax` as the closest tool) is gone. Worth a one-line callback in Vision or an Open Question.

### Gap 3 — Secondary audience (broader OSS community) and community-contributed models vision dropped

The brief's "Who This Serves" names two audiences: **Primary** (the author/quant researchers needing scenarios) and **Secondary** ("the broader open-source quant/ML research community, who could adopt published QuantScenarioBench datasets as a shared benchmark rather than each publishing one-off synthetic data alongside individual papers"). The brief's Vision section also explicitly anticipates "community-contributed models against the same state-space interface" as part of long-term success.

The PRD's §2 Target User only has "Jobs To Be Done" and "Non-Users" — it never names the secondary OSS-community audience as such, nor does it carry forward the long-term vision of external contributors adding models. FR-14 ("any of the three v1 datasets... external researcher can load_dataset") implicitly serves the secondary audience as a *dataset consumer*, but the brief's broader point — community as a *contributor* base for new models, and as a benchmark-adoption target distinct from the primary researcher persona — has no trace. This is a meaningful flattening: the brief frames secondary-audience adoption as a vision-level success signal (echoed in brief's Success Criteria: "external usage signals... as evidence the framework and published datasets are actually adopted by other researchers, not just used internally"), and the PRD does carry that forward as SM-5, but only the *usage* half, not the *contribution* half (community members adding models, not just consuming datasets).

**Severity:** Minor — SM-5 partially covers the adoption angle; the contributor/community-model angle is the more notable miss, since UJ-3 in the PRD ("A contributor adds a new market model...") is written generically and doesn't connect to the brief's explicit "community-contributed models" vision framing.

### Gap 4 — Dataset hosting/generation cost risk is mentioned but the brief's "non-trivial cost" framing is softened

The brief's Open Questions & Risks states: *"Dataset generation and hosting cost at scale is unestimated — large benchmark datasets (many scenarios × many models) may have non-trivial compute and storage costs once published."* This is one of four named risks in the brief, presented with real weight (it's listed alongside reproducibility-across-backends and solo-maintainer bandwidth as a core risk).

The PRD does carry this forward as Open Question #7 ("Dataset generation and hosting cost at scale — flagged as an unresolved risk in the brief; no budget or ceiling defined yet") — so this is *not* a silent drop. It's listed correctly. This is flagged here only as a verification note, not a gap: confirmed present and accurately represented.

### Gap 5 — Solo-maintainer bandwidth risk and its explicit rationale for the narrow v1 scope is not surfaced

The brief's Open Questions & Risks names *"Solo-maintainer bandwidth against a multi-model roadmap... the narrow, API-first v1 scope above exists specifically to manage that risk rather than front-load the full model zoo."* This is a "why" statement explaining the *reason* for the narrow v1 scope (three models, not the full eventual zoo) — not just a scope decision but its risk-driven justification.

The PRD's MVP Scope (§6) and Non-Goals (§5) correctly narrow scope to three models, matching the brief's decision. But the PRD never states *why* — there's no mention of solo-maintainer bandwidth as the risk being managed. The PRD's Vision (§1) says "even though the first implementation stays narrow" but doesn't connect this to bandwidth/risk management; it reads as a design-quality choice rather than a resourcing-risk mitigation. This is a flattening of rationale: the scope outcome matches, but the "why" (a named, real constraint — one person, ambitious roadmap) is silently dropped. Downstream readers (e.g., architecture/epics planning) lose the signal that aggressive parallelization or rapid scope growth post-v1 carries a known maintainer-capacity risk.

**Severity:** Minor — doesn't affect v1 requirements, but it's exactly the kind of risk-framing the brief emphasized and the PRD's Open Questions (§8) section could have carried forward as an explicit risk note, the way it did for the hosting-cost risk (Gap 4 shows the PRD knows how to do this correctly).

## Items Checked and Confirmed NOT Gaps (explicit, deliberate narrowing — correctly handled)

- **Oracle labels** — brief leaves implementation open; PRD correctly defers (§2.2, §4.3 FR-9 Out of Scope, §5, §6.2, Glossary). Well handled.
- **Reproducibility backend-scoping** — brief flags this as a risk needing precise definition; PRD resolves it explicitly via FR-4 + Cross-Cutting NFRs ("Determinism is backend-scoped"). This is the PRD doing exactly what's expected — going deeper/more specific than the brief.
- **AI-dev-workflow evaluation exclusion** — brief excludes it from success criteria; PRD excludes it from Non-Goals (§5) and references it consistently. Correctly carried forward.
- **License open question** — brief says "fully open source" with no license named; PRD correctly flags this as Open Question #1.
- **GPU/TPU performance non-goal** — carried forward accurately into Non-Goals and as Counter-metric SM-C1.
- **Calibration-to-real-markets exclusion** — carried forward accurately (§2.2, §5).
- **SABR/jump-diffusion deferral** — carried forward accurately (§6.2, Glossary, Vision references "the first candidates being SABR or jump-diffusion" only in the brief, not named in PRD's MVP Scope, but PRD's "any further Market Model beyond the v1 three" is an acceptable generalization — not a gap, just slightly less specific, which is allowed since PRD is downstream).

## Summary

Of the four named risks in the brief's "Open Questions & Risks," the PRD explicitly carries forward two (reproducibility-across-backends, hosting cost) but silently drops the rationale behind a third (solo-maintainer bandwidth → narrow scope decision), and the fourth (oracle label fidelity) is appropriately deferred per the task's allowed-narrowing rule. The most concrete miss is the `diffrax` build-on-it assumption, which is a specific technical commitment in the brief that has no trace anywhere in the PRD, despite the PRD explicitly being written to feed downstream architecture work.
