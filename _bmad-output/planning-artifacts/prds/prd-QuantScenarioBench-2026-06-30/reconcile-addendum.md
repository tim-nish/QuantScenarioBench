---
title: Input Reconciliation - addendum.md vs prd.md
created: 2026-06-30
---

# Reconciliation: addendum.md against prd.md

**Source input:** `_bmad-output/planning-artifacts/briefs/brief-QuantScenarioBench-2026-06-30/addendum.md`
**PRD checked:** `_bmad-output/planning-artifacts/prds/prd-QuantScenarioBench-2026-06-30/prd.md`

## Scope of this check

The addendum is explicitly a dumping ground for landscape research, rejected-alternatives reasoning, and out-of-scope process context that should NOT be pulled into the PRD narrative verbatim. Most of its content (the library survey table, the HF dataset landscape scan, the BMAD-workflow motivation) is correctly absent from the PRD as prose — that is by design, not a gap. This check looks only for addendum content that should have shaped a PRD *decision* (an FR, an NFR, an architectural commitment, a glossary term) but appears to be missing or inconsistent.

## Findings

### Gap 1 — diffrax is named as "most likely foundation layer" but the PRD has no JAX-ecosystem dependency NFR

Addendum: "Most likely foundation layer for QuantScenarioBench's actual path simulation rather than something to duplicate" (diffrax entry, Landscape research).

PRD: Cross-Cutting NFRs include "JAX-native computation. All Market Model simulation logic is implemented in JAX (jit/vmap-compatible)" but never mentions diffrax, an SDE/ODE solver library, or any expected numerical-integration foundation. A search of the PRD for "diffrax" returns zero matches.

This isn't necessarily wrong — the PRD may intentionally defer solver-library choice to the architecture doc — but the addendum frames diffrax as the *likely* foundation for the rBergomi/Heston SDE simulation specifically (since rBergomi's Stochastic Runge-Kutta / autodiff-through-paths needs are non-trivial and diffrax is called out as providing them "natively"). The PRD's FR-9 (rBergomi) and Non-Goals' "AD-through-paths... long-term direction" line both gesture at functionality diffrax exists to provide, without ever naming a numerical-integration dependency or committing to (or explicitly deferring) one. Worth flagging to the architecture stage rather than silently dropping — currently there is no `[ASSUMPTION]` tag or Open Question pointing at "what SDE solver library does v1 actually use," which seems like a real open question the addendum surfaces but the PRD doesn't carry forward even as a flagged unknown.

**Severity:** Moderate. Likely intentional deferral to architecture, but the PRD has an `[ASSUMPTION]`/Open-Questions convention for exactly this kind of unresolved-but-decision-relevant item, and this one isn't captured there.

### Gap 2 — "Why not extend tf-quant-finance" rejection reasoning isn't reflected as a constraint anywhere in the PRD

Addendum: tf-quant-finance is rejected as a base specifically because "the project is explicitly JAX-scoped... and tf-quant-finance's architecture is tied to TensorFlow idioms, not a thin wrapper swap." This is presented as a settled architectural decision (build new, JAX-only, no TF dependency).

PRD: The JAX-native NFR exists ("All Market Model simulation logic is implemented in JAX") but the PRD never states a no-TensorFlow-dependency constraint or otherwise closes the door on a hybrid/wrapper approach. This is a minor gap — the JAX-native NFR as worded is consistent with (not contradictory to) the addendum's reasoning, and FR text never suggests reusing tf-quant-finance. Flagging only because the rejection reasoning ("not a thin wrapper swap") implies a stronger constraint than "JAX-native" alone conveys — e.g., it rules out using tf-quant-finance's pricing math even as a correctness oracle for FR-7/FR-8 numerical-tolerance checks. The PRD doesn't say what reference implementation FR-7 (Black-Scholes) and FR-8 (Heston) correctness checks will validate against, so there's no way to confirm whether this constraint is actually being honored or violated.

**Severity:** Low. No direct contradiction found, just an unstated boundary that downstream architecture work could inadvertently cross (e.g., picking tf-quant-finance as a numerical reference implementation for tolerance tests).

### Gap 3 — rBergomi "no closed form, MC is the literature standard" claim is correctly reflected, but the addendum's specific alternative methods (pathwise/likelihood-ratio estimators, Markovian approximations like aBergomi) are absent from FR-9 and the Open Questions

Addendum: gives three named approaches for ground-truth sensitivities without closed forms — pathwise/likelihood-ratio MC estimators, Markovian approximations (aBergomi), and autodiff through simulated paths.

PRD: FR-9 correctly states rBergomi has no closed-form/semi-closed-form reference and falls back to "distributional/statistical sanity checks," tagged `[ASSUMPTION: specific statistical test suite not yet defined]`. This is consistent with the addendum, not contradictory. However, the addendum's aBergomi (Markovian approximation) is a candidate for a *faster, semi-closed-form-like* correctness reference that the PRD doesn't mention as an option — meaning FR-9's open assumption about "the specific statistical test suite" might have been narrowed (or at least informed) by aBergomi as an approximate oracle, but the addendum's reasoning for that path doesn't appear to have been considered.

**Severity:** Low / informational. Not a contradiction — FR-9's `[ASSUMPTION]` tag already correctly defers this decision. Flagging only because aBergomi is a concrete addendum-provided option for resolving that open assumption and isn't referenced in §8 Open Questions or §9 Assumptions Index, so the downstream architecture/stories work may not know this option exists unless it re-reads the addendum.

## Non-gaps (confirmed correctly excluded, no action needed)

- The full landscape research table (jaxfin, QuantLib, py_vollib, fast-vollib) — correctly omitted from PRD narrative; not decision-relevant beyond the "why not extend" reasoning already addressed above.
- HF dataset landscape scan (no existing synthetic-scenario benchmark dataset found) — this is competitive-context, not a PRD decision input; correctly absent. The PRD's FR-13/14/16 dataset requirements are not contradicted by it.
- BMAD/Claude Code AI-dev-workflow motivation — explicitly excluded from brief success criteria per the addendum itself, and the PRD correctly mirrors this exclusion (§5 Non-Goals: "Evaluating the BMAD + Claude Code AI development workflow... a development-process objective, not a product requirement"), with a forward reference back to the brief addendum. Consistent, no gap.

## Summary

No outright contradictions between the addendum and the PRD were found. The PRD is consistent with the addendum's two firm rejections (no TF-based extension, no diffrax-only "good enough" stance) and with its rBergomi MC-is-standard conclusion. The gaps identified are all of the same shape: the addendum names specific technical building blocks or candidate decisions (diffrax as foundation, aBergomi as approximate oracle, "not a thin wrapper" as a stronger-than-stated constraint) that the PRD's existing `[ASSUMPTION]`/Open Questions apparatus would normally be expected to carry forward, but doesn't currently reference. These are good candidates for either (a) explicit Open Questions additions, or (b) confirmation that they're intentionally deferred to the architecture document.
