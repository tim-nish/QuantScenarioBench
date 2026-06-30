---
lens: web-verification
target: ARCHITECTURE-SPINE.md
reviewed: 2026-06-30
---

# Web Verification Review — QuantScenarioBench Architecture Spine

## Scope

Checked every named technology/version claim in the Stack table and every AD that depends on a
specific library capability, against (a) the memlog's audit trail of what was actually verified
and (b) independent live web checks performed during this review.

## 1. Memlog coverage of Stack-table claims

The memlog contains exactly one `(version)` entry (line 9), dated 2026-06-30, covering:

- diffrax 0.7.2 on PyPI requires Python >=3.11, jax>=0.4.38, equinox>=0.11.10
- diffrax solvers/terms subclass equinox.Module
- eqx.Module is simultaneously an ABC (abstractmethod works) and a registered JAX pytree
- eqx.field(static=True) marks non-array fields as pytree aux_data, not traced leaves
- diffrax.VirtualBrownianTree generates Brownian increments on-the-fly (tree-search algorithm,
  no full noise-path materialization), passed as the control to ControlTerm
- diffrax.UnsafeBrownianPath exists as a faster, determinism/backprop-restricted alternative

This single entry backs every cell in the Stack table (Python, jax, diffrax, equinox versions)
and the library-capability assertions underlying AD-1 (eqx.Module ABC), AD-2 (eqx.field static
semantics), AD-3 (VirtualBrownianTree behavior), and AD-6 (eqx.Module as project-wide pytree
convention). All of these trace to a real verification event, not bare assertion.

## 2. Independent spot-checks performed live (2026-06-30)

| Claim | Method | Result |
| --- | --- | --- |
| diffrax 0.7.2 requires Python >=3.11, jax>=0.4.38, equinox>=0.11.10 | Fetched `https://pypi.org/pypi/diffrax/0.7.2/json` directly (raw release metadata, not a summarized search) | **Confirmed exactly.** `requires_python: ">=3.11"`; `requires_dist` includes `equinox>=0.11.10`, `jax>=0.4.38`, plus jaxtyping/lineax/optimistix/typing-extensions/wadler-lindig not mentioned in the spine but not contradicting it either. |
| eqx.field(static=True) marks a field as pytree aux_data, not a traced leaf | Fetched Equinox's own docs page (`docs.kidger.site/equinox/api/module/advanced_fields/`) | **Confirmed.** Static field becomes part of PyTree structure (aux_data) rather than a leaf; does not interact with jit/grad. Docs additionally note it should be used rarely and `eqx.partition` is often preferred — this nuance isn't contradicted by AD-2's usage (metadata is non-array, the documented correct use case) but is worth the team's awareness. |
| equinox 0.11.10 (the diffrax-pinned floor) requires JAX >=0.4.13, Python 3.9+ | Fetched `https://pypi.org/pypi/equinox/0.11.10/json` | **Confirmed, and consistent** — equinox's own floor (jax>=0.4.13) is looser than diffrax's effective floor (jax>=0.4.38), so diffrax's pin is the binding constraint, as the spine implies. |
| eqx.Module is simultaneously an ABC and a registered JAX pytree | Web search of Equinox docs/arXiv paper | **Confirmed.** Equinox's own docs state "Equinox modules are all ABCs by default" (abc.abstractmethod works) and that "all eqx.Module really does is register your class with JAX as a PyTree node." |
| diffrax.VirtualBrownianTree generates Brownian motion on-the-fly without materializing the full path | Web search of Diffrax docs / arXiv (2405.06464) | **Confirmed.** VBT's path is uniquely determined by a single PRNG seed; samples need not be stored, giving constant memory footprint — matches AD-3's rationale exactly. |

One initial search response (a generic AI-summarized snippet) incorrectly returned "Python 3.10+"
for diffrax 0.7.2 — this was caught and overridden by going to the authoritative PyPI JSON
metadata directly, which is the correct, citable source. This is a useful illustration of why
direct-source verification (not summarized search results) matters for this kind of claim.

## 3. Claims in the spine with no corresponding memlog verification

- **AD-7 (float64 / JAX x64 mode)** — The mechanism behind this AD (JAX defaults to float32;
  x64 must be explicitly enabled, e.g. via `jax.config.update("jax_enable_x64", True)` at process
  entry) is real and not in dispute, but the memlog records this purely as `(decision by user)`
  with no `(version)` or web-verification entry. It is a policy decision more than a
  library-capability claim, and it cites no specific API surface that could be stale, so the risk
  of staleness is low — but strictly, it has zero corresponding verification line in the memlog.
- **AD-9 / dependency-direction diagram** — architectural convention, not a library claim; no
  verification needed or expected.
- **"diffrax solvers/terms subclass equinox.Module"** (memlog line 9, used as general supporting
  context, not cited as a standalone spine AD) — not independently re-verified in this review
  (diffrax's Terms API docs page didn't state the inheritance explicitly); doesn't affect any
  spine commitment since the spine's own ADs don't depend on this specific sub-fact.

No Stack-table cell and no AD-1/2/3/4/6 capability claim is missing memlog backing. AD-7 is the
one committed technical decision in the spine lacking an explicit verification trail, though its
mechanism is uncontroversial.

## Verdict

The spine's load-bearing technology and version claims are accurate and were genuinely
web-verified, not asserted from training data — independent re-verification today confirms
diffrax 0.7.2's Python/jax/equinox pins, eqx.Module's ABC+pytree duality, eqx.field(static=True)
semantics, and VirtualBrownianTree's on-the-fly/no-materialization behavior, all exactly as
recorded in the memlog's single `(version)` entry. The only gap is AD-7 (float64/x64 mode),
which is recorded as a user decision with no explicit web-verification line in the memlog,
though the underlying mechanism is not in dispute.
