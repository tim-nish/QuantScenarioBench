# Adversarial Finalize Review — ARCHITECTURE-SPINE.md (QuantScenarioBench)

Reviewer lens: adversarial (mandatory finalize_reviewers floor #2)
Method: for each AD, construct two independent engineers, each reading the Rule literally and complying, and show their outputs are structurally/behaviorally incompatible without either one violating the Rule as written.

---

## AD-1 — State-Space Interface is an `equinox.Module` ABC

**Rule as written:** `MarketModel` is an `equinox.Module` subclass with abstract methods enforced at construction time. Every concrete Market Model subclasses `MarketModel`; none registers itself as a pytree by hand.

**Scenario:** Engineer A implements `Heston`, Engineer B implements `RoughBergomi`, both reading only AD-1 (and the namespace map, which says nothing about method signatures).

- A defines `MarketModel.drift(self, t, state) -> Array` and `MarketModel.diffusion(self, t, state) -> Array` as the abstract methods — two separate calls per step.
- B defines a single abstract method `MarketModel.drift_diffusion(self, t, state) -> tuple[Array, Array]` — one call per step.
- Both are legitimate "abstract methods enforced at construction time" on an `equinox.Module` subclass. Neither registers a pytree by hand. AD-1 is satisfied by both, yet `Heston` and `RoughBergomi` are not interchangeable inputs to anything that calls the ABC's methods — including the Solver Layer (AD-4) and the FR-11 conformance suite (`quantscenariobench.testing`), which would have to special-case which calling convention each model uses.
- **Gap:** AD-1 pins the ABC's *base class* and *construction-time enforcement*, never the actual abstract method names, arity, argument order (`(t, state)` vs `(state, t)`), or return shape/dtype convention (batched vs unbatched, real vs complex for rough volatility kernels). "Abstract methods" is asserted but never enumerated anywhere in the spine.

---

## AD-2 — `Scenario` is an `equinox.Module` with a fixed dynamic/static field split

**Rule as written:** `observation`/`latent_state` are dynamic `eqx.Module` fields; `metadata` is `eqx.field(static=True)`.

**Scenario:** Engineer A builds the Solver Layer (producer of `Scenario`); Engineer B builds Export (consumer of `Scenario`).

- A's Solver Layer emits `observation` with shape `(n_paths, n_steps, n_dim)` (paths-major, the natural diffrax `vmap`-over-paths layout) and `latent_state` as a single stacked `Array` of shape `(n_paths, n_steps, n_latent)`.
- B's Export, reading only AD-2, assumes `latent_state` is itself a *nested* pytree of named sub-arrays (e.g., `{"variance": Array, "vol_of_vol_path": Array}` for Heston) because "dynamic field" says nothing about whether the field is a single array or a sub-pytree, and a per-field-named Parquet schema (which AD-5 implies) is far more natural from a nested structure.
- Both comply with AD-2's letter ("dynamic field, not static"). Export's pytree-flatten (AD-5) over A's `Scenario` either produces one opaque `latent_state` column of stacked arrays, or crashes/produces meaningless flattened leaf names, depending on which producer it's pointed at.
- **Gap:** AD-2 fixes static-vs-dynamic but not the *internal shape contract* (axis order, batch convention, scalar-vs-pytree-valued fields) of `observation`/`latent_state` — exactly the thing AD-5's "generic over Scenario" promise depends on.

---

## AD-3 — Randomness defaults to `VirtualBrownianTree`; materialization is a separate path

**Rule as written:** Default path uses `diffrax.VirtualBrownianTree`; materialized `Randomness` (FR-5) "takes an explicit, separate construction path — never a runtime branch inside the default path."

**Scenario:** Engineer A implements the default-path Solver entrypoint; Engineer B implements the materialized-randomness extension, both bound by "separate ... path, never a runtime branch inside the default path."

- A reads "separate path" as: `solve_sde(model, time_grid, key)` is the only function; materialization is controlled by an `eqx.field`-level flag on a `Randomness` config object passed into the *same* function, dispatched via `eqx.filter` / `lax.cond`-free Python-level `if isinstance(randomness, Materialized)` at trace time (not "inside" the traced/jitted body, so by A's reading it is not "a runtime branch inside the default path" — the branch happens at Python trace time, before jit).
- B reads "separate path" as literally a separate top-level function: `solve_sde()` for default, `solve_sde_materialized()` for FR-5, with no shared dispatch point at all.
- Both implementations satisfy "never a runtime branch inside the default [traced] path" under a defensible reading. But callers of the Solver Layer now face two incompatible APIs depending on which engineer's code lands: a single function with a config-object switch, vs. two distinctly-named entrypoints. `quantscenariobench.api.simulate()` (AD-9's sole importer of `solver`) cannot be written against both.
- **Gap:** "Separate construction path" is never given a concrete API shape (new function? new class? constructor kwarg dispatched at trace time?), so "never a runtime branch" is satisfiable by readings that still produce divergent public shapes of the Solver Layer.

---

## AD-4 — Solver Layer wraps `diffrax` exclusively

**Rule as written:** `diffrax` imported only inside `quantscenariobench.solver`; a Market Model supplies drift/diffusion functions and parameters; never constructs `Term`/`SaveAt`/solver instance itself.

**Scenario:** Engineer A implements `BlackScholes`, Engineer B implements `Heston`, each supplying "drift/diffusion functions and parameters" to the Solver Layer per the Rule.

- A's `BlackScholes.drift_diffusion()` returns plain JAX-traceable Python functions closed over the model's own `equinox.Module` fields, called as `fn(t, y)`.
- B's `Heston` instead returns functions that expect `fn(t, y, args)` because Heston's diffusion needs the variance-process state passed explicitly as `args` (diffrax's own `Term` signature supports an `args` slot) — a calling convention AD-4 never forecloses since it only says "drift/diffusion functions and parameters," not their signature.
- Both never construct a `diffrax.Term`/`SaveAt`/solver instance — fully compliant with the Rule's literal prohibition. But the Solver Layer now needs model-specific branching to know whether to call the supplied drift fn with 2 or 3 positional arguments — exactly the "incompatible ... that both 'pass' yet produce inconsistent Scenario semantics" failure AD-4's own "Prevents" clause claims to rule out. The Rule restricts *where diffrax is imported*, not the *shape of the producer/consumer contract* crossing that boundary — so the thing AD-4 says it prevents is not actually prevented by the Rule as written.
- **Gap:** AD-4 is solely an import-location constraint; the TimeGrid→SaveAt mapping and the drift/diffusion function signature (explicitly named in "Prevents") are never pinned by the Rule itself, only gestured at as a goal.

---

## AD-5 — Dataset export is generic over the `Scenario` schema

**Rule as written:** Export derives Parquet columns by pytree-flattening a `Scenario`; imports `quantscenariobench.interface` only, never a concrete Market Model.

**Scenario:** Engineer A implements the Parquet column-naming logic; Engineer B implements the Hugging Face dataset-card / upload logic, both inside `quantscenariobench.export` and both reading only AD-5 + AD-8.

- A's flattener joins pytree paths with dots: a nested `metadata.parameters.kappa` leaf becomes column `metadata.parameters.kappa`.
- B's dataset-card generator (also producing column-level documentation per FR-15) assumes flattened leaf names are joined with double-underscore (`metadata__parameters__kappa`) — a common Hugging Face / Arrow-safe convention since dots are sometimes reserved in column-name parsing.
- Both are "deriving Parquet columns by pytree-flattening a Scenario" per the Rule's letter; neither imports a concrete Market Model. The resulting Parquet file and its accompanying dataset card disagree on column names for the same data.
- **Gap:** AD-5 fixes the *flattening source* (pytree-flatten of Scenario) but never fixes the *naming/joining convention* for nested leaf paths, nor whether static `metadata` fields become columns (repeated per row) vs. top-level dataset/file attributes — both are "generic over Scenario" by a literal reading.

---

## AD-6 — `equinox` is a project-wide pytree convention

**Rule as written:** Every JAX-PyTree-typed dataclass — Market Model parameter classes and `Scenario` alike — is an `equinox.Module`; none uses `jax.tree_util.register_pytree_node_class` directly.

**Scenario:** Engineer A implements `BlackScholes` parameters; Engineer B implements `SABR` (a future model, but the same divergence applies to `Heston` vs `RoughBergomi` today).

- A makes every field of `BlackScholesParams` a dynamic (traced) `eqx.Module` field, including the model's display name/label string, because nothing in AD-6 says fields must be split dynamic/static — that split is only asserted for `Scenario` (AD-2), not for Market Model parameter classes.
- B, on `SABRParams`, marks string/categorical fields `eqx.field(static=True)` by analogy with AD-2's Scenario pattern.
- Both are `equinox.Module` subclasses, neither hand-registers a pytree — fully AD-6-compliant. But `jax.jit`/`vmap` over a *list* of mixed Market Models (e.g., batched cross-model benchmarking, implied by the "Market Model Zoo" framing in the PRD) breaks: A's params re-trace per distinct string value if ever static-promoted by other code expecting AD-2-style discipline, while B's choice means `SABRParams` is not even comparison-equal in the same way `tree_flatten` would assume A's class behaves.
- **Gap:** AD-6 mandates *which base class* models must use, not the dynamic/static split discipline for Market Model parameter classes themselves — only `Scenario`'s split is fixed (AD-2). Two models can each freely choose, and the Rule's text does not forbid it.

---

## AD-7 — float64 (JAX x64) is the fixed v1 precision policy

**Rule as written:** x64 mode "enabled once, at process/package entry." No Market Model or Solver Layer code overrides dtype per-call.

**Scenario:** Engineer A owns `quantscenariobench/__init__.py`; Engineer B owns `quantscenariobench/api/__init__.py`, working independently one level down, each believing "process/package entry" refers to their own module.

- A puts `jax.config.update("jax_enable_x64", True)` as a top-level side effect in `quantscenariobench/__init__.py` (the literal "package entry").
- B, not coordinating with A and reading "process/package entry" as "wherever the public API is first touched," puts the identical call in `quantscenariobench/api/__init__.py`, guarded by `if not jax.config.jax_enable_x64:` (defensive idempotency).
- Both individually comply with "enabled once... at... entry" and "no override per-call." But: (a) if a caller does `from quantscenariobench.models import BlackScholes` without ever importing `quantscenariobench` or `quantscenariobench.api` (a path AD-9's diagram explicitly allows — `caller → models` directly, no edge through `api`), x64 is never enabled at all, silently downgrading every model to float32 with no error; (b) if both A's and B's guards exist, the "enabled once" framing is itself violated by construction (two call sites), even though each individually intends idempotency.
- **Gap:** "Process/package entry" is not a single, unambiguous import path given AD-9's diagram shows callers may import `models` (and, by extension, `interface`) without ever importing `api` or top-level `quantscenariobench`. The Rule assumes a single entry point that the dependency diagram does not actually guarantee gets hit.

---

## AD-8 — Metadata's minimum guaranteed field set is fixed

**Rule as written:** `Scenario.metadata` always carries seed, PRNG key info, Market Model name and version-stamped parameter values, `TimeGrid` reference, library version, dataset version, generation timestamp. No omission, no renaming.

**Scenario:** Engineer A implements `Heston`'s metadata population; Engineer B implements `RoughBergomi`'s, both filling the same fixed field names from the Consistency Conventions table (`seed`, `prng_key_info`, `model_name`, `model_version`, `parameters`, `time_grid`, `library_version`, `dataset_version`, `generated_at`).

- A populates `parameters` as a flat `dict[str, float]`: `{"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04}`.
- B populates `parameters` as the model's own `eqx.Module` parameter pytree object itself (not unpacked to a dict), reasoning that "version-stamped parameter values" most naturally means "the actual versioned parameter object," and nothing in AD-8 or the Consistency Conventions table says the field's *value type* must be a plain dict.
- Both use the exact field name `parameters` (per the fixed naming convention) and both supply "version-stamped parameter values." AD-5's generic pytree-flattener now produces a flat Parquet column per scalar param for A's Heston rows, but for B's RoughBergomi rows it either produces a single opaque object column or a differently-shaped set of nested columns — the per-model column schema of the supposedly model-agnostic export is now silently model-dependent, which is precisely what AD-5's "Prevents" clause (no model-specific coupling re-entering export) was meant to rule out, and AD-8 does not forbid it.
- **Gap:** AD-8 (and the Consistency Conventions table) fixes field *names* exhaustively but never fixes field *value types/shapes* — "parameters" as dict-of-scalars vs. nested-pytree-object is left open, and that choice directly determines whether AD-5's generic flattening actually produces a uniform schema across models.

---

## AD-9 — Dependency direction is one-way: Models → Interface ← Solver/API/Export

**Rule as written:** A Market Model module may import only `quantscenariobench.interface`. `solver`/`api`/`export` may import `interface` (and, for `solver` only, third-party `diffrax`/`equinox`). Only `api` may import concrete Market Models, as caller-supplied arguments, never hardcoded.

**Scenario:** Engineer A implements `quantscenariobench.testing` (FR-11 conformance suite); Engineer B implements `quantscenariobench.export`, both reading AD-9 + the diagram literally.

- The diagram shows `testing → interface` only. AD-9's Rule text, however, never actually mentions `testing` at all — it enumerates `solver`, `api`, `export`, and "a Market Model module," but the conformance suite's permitted imports are asserted only in the diagram, not in the Rule's prose.
- A, building the FR-11 conformance suite, needs to run the suite against the *real* `BlackScholes`/`Heston`/`RoughBergomi` models (not just the test-only dummy) to be useful as regression protection — so A has `quantscenariobench.testing` import `quantscenariobench.models` directly, reasoning that the Rule's prose restricts `solver`/`api`/`export`/Market-Model-modules, and says nothing prohibiting `testing` from importing `models`.
- B, building Export, reads the same prose and the diagram's `export → interface` edge as exhaustive, and additionally allows `export` to import `quantscenariobench.testing` to reuse the dummy Market Model for its own doctest/example generation (FR-15 dataset-card examples) — again because the prose never mentions `export`'s relationship to `testing` either way, only to `interface`.
- Neither A nor B's choice contradicts the Rule's literal prose (which is silent on `testing`'s allowed imports and totally silent on whether `export` may import `testing`). Yet both choices undermine AD-9's stated "Prevents" — a chain `export → testing → models` would let a concrete Market Model reach Export-relevant code paths indirectly, exactly the bypass-of-the-sole-integration-point AD-9 exists to forbid, just routed through `testing` instead of directly.
- **Gap:** the Rule's prose enumerates allowed imports for `models`, `solver`, `api`, `export` but never mentions `quantscenariobench.testing`'s import rules at all — only the diagram (a non-normative illustration per the doc's own framing of "see diagram") constrains it, and the diagram doesn't forbid `export → testing` or `testing → models` either, since those edges are simply absent rather than explicitly prohibited. Absence-of-edge is not stated to mean prohibition anywhere in the spine.

---

## Consistency Conventions table — additional gaps

1. **Naming row** fixes class names and `simulate`/`solve_sde` entrypoint names, but not file/module names within `quantscenariobench.models` (e.g., is `BlackScholes` in `models/black_scholes.py`, `models/blackscholes.py`, or `models/_black_scholes.py`?) — two engineers adding `Heston` and `RoughBergomi` could each pick a different file-naming scheme (snake_case vs. matching-class-case) with nothing in the table to arbitrate, producing an inconsistent `models/` directory.
2. **Metadata field names row** fixes the *keys* exhaustively (good) but, as shown under AD-8 above, never fixes value *types*, leaving the `parameters` field's shape (dict vs pytree-object) and the `time_grid` field's shape (raw array vs `TimeGrid` object vs serialized dict) open to divergence between two Market Model authors.
3. **Soft validation row** fixes the warning *class* (`QuantScenarioBenchValidationWarning`) but not the warning *message format/contents* convention (structured fields vs free-text) — two engineers implementing Feller-condition-equivalent checks for two different models could emit warnings with incompatible message shapes, which matters if any downstream tooling (dataset card generation, CI) parses warning text rather than just checking the class.
4. **State & cross-cutting row** says "no module outside `quantscenariobench.solver` touches a JAX `PRNGKey` split for simulation purposes" — but a Market Model's own stochastic *parameter sampling* (if a future model wants randomized initial conditions, or if FR-6 soft-validation needs a key for Monte-Carlo-based diagnostic checks) is arguably not "for simulation purposes" under one reading and is squarely "simulation" under another; the boundary phrase "for simulation purposes" is not defined anywhere, leaving two engineers free to each decide independently whether their PRNGKey usage counts.

## Dependency-direction diagram — additional gaps

1. The diagram shows `caller → models` and `caller → api` as separate edges with no arrow constraining their relative order or co-occurrence. Nothing in the diagram or AD-9 prevents one engineer's "caller" code from importing only `models` and hand-rolling its own diffrax-free Euler-Maruyama loop over a Market Model's drift/diffusion methods (since `models` exposes them, per AD-1, with no access modifier hiding them from non-`solver` callers) — fully bypassing the Solver Layer that AD-4 was designed to make the sole diffrax integration point. AD-4's Rule restricts what a *Market Model* may construct, not what arbitrary caller code may do with a Market Model's public methods once obtained.
2. The diagram has no edge for `quantscenariobench.testing → quantscenariobench.models`, but as shown under AD-9 above, nothing forbids adding one — the diagram constrains by inclusion, never states it is exhaustive/closed-world, and the spine never says "edges not shown are prohibited."
3. Import direction is shown at the *package* level only; nothing in the diagram or AD-9 prevents a circular *symbol-level* dependency disguised as one-directional package imports — e.g., `interface.py` defining `MarketModel` with a method whose default implementation lazily does `from quantscenariobench.solver import solve_sde` inside the function body (a deferred/local import, not a module-level one) to provide a convenience `model.simulate_self()` helper. Module-level static analysis of "imports" (which is presumably what AD-9 enforcement would use, e.g. import-linter) would not even flag this without dedicated configuration, since the Rule's prose talks about "import" without specifying module-level vs. any-level.

---

## Summary count

Total concrete divergence holes identified: **9 AD-level + 4 Consistency Convention + 3 diagram-level = 16**, all reachable via literal compliance with the spine's Rule text as written.
