from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """The terminal, JSON-serializable artifact of one run_benchmark() call (AD-17, AD-24).

    A plain immutable dataclass — never an equinox.Module — because a
    BenchmarkResult is a terminal artifact, never re-traced through
    jit/vmap, unlike Scenario (AD-2). Every field is JSON-native
    (str, float, int, dict, list); no JAX arrays, no eqx.Module fields.
    """

    strategy_name: str
    strategy_parameters: dict
    metrics: dict[str, float]
    asset_scenario_ids: list
    time_grid_reference: str
    library_version: str
    generated_at: str
