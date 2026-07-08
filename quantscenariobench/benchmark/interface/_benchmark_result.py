from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """The terminal, JSON-serializable artifact of one run_benchmark() call (AD-17, AD-24).

    A plain immutable dataclass — never an equinox.Module — because a
    BenchmarkResult is a terminal artifact, never re-traced through
    jit/vmap, unlike Scenario (AD-2). Every field is JSON-native
    (str, float, int, dict, list); no JAX arrays, no eqx.Module fields.

    rebalance_schedule is additive (FR-44, AD-33): the plain-dict
    materialization of the RebalanceSchedule run_benchmark() was called
    with (e.g. {"k": 21}), or None for buy-and-hold (the default and
    every result published before this field existed) — kept as a plain
    dict, not a RebalanceSchedule reference, so old published JSON
    lacking this key still loads via the dataclass field's own default
    (AC7), with no custom reconstruction required.

    cost_model is additive (FR-45, AD-34): the plain-dict materialization
    of the active TransactionCostModel (e.g.
    {"model": "ProportionalCost", "one_way_bps": 10.0}), or None when no
    cost model was supplied — the same plain-dict, not-a-typed-reference
    posture as rebalance_schedule, for the same backward-compatible-load
    reason (AC1).
    """

    strategy_name: str
    strategy_parameters: dict
    metrics: dict[str, float]
    asset_scenario_ids: list
    time_grid_reference: str
    library_version: str
    generated_at: str
    rebalance_schedule: Optional[dict] = None
    cost_model: Optional[dict] = None
