from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...interface import Scenario
from ._derive_returns import derive_returns


def compose_returns(scenarios: Sequence[Scenario]) -> Float[Array, "t n"]:
    """Compose N single-asset Scenarios into one aligned multi-asset
    returns matrix (FR-26).

    Every constituent Scenario must carry an identical TimeGrid (same
    length, same time points); a mismatch raises before any return
    derivation is attempted (AD-22) — no implicit padding, truncation, or
    resampling to reconcile misaligned grids.
    """
    if len(scenarios) == 0:
        raise ValueError("compose_returns requires at least one Scenario")

    reference_time_grid = scenarios[0].metadata.time_grid
    for scenario in scenarios[1:]:
        time_grid = scenario.metadata.time_grid
        if len(time_grid) != len(reference_time_grid) or not bool(
            jnp.array_equal(time_grid.t, reference_time_grid.t)
        ):
            raise ValueError(
                "compose_returns requires every constituent Scenario to "
                "share an identical TimeGrid (AD-22)"
            )

    per_asset_returns = [derive_returns(scenario) for scenario in scenarios]
    return jnp.stack(per_asset_returns, axis=1)
