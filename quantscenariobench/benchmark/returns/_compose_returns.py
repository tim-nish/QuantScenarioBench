from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...interface import Scenario
from ._derive_returns import derive_returns


def compose_returns(
    scenarios: Sequence[Scenario], path_index: Optional[int] = None
) -> Float[Array, "t n"]:
    """Compose N single-asset Scenarios into one aligned multi-asset
    returns matrix (FR-26, FR-46).

    Every constituent Scenario must carry an identical TimeGrid (same
    length, same time points); a mismatch raises before any return
    derivation is attempted (AD-22) — no implicit padding, truncation, or
    resampling to reconcile misaligned grids.

    path_index=None (the default, unchanged since before Story 10.3):
    every constituent Scenario's observation must already be a single
    selected 1-D path — today's benchmark-boundary convention.

    path_index=<int> (FR-46, AD-35): selects the same path index out of
    every constituent Scenario's full (n_paths, len(time_grid))
    observation ensemble — reusing already-simulated paths for
    distributional evaluation across R repeats, never re-simulating.
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

    per_asset_returns = [
        derive_returns(scenario, path_index) for scenario in scenarios
    ]
    return jnp.stack(per_asset_returns, axis=1)
