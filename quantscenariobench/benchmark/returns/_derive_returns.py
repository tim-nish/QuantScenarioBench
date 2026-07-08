from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...interface import Scenario


def derive_returns(
    scenario: Scenario, path_index: Optional[int] = None
) -> Float[Array, " t"]:
    """Simple/arithmetic period returns from a Scenario (FR-28, AD-16, FR-46).

    return(t) = (price(t) - price(t-1)) / price(t-1), computed once per
    TimeGrid step from a single strictly-positive price path (AD-22).

    path_index=None (the default, unchanged since before Story 10.3):
    requires scenario.observation to already be a one-dimensional price
    series — the existing benchmark-boundary convention (one selected
    path per Scenario) — and every prior caller keeps working unmodified.

    path_index=<int> (FR-46, AD-35): selects one path out of a Scenario's
    full (n_paths, len(time_grid)) observation ensemble — reusing an
    already-simulated Scenario for distributional evaluation across R
    repeats, never re-simulating — before applying the identical
    positivity check and return derivation to that path.

    The shape check is static (safe under jit, since ndim is always
    concrete); the positivity check is data-dependent, so it is enforced
    via equinox.error_if — which raises at runtime without breaking
    tracing — keeping derive_returns jit-compatible end to end (AD-25).
    """
    observation = scenario.observation
    if path_index is None:
        if observation.ndim != 1:
            raise ValueError(
                "derive_returns requires scenario.observation to be a "
                f"one-dimensional price series; got shape {observation.shape} "
                "(AD-22)"
            )
    else:
        if observation.ndim != 2:
            raise ValueError(
                "derive_returns with path_index requires scenario.observation "
                f"to be a 2-D (n_paths, t) ensemble; got shape "
                f"{observation.shape} (AD-22, FR-46)"
            )
        observation = observation[path_index]

    observation = eqx.error_if(
        observation,
        jnp.any(observation <= 0),
        "derive_returns requires scenario.observation to be a "
        "strictly-positive price series (AD-22)",
    )
    return (observation[1:] - observation[:-1]) / observation[:-1]
