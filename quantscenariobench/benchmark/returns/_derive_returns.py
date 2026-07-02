from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...interface import Scenario


def derive_returns(scenario: Scenario) -> Float[Array, " t"]:
    """Simple/arithmetic period returns from a Scenario (FR-28, AD-16).

    return(t) = (price(t) - price(t-1)) / price(t-1), computed once per
    TimeGrid step directly from scenario.observation. Requires
    scenario.observation to be a one-dimensional, strictly-positive price
    series (AD-22); a Scenario whose observation is not of this shape is
    not benchmark-layer-usable and is rejected.

    The shape check is static (safe under jit, since ndim is always
    concrete); the positivity check is data-dependent, so it is enforced
    via equinox.error_if — which raises at runtime without breaking
    tracing — keeping derive_returns jit-compatible end to end (AD-25).
    """
    observation = scenario.observation
    if observation.ndim != 1:
        raise ValueError(
            "derive_returns requires scenario.observation to be a "
            f"one-dimensional price series; got shape {observation.shape} "
            "(AD-22)"
        )
    observation = eqx.error_if(
        observation,
        jnp.any(observation <= 0),
        "derive_returns requires scenario.observation to be a "
        "strictly-positive price series (AD-22)",
    )
    return (observation[1:] - observation[:-1]) / observation[:-1]
