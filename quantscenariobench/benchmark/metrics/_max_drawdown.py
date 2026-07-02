from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def max_drawdown(returns: Float[Array, " t"]) -> Float[Array, ""]:
    """Maximum Drawdown of a Portfolio Return series (FR-18).

    Wealth is reconstructed from returns via cumulative compounding
    (wealth(0) implicit at 1.0). Maximum Drawdown is the largest
    peak-to-trough decline of that wealth path, reported as a
    non-positive fraction (e.g. -0.2 for a 20% drawdown).
    """
    wealth = jnp.cumprod(1.0 + returns)
    running_peak = jax.lax.cummax(wealth)
    drawdown = (wealth - running_peak) / running_peak
    return jnp.min(drawdown)


max_drawdown.name = "max_drawdown"
