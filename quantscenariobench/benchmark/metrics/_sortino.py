from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def sortino_ratio(returns: Float[Array, " t"]) -> Float[Array, ""]:
    """Sortino Ratio of a Portfolio Return series (FR-17).

    Risk-free rate 0 (minimum acceptable return 0), no annualization:
    mean(returns) / downside_deviation(returns), where downside deviation
    is the root-mean-square of the negative part of each period's return
    (periods with non-negative returns contribute 0). Returns 0.0 (rather
    than raising) when there are no negative returns (AD-18).
    """
    mean = jnp.mean(returns)
    downside = jnp.minimum(returns, 0.0)
    downside_deviation = jnp.sqrt(jnp.mean(downside ** 2))
    is_degenerate = downside_deviation == 0.0
    safe_downside_deviation = jnp.where(is_degenerate, 1.0, downside_deviation)
    return jnp.where(is_degenerate, 0.0, mean / safe_downside_deviation)


sortino_ratio.name = "sortino_ratio"
