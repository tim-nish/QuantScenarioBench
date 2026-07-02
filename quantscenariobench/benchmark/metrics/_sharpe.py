from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def sharpe_ratio(returns: Float[Array, " t"]) -> Float[Array, ""]:
    """Sharpe Ratio of a Portfolio Return series (FR-16).

    Risk-free rate 0, no annualization: mean(returns) / std(returns).
    Returns 0.0 (rather than raising or NaN/inf) when the return series
    has zero variance (AD-18).
    """
    mean = jnp.mean(returns)
    std = jnp.std(returns)
    is_degenerate = std == 0.0
    safe_std = jnp.where(is_degenerate, 1.0, std)
    return jnp.where(is_degenerate, 0.0, mean / safe_std)


sharpe_ratio.name = "sharpe_ratio"
