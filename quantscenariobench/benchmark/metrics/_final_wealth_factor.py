from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def final_wealth_factor(returns: Float[Array, " t"]) -> Float[Array, ""]:
    """Final Wealth Factor of a Portfolio Return series (FR-19).

    The compounded growth of one unit of wealth over the full return
    series: prod(1 + returns). A value of 1.2 means the portfolio ended
    at 1.2x its starting wealth.

    See README "Metric Conventions" for the risk-free-rate, annualization,
    and compounding conventions shared across this metrics package.
    """
    return jnp.prod(1.0 + returns)


final_wealth_factor.name = "final_wealth_factor"
