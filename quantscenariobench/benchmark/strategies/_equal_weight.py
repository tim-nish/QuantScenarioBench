from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import BaselineStrategy, PortfolioWeights


class EqualWeight(BaselineStrategy):
    """Allocates equal weight across all constituent assets (FR-20).

    Ignores historical_returns' content entirely — the simplest possible
    standardized comparison anchor, requiring no historical-data fitting.
    Needs no solver call and computes its weights directly, entirely in
    jax.numpy (AD-25).
    """

    def allocate(self, historical_returns: Float[Array, "t n"]) -> PortfolioWeights:
        n = historical_returns.shape[1]
        weights = jnp.full((n,), 1.0 / n)
        return PortfolioWeights(weights, n_assets=n)
