from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

_SUM_TOLERANCE = 1e-6


class PortfolioWeights(eqx.Module):
    """A validated portfolio allocation vector (AD-20).

    Every BaselineStrategy/ForecastOptimizer.allocate() call must return
    one of these. Three invariants are enforced at construction rather
    than left to strategy discipline: entries sum to 1.0 within 1e-6,
    every entry is >= 0 (long-only, universal for v1), and — when the
    caller supplies n_assets — weights has exactly that many entries.
    """

    weights: Float[Array, " n"]

    def __init__(
        self,
        weights: Float[Array, " n"],
        n_assets: int | None = None,
    ) -> None:
        weights = jnp.asarray(weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError(
                f"PortfolioWeights requires a 1-D array; got shape {weights.shape}"
            )
        if n_assets is not None and weights.shape[0] != n_assets:
            raise ValueError(
                f"PortfolioWeights has {weights.shape[0]} entries but "
                f"{n_assets} constituent assets were expected"
            )
        if not bool(jnp.all(weights >= 0)):
            raise ValueError(
                "PortfolioWeights requires every entry to be >= 0 "
                "(long-only is a v1-universal invariant, AD-20)"
            )
        weight_sum = jnp.sum(weights)
        if not bool(jnp.abs(weight_sum - 1.0) <= _SUM_TOLERANCE):
            raise ValueError(
                f"PortfolioWeights entries must sum to 1.0 within "
                f"{_SUM_TOLERANCE}; got {float(weight_sum)!r}"
            )
        object.__setattr__(self, "weights", weights)

    def __len__(self) -> int:
        return int(self.weights.shape[0])
