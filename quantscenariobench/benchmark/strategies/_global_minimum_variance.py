from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import BaselineStrategy, PortfolioWeights
from ..solver import solve_allocation


class GlobalMinimumVariance(BaselineStrategy):
    """Minimizes portfolio variance given historical_returns' covariance (FR-21).

    long_only=False: closed-form covariance inversion via jax.numpy.linalg,
    fully JAX-native, no solver call (AD-14, AD-25). long_only=True: the
    long-only-constrained quadratic program, solved via
    quantscenariobench.benchmark.solver.solve_allocation
    (scipy.optimize.minimize, SLSQP) — the project's first (deliberately
    bounded) non-JAX-native dependency.
    """

    long_only: bool

    def allocate(self, historical_returns: Float[Array, "t n"]) -> PortfolioWeights:
        n = historical_returns.shape[1]
        covariance = jnp.cov(historical_returns, rowvar=False)

        if self.long_only:
            weights = solve_allocation(covariance)
        else:
            ones = jnp.ones((n,))
            inv_cov_ones = jnp.linalg.solve(covariance, ones)
            weights = inv_cov_ones / jnp.sum(inv_cov_ones)

        return PortfolioWeights(weights, n_assets=n)
