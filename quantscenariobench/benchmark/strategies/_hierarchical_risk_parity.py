from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import BaselineStrategy, PortfolioWeights
from ..solver import hierarchical_risk_parity_weights


class HierarchicalRiskParity(BaselineStrategy):
    """Lopez de Prado (2016)'s Hierarchical Risk Parity baseline (FR-48).

    Covariance-robust: no matrix inversion, so it behaves where GMV's
    inverse is unstable. Always calls
    quantscenariobench.benchmark.solver.hierarchical_risk_parity_weights
    (scipy.cluster.hierarchy, scipy.spatial.distance) — never imports
    scipy directly (AD-14).

    linkage_method is a required-by-default, recorded constructor field
    (AD-15's precedent, mirroring GlobalMinimumVariance.long_only /
    CVaROptimization.confidence_level) so it is always part of the
    strategy's recorded parameters, keeping a later BenchmarkResult
    reproducible.
    """

    linkage_method: str = "single"

    def allocate(self, historical_returns: Float[Array, "t n"]) -> PortfolioWeights:
        n = historical_returns.shape[1]
        covariance = jnp.cov(historical_returns, rowvar=False)
        weights = hierarchical_risk_parity_weights(covariance, self.linkage_method)
        return PortfolioWeights(weights, n_assets=n)
