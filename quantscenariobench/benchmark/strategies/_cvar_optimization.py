from __future__ import annotations

from jaxtyping import Array, Float

from ..interface import BaselineStrategy, PortfolioWeights
from ..solver import solve_allocation


class CVaROptimization(BaselineStrategy):
    """Minimizes Conditional Value-at-Risk at a required confidence level (FR-22).

    confidence_level is a required constructor argument — never an
    internal hardcoded default (AD-15) — so it is always part of the
    strategy's recorded identity/parameters, keeping a later
    BenchmarkResult reproducible. The v1 default value for callers who
    don't specify one explicitly is 0.95, confirmed 2026-07-02.

    Always calls quantscenariobench.benchmark.solver.solve_allocation
    (scipy.optimize.linprog, Rockafellar-Uryasev formulation, AD-14) —
    never imports scipy directly.
    """

    confidence_level: float

    def allocate(self, historical_returns: Float[Array, "t n"]) -> PortfolioWeights:
        n = historical_returns.shape[1]
        weights = solve_allocation(
            returns=historical_returns, confidence_level=self.confidence_level
        )
        return PortfolioWeights(weights, n_assets=n)
