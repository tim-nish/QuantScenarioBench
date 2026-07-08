from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Float

from ._portfolio_weights import PortfolioWeights


class PolicyStrategy(eqx.Module, ABC):
    """Abstract base for a time-varying allocation policy (AD-33, FR-44).

    Unlike BaselineStrategy/ForecastOptimizer's single allocate() call,
    a PolicyStrategy is called once per rebalance date by run_benchmark()
    (AD-33): allocate_sequence(observed_returns) receives every return
    observed strictly before the current rebalance date — historical_returns
    concatenated with evaluation returns up to, but never including, that
    date (the no-lookahead invariant, FR-44) — and returns the
    PortfolioWeights to hold until the next rebalance (or, with
    rebalance_schedule=None, for the entire evaluation window).
    """

    @abstractmethod
    def allocate_sequence(
        self, observed_returns: Float[Array, "t n"]
    ) -> PortfolioWeights:
        raise NotImplementedError
