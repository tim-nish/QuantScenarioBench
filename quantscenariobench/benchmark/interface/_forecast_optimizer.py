from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Float

from ._portfolio_weights import PortfolioWeights


class ForecastOptimizer(eqx.Module, ABC):
    """Abstract base for portfolio allocation strategies that additionally
    consume an externally supplied forecast (AD-13).

    forecast is a fixed-shape point forecast — one predicted next-period
    return per constituent asset, matching historical_returns' asset
    ordering (AD-21). Distributional/quantile forecasts are out of scope
    for v1.
    """

    @abstractmethod
    def allocate(
        self,
        historical_returns: Float[Array, "t n"],
        forecast: Float[Array, " n"],
    ) -> PortfolioWeights:
        raise NotImplementedError
