from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Float

from ._portfolio_weights import PortfolioWeights


class BaselineStrategy(eqx.Module, ABC):
    """Abstract base for fixed, non-learned portfolio allocation strategies (AD-13).

    Every concrete Traditional Baseline (Equal Weight, Global Minimum
    Variance, CVaR Optimization) subclasses this and implements only
    allocate(); none registers itself as a pytree by hand (AD-6).
    """

    @abstractmethod
    def allocate(
        self, historical_returns: Float[Array, "t n"]
    ) -> PortfolioWeights:
        raise NotImplementedError
