"""Test-only ForecastOptimizer for the conformance suite (FR-25).

DummyForecastOptimizer is a trivial equal-weight allocator whose only
purpose is to serve as a conforming ForecastOptimizer implementation in
conformance tests. It must not be exported from any non-testing module
(FR-25, AD-19) — mirrors DummyModel's treatment (FR-11).
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..interface import ForecastOptimizer, PortfolioWeights


class DummyForecastOptimizer(ForecastOptimizer):
    """Minimal ForecastOptimizer for conformance testing only.

    Ignores historical_returns and forecast values; allocates equal
    weight across the assets implied by forecast's shape. Exists solely
    to prove the ForecastOptimizer interface is satisfiable by an
    implementation the Runner was never written against.
    """

    def allocate(self, historical_returns: Any, forecast: Any) -> PortfolioWeights:
        n = forecast.shape[0]
        return PortfolioWeights(jnp.full((n,), 1.0 / n), n_assets=n)
