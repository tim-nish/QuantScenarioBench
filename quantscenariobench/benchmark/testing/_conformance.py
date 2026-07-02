"""Reusable conformance assertions for BaselineStrategy/ForecastOptimizer
implementors (FR-25).

This module imports only from quantscenariobench.benchmark.interface and
the standard library / jax (AD-19) — never
quantscenariobench.benchmark.strategies or quantscenariobench.benchmark.runner.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..interface import BaselineStrategy, ForecastOptimizer, PortfolioWeights


def assert_portfolio_weights_valid(weights: Any, n_assets: int) -> None:
    """Assert weights is a PortfolioWeights of the expected shape (AD-20).

    PortfolioWeights' own constructor already enforces the sum-to-one and
    non-negativity invariants at construction time; this assertion checks
    that allocate() actually returned the validated type, not a raw array,
    and that its shape matches the requested portfolio.
    """
    assert isinstance(weights, PortfolioWeights), \
        f"allocate() must return a PortfolioWeights, got {type(weights)!r}"
    assert weights.weights.shape == (n_assets,), \
        f"PortfolioWeights shape {weights.weights.shape} != expected ({n_assets},)"


def assert_deterministic_weights(weights_a: PortfolioWeights, weights_b: PortfolioWeights) -> None:
    """Assert two PortfolioWeights from identical inputs are bit-identical."""
    assert jnp.array_equal(weights_a.weights, weights_b.weights), \
        "allocate() is not deterministic: repeated calls with identical " \
        "arguments produced different PortfolioWeights"


def assert_abc_enforcement(abc_cls: type) -> None:
    """Assert a subclass of abc_cls missing allocate() raises on instantiation (AD-13)."""
    incomplete = type(f"Incomplete{abc_cls.__name__}", (abc_cls,), {})
    try:
        incomplete()
    except TypeError:
        return
    raise AssertionError(
        f"{abc_cls.__name__} subclass missing allocate() did not raise "
        "TypeError on instantiation"
    )


def assert_baseline_strategy_conforms(
    strategy: BaselineStrategy,
    historical_returns: Any,
) -> None:
    """Run the full conformance suite against a BaselineStrategy (FR-25).

    Checks PortfolioWeights shape/type and determinism across two calls
    with identical arguments.
    """
    n_assets = historical_returns.shape[1]

    weights_a = strategy.allocate(historical_returns)
    assert_portfolio_weights_valid(weights_a, n_assets)

    weights_b = strategy.allocate(historical_returns)
    assert_deterministic_weights(weights_a, weights_b)


def assert_forecast_optimizer_conforms(
    optimizer: ForecastOptimizer,
    historical_returns: Any,
    forecast: Any,
) -> None:
    """Run the full conformance suite against a ForecastOptimizer (FR-25).

    Checks PortfolioWeights shape/type and determinism across two calls
    with identical arguments.
    """
    n_assets = historical_returns.shape[1]

    weights_a = optimizer.allocate(historical_returns, forecast)
    assert_portfolio_weights_valid(weights_a, n_assets)

    weights_b = optimizer.allocate(historical_returns, forecast)
    assert_deterministic_weights(weights_a, weights_b)
