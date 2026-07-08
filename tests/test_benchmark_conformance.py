"""
Story 4.2 — Portfolio Optimizer Conformance Suite
(incl. Test-Only Dummy ForecastOptimizer)

Covers all acceptance criteria from GitHub Issue #25.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC: quantscenariobench.benchmark.testing imports only from
# quantscenariobench.benchmark.interface and test tooling — never from
# quantscenariobench.benchmark.strategies or quantscenariobench.benchmark.runner (AD-19)
# ---------------------------------------------------------------------------

_FORBIDDEN_TESTING_IMPORT = re.compile(
    r"(?:import|from)\s+quantscenariobench\.benchmark\.(strategies|runner)\b"
)


def test_benchmark_testing_never_imports_strategies_or_runner():
    testing_dir = _pkg_root() / "benchmark" / "testing"
    violations = []
    for py_file in testing_dir.rglob("*.py"):
        if _FORBIDDEN_TESTING_IMPORT.search(py_file.read_text()):
            violations.append(str(py_file.relative_to(_pkg_root().parent)))
    assert not violations, (
        "AD-19 violation: quantscenariobench.benchmark.testing must never "
        f"import benchmark.strategies/runner: {violations}"
    )


# ---------------------------------------------------------------------------
# AC: the dummy ForecastOptimizer's allocate() returns a PortfolioWeights
# satisfying all of AD-20's invariants (FR-25)
# ---------------------------------------------------------------------------

def test_dummy_forecast_optimizer_returns_valid_portfolio_weights():
    from quantscenariobench.benchmark.interface import ForecastOptimizer, PortfolioWeights
    from quantscenariobench.benchmark.testing import DummyForecastOptimizer

    dummy = DummyForecastOptimizer()
    assert isinstance(dummy, ForecastOptimizer)

    historical_returns = jnp.ones((20, 5))
    forecast = jnp.array([0.01, -0.02, 0.03, 0.0, -0.01])

    weights = dummy.allocate(historical_returns, forecast)
    assert isinstance(weights, PortfolioWeights)
    assert weights.weights.shape == (5,)
    assert bool(jnp.all(weights.weights >= 0))
    assert bool(jnp.abs(jnp.sum(weights.weights) - 1.0) <= 1e-6)


# ---------------------------------------------------------------------------
# AC: the dummy ForecastOptimizer is not exported from
# quantscenariobench.benchmark.strategies and exists only inside
# quantscenariobench.benchmark.testing (FR-25, mirrors FR-11)
# ---------------------------------------------------------------------------

def test_dummy_forecast_optimizer_not_exported_outside_testing():
    from quantscenariobench.benchmark import interface as benchmark_interface
    assert not hasattr(benchmark_interface, "DummyForecastOptimizer"), \
        "DummyForecastOptimizer must not be exported from benchmark.interface"

    strategies_dir = _pkg_root() / "benchmark" / "strategies"
    if strategies_dir.is_dir():
        for py_file in strategies_dir.rglob("*.py"):
            assert "DummyForecastOptimizer" not in py_file.read_text(), (
                f"{py_file} must not reference DummyForecastOptimizer "
                "(test-only, FR-25)"
            )

    from quantscenariobench.benchmark.testing import DummyForecastOptimizer
    assert DummyForecastOptimizer.__module__.startswith(
        "quantscenariobench.benchmark.testing"
    )


# ---------------------------------------------------------------------------
# AC: ABC-enforcement — a class subclassing BaselineStrategy/ForecastOptimizer
# without implementing allocate() raises on instantiation (AD-13)
# ---------------------------------------------------------------------------

def test_abc_enforcement_baseline_strategy():
    from quantscenariobench.benchmark.interface import BaselineStrategy
    from quantscenariobench.benchmark.testing import assert_abc_enforcement

    assert_abc_enforcement(BaselineStrategy)


def test_abc_enforcement_forecast_optimizer():
    from quantscenariobench.benchmark.interface import ForecastOptimizer
    from quantscenariobench.benchmark.testing import assert_abc_enforcement

    assert_abc_enforcement(ForecastOptimizer)


def test_abc_enforcement_raises_type_error_directly():
    from quantscenariobench.benchmark.interface import BaselineStrategy

    class IncompleteStrategy(BaselineStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteStrategy()


# ---------------------------------------------------------------------------
# AC: determinism — allocate() called twice with identical arguments returns
# bit-identical PortfolioWeights
# ---------------------------------------------------------------------------

def test_determinism_dummy_forecast_optimizer():
    from quantscenariobench.benchmark.testing import (
        DummyForecastOptimizer,
        assert_forecast_optimizer_conforms,
    )

    dummy = DummyForecastOptimizer()
    historical_returns = jnp.ones((10, 4))
    forecast = jnp.array([0.01, 0.02, -0.01, 0.0])

    assert_forecast_optimizer_conforms(dummy, historical_returns, forecast)


def test_determinism_baseline_strategy():
    from quantscenariobench.benchmark.interface import BaselineStrategy, PortfolioWeights
    from quantscenariobench.benchmark.testing import assert_baseline_strategy_conforms

    class EqualWeightish(BaselineStrategy):
        def allocate(self, historical_returns):
            n = historical_returns.shape[1]
            return PortfolioWeights(jnp.full((n,), 1.0 / n), n_assets=n)

    strat = EqualWeightish()
    historical_returns = jnp.ones((15, 3))

    assert_baseline_strategy_conforms(strat, historical_returns)


# ---------------------------------------------------------------------------
# AC: PortfolioWeights shape test — allocate() for an N-asset portfolio
# returns PortfolioWeights of shape (N,)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_assets", [1, 2, 5, 10])
def test_portfolio_weights_shape_matches_n_assets(n_assets):
    from quantscenariobench.benchmark.testing import (
        DummyForecastOptimizer,
        assert_portfolio_weights_valid,
    )

    dummy = DummyForecastOptimizer()
    historical_returns = jnp.ones((8, n_assets))
    forecast = jnp.zeros((n_assets,))

    weights = dummy.allocate(historical_returns, forecast)
    assert_portfolio_weights_valid(weights, n_assets)
    assert weights.weights.shape == (n_assets,)


# ---------------------------------------------------------------------------
# Story 9.1 (Issue #79) AC8: assert_metric_conforms covers the new Metric
# protocol the same way assert_baseline_strategy_conforms covers
# BaselineStrategy/ForecastOptimizer (FR-40, NFR-3 extended)
# ---------------------------------------------------------------------------

def _dummy_metric_context():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    from quantscenariobench.benchmark.metrics import MetricContext

    return MetricContext(
        portfolio_returns=jnp.array([0.01, -0.02, 0.03]),
        weights=PortfolioWeights(jnp.array([0.5, 0.5])),
        evaluation_returns=jnp.ones((3, 2)),
    )


def test_assert_metric_conforms_accepts_a_conforming_metric():
    from quantscenariobench.benchmark.testing import assert_metric_conforms

    class DummyMetric:
        name = "dummy_metric"
        direction = "higher_is_better"
        params = None

        def __call__(self, context):
            return jnp.mean(context.portfolio_returns)

    assert_metric_conforms(DummyMetric(), _dummy_metric_context())


def test_assert_metric_conforms_rejects_non_scalar_output():
    from quantscenariobench.benchmark.testing import assert_metric_conforms

    class NonScalarMetric:
        name = "non_scalar_metric"
        direction = "higher_is_better"
        params = None

        def __call__(self, context):
            return context.portfolio_returns

    with pytest.raises(AssertionError):
        assert_metric_conforms(NonScalarMetric(), _dummy_metric_context())


def test_assert_metric_conforms_rejects_invalid_direction():
    from quantscenariobench.benchmark.testing import assert_metric_conforms

    class InvalidDirectionMetric:
        name = "invalid_direction_metric"
        direction = "sideways"
        params = None

        def __call__(self, context):
            return jnp.mean(context.portfolio_returns)

    with pytest.raises(AssertionError):
        assert_metric_conforms(InvalidDirectionMetric(), _dummy_metric_context())


def test_assert_metric_conforms_covers_default_metrics():
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS
    from quantscenariobench.benchmark.testing import assert_metric_conforms

    context = _dummy_metric_context()
    for metric in DEFAULT_METRICS:
        assert_metric_conforms(metric, context)
