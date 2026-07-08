"""
Story 6.2 — Benchmark Runner Orchestration & Extensibility Proof

Covers all acceptance criteria from GitHub Issue #31.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def _returns(key, t, n):
    return jax.random.normal(key, (t, n)) * 0.01


_HIST = _returns(jax.random.PRNGKey(0), 20, 3)
_EVAL = _returns(jax.random.PRNGKey(1), 10, 3)


# ---------------------------------------------------------------------------
# AC: allocate(historical_returns) called exactly once; PortfolioWeights
# applied unchanged across the full evaluation_returns window (FR-27, AD-23)
# ---------------------------------------------------------------------------

def test_run_benchmark_calls_baseline_allocate_once_and_applies_weights_unchanged(monkeypatch):
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    original_allocate = EqualWeight.allocate
    calls = {"count": 0}

    def spy(self, historical_returns):
        calls["count"] += 1
        return original_allocate(self, historical_returns)

    monkeypatch.setattr(EqualWeight, "allocate", spy)

    strat = EqualWeight()
    result = run_benchmark(strat, _HIST, _EVAL)

    assert calls["count"] == 1
    weights = jnp.full((3,), 1.0 / 3)
    expected_portfolio_returns = _EVAL @ weights
    expected_wealth_factor = float(jnp.prod(1.0 + expected_portfolio_returns))
    assert result.metrics["final_wealth_factor"] == pytest.approx(expected_wealth_factor)


# ---------------------------------------------------------------------------
# AC: ForecastOptimizer + forecast dispatches via isinstance and calls
# allocate(historical_returns, forecast) (AD-23)
# ---------------------------------------------------------------------------

def test_run_benchmark_dispatches_forecast_optimizer(monkeypatch):
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.testing import DummyForecastOptimizer

    original_allocate = DummyForecastOptimizer.allocate
    seen_args = {}

    def spy(self, historical_returns, forecast):
        seen_args["historical_returns"] = historical_returns
        seen_args["forecast"] = forecast
        return original_allocate(self, historical_returns, forecast)

    monkeypatch.setattr(DummyForecastOptimizer, "allocate", spy)

    dummy = DummyForecastOptimizer()
    forecast = jnp.array([0.01, -0.02, 0.03])
    run_benchmark(dummy, _HIST, _EVAL, forecast=forecast)

    assert jnp.array_equal(seen_args["historical_returns"], _HIST)
    assert jnp.array_equal(seen_args["forecast"], forecast)


# ---------------------------------------------------------------------------
# AC: ForecastOptimizer without forecast raises (AD-23)
# ---------------------------------------------------------------------------

def test_run_benchmark_requires_forecast_for_forecast_optimizer():
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.testing import DummyForecastOptimizer

    with pytest.raises(ValueError):
        run_benchmark(DummyForecastOptimizer(), _HIST, _EVAL)


# ---------------------------------------------------------------------------
# AC: BaselineStrategy with a caller-supplied forecast raises (AD-23)
# ---------------------------------------------------------------------------

def test_run_benchmark_rejects_forecast_for_baseline_strategy():
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    with pytest.raises(ValueError):
        run_benchmark(
            EqualWeight(), _HIST, _EVAL, forecast=jnp.array([0.01, 0.0, -0.01])
        )


# ---------------------------------------------------------------------------
# AC: run_benchmark's source contains no strategy-specific branching beyond
# the isinstance(ForecastOptimizer) dispatch (FR-27)
# ---------------------------------------------------------------------------

def _extract_function_source(src: str, function_name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"function {function_name} not found")


def test_run_benchmark_has_no_strategy_specific_branching():
    src = (_pkg_root() / "benchmark" / "runner" / "_run_benchmark.py").read_text()
    run_benchmark_src = _extract_function_source(src, "run_benchmark")

    isinstance_calls = re.findall(r"isinstance\(([^)]*)\)", run_benchmark_src)
    assert isinstance_calls == ["strategy, ForecastOptimizer"], (
        f"run_benchmark's body must contain no strategy-specific branching "
        f"beyond isinstance(strategy, ForecastOptimizer); found: {isinstance_calls}"
    )

    concrete_strategy_names = ("EqualWeight", "GlobalMinimumVariance", "CVaROptimization")
    for name in concrete_strategy_names:
        assert name not in src, (
            f"run_benchmark must never hardcode a concrete strategy ({name}); "
            "strategy behaviour lives entirely behind the Portfolio Optimizer "
            "Interface (AD-19)"
        )


def test_runner_never_imports_concrete_strategies_module():
    src = (_pkg_root() / "benchmark" / "runner" / "_run_benchmark.py").read_text()
    assert "benchmark.strategies" not in src and "from ..strategies" not in src


# ---------------------------------------------------------------------------
# AC: identical strategy + returns called twice produce identical
# BenchmarkResults (FR-27, NFR-6)
# ---------------------------------------------------------------------------

def test_run_benchmark_is_deterministic():
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    strat = GlobalMinimumVariance(long_only=True)
    result_a = run_benchmark(strat, _HIST, _EVAL)
    result_b = run_benchmark(strat, _HIST, _EVAL)

    # generated_at legitimately varies by wall-clock time between calls
    # (same convention as Metadata.generated_at, AD-8) — compare every
    # other field for exact equality.
    fields_to_compare = [
        f.name for f in dataclasses.fields(result_a) if f.name != "generated_at"
    ]
    for name in fields_to_compare:
        assert getattr(result_a, name) == getattr(result_b, name), name


# ---------------------------------------------------------------------------
# AC: the Story 4.2 conformance suite's dummy ForecastOptimizer runs
# successfully through the full pipeline with zero Runner changes
# (FR-25, AD-19)
# ---------------------------------------------------------------------------

def test_run_benchmark_with_dummy_forecast_optimizer():
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.testing import DummyForecastOptimizer

    forecast = jnp.array([0.01, -0.02, 0.03])
    result = run_benchmark(DummyForecastOptimizer(), _HIST, _EVAL, forecast=forecast)
    assert result.strategy_name == "DummyForecastOptimizer"
    assert set(result.metrics) == {
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
    }


# ---------------------------------------------------------------------------
# AC: each of the three Epic 5 Traditional Baselines runs successfully
# through the identical pipeline with zero Runner changes (FR-23)
# ---------------------------------------------------------------------------

def test_run_benchmark_with_each_epic5_baseline():
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import (
        CVaROptimization,
        EqualWeight,
        GlobalMinimumVariance,
    )

    strategies = [
        EqualWeight(),
        GlobalMinimumVariance(long_only=False),
        GlobalMinimumVariance(long_only=True),
        CVaROptimization(confidence_level=0.95),
    ]
    for strat in strategies:
        result = run_benchmark(strat, _HIST, _EVAL)
        assert result.strategy_name == type(strat).__name__
        assert set(result.metrics) == {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
        }


# ---------------------------------------------------------------------------
# AC: a metrics registry argument is iterated over generically, calling
# each MetricFn on the derived portfolio return series (AD-18)
# ---------------------------------------------------------------------------

def test_run_benchmark_iterates_custom_metrics_registry_generically():
    from quantscenariobench.benchmark.metrics import wrap_legacy_metric
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    def custom_metric(returns):
        return jnp.sum(returns)
    custom_metric.name = "total_return"

    result = run_benchmark(
        EqualWeight(), _HIST, _EVAL,
        metrics=(wrap_legacy_metric(custom_metric, direction="higher_is_better"),),
    )

    weights = jnp.full((3,), 1.0 / 3)
    expected = float(jnp.sum(_EVAL @ weights))
    assert set(result.metrics) == {"total_return"}
    assert result.metrics["total_return"] == pytest.approx(expected)


def test_run_benchmark_raises_on_duplicate_metric_names():
    from quantscenariobench.benchmark.metrics import wrap_legacy_metric
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    def metric_a(returns):
        return jnp.mean(returns)
    metric_a.name = "dup"

    def metric_b(returns):
        return jnp.std(returns)
    metric_b.name = "dup"

    with pytest.raises(ValueError):
        run_benchmark(
            EqualWeight(), _HIST, _EVAL,
            metrics=(
                wrap_legacy_metric(metric_a, direction="higher_is_better"),
                wrap_legacy_metric(metric_b, direction="higher_is_better"),
            ),
        )


# ---------------------------------------------------------------------------
# Story 9.1 (Issue #79) AC3: a trivial weight-dependent Metric implementing
# the new context-aware protocol runs through run_benchmark() with no
# Runner changes beyond this story (FR-40)
# ---------------------------------------------------------------------------

def test_run_benchmark_scores_a_weight_dependent_metric():
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    class SumOfSquaredWeights:
        name = "sum_of_squared_weights"
        direction = "lower_is_better"
        params = None

        def __call__(self, context):
            return jnp.sum(context.weights.weights ** 2)

    result = run_benchmark(EqualWeight(), _HIST, _EVAL, metrics=(SumOfSquaredWeights(),))

    n_assets = _HIST.shape[1]
    expected = float(jnp.sum(jnp.full((n_assets,), 1.0 / n_assets) ** 2))
    assert result.metrics == {"sum_of_squared_weights": pytest.approx(expected)}
