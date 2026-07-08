"""
Story 4.1 — Portfolio Optimizer Interface Core Types
(BaselineStrategy, ForecastOptimizer, PortfolioWeights, BenchmarkResult)

Covers all acceptance criteria from GitHub Issue #24.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


# ---------------------------------------------------------------------------
# AC: quantscenariobench.benchmark.interface exports the four public types
# ---------------------------------------------------------------------------

def test_benchmark_interface_exports_all_four_types():
    from quantscenariobench.benchmark import interface
    for name in ("BaselineStrategy", "ForecastOptimizer", "PortfolioWeights", "BenchmarkResult"):
        assert hasattr(interface, name), f"benchmark.interface must export {name}"


# ---------------------------------------------------------------------------
# AC: BaselineStrategy / ForecastOptimizer cannot be instantiated directly
# ---------------------------------------------------------------------------

def test_baseline_strategy_direct_instantiation_raises_type_error():
    from quantscenariobench.benchmark.interface import BaselineStrategy
    with pytest.raises(TypeError, match="abstract"):
        BaselineStrategy()


def test_forecast_optimizer_direct_instantiation_raises_type_error():
    from quantscenariobench.benchmark.interface import ForecastOptimizer
    with pytest.raises(TypeError, match="abstract"):
        ForecastOptimizer()


# ---------------------------------------------------------------------------
# AC: BaselineStrategy is an equinox.Module ABC with one abstract method,
# allocate(historical_returns) -> PortfolioWeights (AD-13)
# ---------------------------------------------------------------------------

def test_baseline_strategy_is_equinox_module_abc():
    from quantscenariobench.benchmark.interface import BaselineStrategy
    assert issubclass(BaselineStrategy, eqx.Module)
    assert inspect.isabstract(BaselineStrategy)
    assert BaselineStrategy.__abstractmethods__ == frozenset({"allocate"})


def test_baseline_strategy_allocate_signature_is_historical_returns_only():
    from quantscenariobench.benchmark.interface import BaselineStrategy
    params = list(inspect.signature(BaselineStrategy.allocate).parameters)
    assert params == ["self", "historical_returns"]


def test_baseline_strategy_concrete_subclass_instantiates():
    from quantscenariobench.benchmark.interface import BaselineStrategy, PortfolioWeights

    class EqualWeightish(BaselineStrategy):
        def allocate(self, historical_returns):
            n = historical_returns.shape[1]
            return PortfolioWeights(jnp.full((n,), 1.0 / n))

    strat = EqualWeightish()
    assert isinstance(strat, BaselineStrategy)
    assert isinstance(strat, eqx.Module)
    weights = strat.allocate(jnp.ones((10, 4)))
    assert isinstance(weights, PortfolioWeights)


# ---------------------------------------------------------------------------
# AC: ForecastOptimizer is an equinox.Module ABC with one abstract method,
# allocate(historical_returns, forecast) -> PortfolioWeights (AD-13, AD-21)
# ---------------------------------------------------------------------------

def test_forecast_optimizer_is_equinox_module_abc():
    from quantscenariobench.benchmark.interface import ForecastOptimizer
    assert issubclass(ForecastOptimizer, eqx.Module)
    assert inspect.isabstract(ForecastOptimizer)
    assert ForecastOptimizer.__abstractmethods__ == frozenset({"allocate"})


def test_forecast_optimizer_allocate_signature_is_returns_and_forecast():
    from quantscenariobench.benchmark.interface import ForecastOptimizer
    params = list(inspect.signature(ForecastOptimizer.allocate).parameters)
    assert params == ["self", "historical_returns", "forecast"]


def test_forecast_optimizer_concrete_subclass_instantiates():
    from quantscenariobench.benchmark.interface import ForecastOptimizer, PortfolioWeights

    class DummyForecastOptimizer(ForecastOptimizer):
        def allocate(self, historical_returns, forecast):
            n = forecast.shape[0]
            return PortfolioWeights(jnp.full((n,), 1.0 / n))

    strat = DummyForecastOptimizer()
    assert isinstance(strat, ForecastOptimizer)
    weights = strat.allocate(jnp.ones((10, 3)), jnp.array([0.01, 0.02, -0.01]))
    assert isinstance(weights, PortfolioWeights)


# ---------------------------------------------------------------------------
# AC: PortfolioWeights invariants enforced at construction (AD-20)
# ---------------------------------------------------------------------------

def test_portfolio_weights_accepts_valid_vector():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    pw = PortfolioWeights(jnp.array([0.25, 0.25, 0.5]))
    assert len(pw) == 3


def test_portfolio_weights_rejects_sum_not_equal_to_one():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    with pytest.raises(ValueError):
        PortfolioWeights(jnp.array([0.5, 0.6]))


def test_portfolio_weights_accepts_sum_within_tolerance():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    # off by 5e-7, within the 1e-6 tolerance
    pw = PortfolioWeights(jnp.array([0.5, 0.5 - 5e-7]))
    assert len(pw) == 2


def test_portfolio_weights_rejects_negative_entry():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    with pytest.raises(ValueError):
        PortfolioWeights(jnp.array([1.2, -0.2]))


def test_portfolio_weights_rejects_n_assets_mismatch():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    with pytest.raises(ValueError):
        PortfolioWeights(jnp.array([0.5, 0.5]), n_assets=3)


def test_portfolio_weights_accepts_matching_n_assets():
    from quantscenariobench.benchmark.interface import PortfolioWeights
    pw = PortfolioWeights(jnp.array([0.5, 0.5]), n_assets=2)
    assert len(pw) == 2


# ---------------------------------------------------------------------------
# AC: BenchmarkResult is a plain, frozen, JSON-native dataclass (AD-17)
# ---------------------------------------------------------------------------

def test_benchmark_result_is_frozen_dataclass_not_equinox_module():
    from quantscenariobench.benchmark.interface import BenchmarkResult
    assert dataclasses.is_dataclass(BenchmarkResult)
    assert BenchmarkResult.__dataclass_params__.frozen is True
    assert not issubclass(BenchmarkResult, eqx.Module)


def test_benchmark_result_fields_are_json_native_types():
    from quantscenariobench.benchmark.interface import BenchmarkResult
    allowed = {str, float, int, dict, list}
    # "Optional[dict]" covers rebalance_schedule (FR-44, AD-33): an
    # additive, opt-in field whose JSON-native value is a dict or null.
    for f in dataclasses.fields(BenchmarkResult):
        assert f.type in {t.__name__ for t in allowed} or f.type in (
            "dict[str, float]",
            "Optional[dict]",
        ), f"BenchmarkResult.{f.name} has non-JSON-native annotation {f.type!r}"


def test_benchmark_result_round_trips_through_json():
    import json
    from quantscenariobench.benchmark.interface import BenchmarkResult

    result = BenchmarkResult(
        strategy_name="EqualWeight",
        strategy_parameters={},
        metrics={"sharpe_ratio": 1.23},
        asset_scenario_ids=["scenario-a", "scenario-b"],
        time_grid_reference="tg-0",
        library_version="1.0.0",
        generated_at="2026-07-02T00:00:00Z",
    )
    payload = json.dumps(dataclasses.asdict(result))
    restored = BenchmarkResult(**json.loads(payload))
    assert restored == result


def test_benchmark_result_is_immutable():
    from quantscenariobench.benchmark.interface import BenchmarkResult

    result = BenchmarkResult(
        strategy_name="EqualWeight",
        strategy_parameters={},
        metrics={},
        asset_scenario_ids=[],
        time_grid_reference="tg-0",
        library_version="1.0.0",
        generated_at="2026-07-02T00:00:00Z",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.strategy_name = "GlobalMinimumVariance"


# ---------------------------------------------------------------------------
# AC: benchmark-layer modules never import quantscenariobench.models or
# quantscenariobench.solver directly (AD-19)
# ---------------------------------------------------------------------------

_BENCHMARK_SUBPACKAGES = (
    "interface", "strategies", "metrics", "returns", "solver", "runner", "testing",
)
_FORBIDDEN = {"models", "solver"}


def _qsb_imports(source: str) -> set[str]:
    pkg = "quantscenariobench"
    return {
        m.group(1)
        for m in re.finditer(rf"(?:import|from)\s+{pkg}\.(\w+)", source)
    }


def test_benchmark_layer_never_imports_scenario_generation_directly():
    pkg_root = Path(__file__).parent.parent / "quantscenariobench"
    benchmark_root = pkg_root / "benchmark"
    violations: list[str] = []

    for sub in _BENCHMARK_SUBPACKAGES:
        sub_dir = benchmark_root / sub
        if not sub_dir.is_dir():
            continue
        for py_file in sub_dir.rglob("*.py"):
            # A benchmark.solver module importing quantscenariobench.solver
            # would be a false positive from substring matching, but the
            # forbidden set here is quantscenariobench.models/solver, i.e.
            # top-level scenario-generation modules, distinct from
            # quantscenariobench.benchmark.solver.
            illegal = _qsb_imports(py_file.read_text()) & _FORBIDDEN
            if illegal:
                violations.append(
                    f"{py_file.relative_to(pkg_root.parent)}: imports {illegal} (not allowed, AD-19)"
                )

    assert not violations, (
        "AD-19 benchmark-layer dependency direction violation:\n" + "\n".join(violations)
    )
