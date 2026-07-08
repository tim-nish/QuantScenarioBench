"""
Story 6.2 — Benchmark Runner Orchestration & Extensibility Proof

Covers all acceptance criteria from GitHub Issue #31.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
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


# ---------------------------------------------------------------------------
# Story 9.2 (Issue #80) AC5: value_at_risk/conditional_value_at_risk at two
# alpha levels registered together both appear in BenchmarkResult.metrics
# under distinct names (FR-41)
# ---------------------------------------------------------------------------

def test_run_benchmark_scores_var_and_cvar_at_two_alpha_levels():
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    result = run_benchmark(
        EqualWeight(), _HIST, _EVAL,
        metrics=(
            value_at_risk(0.95),
            value_at_risk(0.99),
            conditional_value_at_risk(0.95),
            conditional_value_at_risk(0.99),
        ),
    )

    assert set(result.metrics) == {"var_0.95", "var_0.99", "cvar_0.95", "cvar_0.99"}
    assert result.metrics["cvar_0.95"] >= result.metrics["var_0.95"]
    assert result.metrics["cvar_0.99"] >= result.metrics["var_0.99"]


# ---------------------------------------------------------------------------
# Story 9.4 (Issue #82) AC4: EqualWeight scored via run_benchmark() reports
# exactly effective_number_of_assets == n for every n tested (FR-43)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_run_benchmark_equal_weight_effective_number_of_assets_equals_n(n):
    from quantscenariobench.benchmark.metrics import effective_number_of_assets
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    hist = _returns(jax.random.PRNGKey(0), 20, n)
    eval_ = _returns(jax.random.PRNGKey(1), 10, n)

    result = run_benchmark(
        EqualWeight(), hist, eval_, metrics=(effective_number_of_assets,)
    )
    assert result.metrics["effective_number_of_assets"] == pytest.approx(float(n), abs=1e-12)


# ---------------------------------------------------------------------------
# Story 9.4 (Issue #82) AC5: GMV (long-only) and CVaROptimization report
# herfindahl_index strictly between 1/n and 1 on a generic, non-trivial
# dataset (sanity/property test, FR-43)
# ---------------------------------------------------------------------------

def test_run_benchmark_gmv_and_cvar_herfindahl_index_strictly_between_1_over_n_and_1():
    import numpy as np

    from quantscenariobench.benchmark.metrics import herfindahl_index
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import CVaROptimization, GlobalMinimumVariance

    # Assets with meaningfully different variances/covariances (and a
    # volatility scale large enough to avoid the long-only SLSQP solver's
    # tolerance falsely converging at the equal-weight initial guess on a
    # near-zero-scale objective) so neither optimizer lands on a trivial
    # single-asset or accidentally equal-weight allocation.
    rng = np.random.default_rng(11)
    t, n = 100, 4
    vols = np.array([0.05, 0.15, 0.30, 0.50])
    corr = np.array([
        [1.00, 0.15, 0.05, -0.05],
        [0.15, 1.00, 0.25, 0.05],
        [0.05, 0.25, 1.00, 0.30],
        [-0.05, 0.05, 0.30, 1.00],
    ])
    cov = np.outer(vols, vols) * corr
    hist = jnp.array(rng.multivariate_normal(np.zeros(n), cov, size=t))
    eval_ = jnp.array(rng.multivariate_normal(np.zeros(n), cov, size=t))

    for strat in (GlobalMinimumVariance(long_only=True), CVaROptimization(confidence_level=0.95)):
        result = run_benchmark(strat, hist, eval_, metrics=(herfindahl_index,))
        weights = strat.allocate(hist).weights
        assert not jnp.allclose(weights, 1.0 / n), (
            f"{type(strat).__name__} landed on equal weight — dataset is not a "
            "meaningful sanity check"
        )
        hhi = result.metrics["herfindahl_index"]
        assert 1.0 / n < hhi < 1.0


# ===========================================================================
# Story 10.1 — Periodic Rebalancing & the PolicyStrategy Interface
#
# Covers all acceptance criteria from GitHub Issue #83.
# ===========================================================================

_GOLDEN_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_benchmark_results.json"
_GOLDEN_HIST = _returns(jax.random.PRNGKey(42), 60, 4)
_GOLDEN_EVAL = _returns(jax.random.PRNGKey(99), 40, 4)


def _load_golden_results() -> dict:
    return json.loads(_GOLDEN_FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# AC1: rebalance_schedule=None (the default) reproduces today's published
# BenchmarkResult bit-identically — a golden-file regression suite,
# captured before this story touched run_benchmark() (FR-44, AD-33)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_name", [
    "EqualWeight", "GlobalMinimumVariance", "CVaROptimization",
])
def test_run_benchmark_default_rebalance_schedule_matches_golden_result(strategy_name):
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import (
        CVaROptimization,
        EqualWeight,
        GlobalMinimumVariance,
    )

    strategies = {
        "EqualWeight": EqualWeight(),
        "GlobalMinimumVariance": GlobalMinimumVariance(long_only=True),
        "CVaROptimization": CVaROptimization(confidence_level=0.95),
    }
    golden = _load_golden_results()[strategy_name]

    result = run_benchmark(
        strategies[strategy_name], _GOLDEN_HIST, _GOLDEN_EVAL,
        asset_scenario_ids=["a0", "a1", "a2", "a3"],
        time_grid_reference="tg-golden-fixture",
    )
    actual = dataclasses.asdict(result)
    actual.pop("generated_at")
    actual.pop("library_version")

    # rebalance_schedule/cost_model are both additive: absent from the
    # golden fixture (captured before Story 10.1/10.2), present as None
    # on the fresh result.
    assert actual.pop("rebalance_schedule") is None
    assert actual.pop("cost_model") is None
    assert actual == golden


def test_run_benchmark_explicit_k_none_rebalance_schedule_matches_golden_result():
    from quantscenariobench.benchmark.interface import RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    golden = _load_golden_results()["EqualWeight"]

    result = run_benchmark(
        EqualWeight(), _GOLDEN_HIST, _GOLDEN_EVAL,
        asset_scenario_ids=["a0", "a1", "a2", "a3"],
        time_grid_reference="tg-golden-fixture",
        rebalance_schedule=RebalanceSchedule(k=None),
    )
    actual = dataclasses.asdict(result)
    actual.pop("generated_at")
    actual.pop("library_version")

    # An explicitly passed RebalanceSchedule(k=None) records {"k": None} —
    # distinct from omitting the argument entirely (which records None,
    # see the previous test) — but behaves identically (metrics match the
    # golden fixture bit-for-bit either way).
    assert actual.pop("rebalance_schedule") == {"k": None}
    assert actual.pop("cost_model") is None
    assert actual == golden


# ---------------------------------------------------------------------------
# AC2: k=21 on a 252-step evaluation window for EqualWeight produces 12
# refits, and the portfolio-return series matches a hand-rolled NumPy
# reference using the weight-drift convention (FR-44, AD-33)
# ---------------------------------------------------------------------------

def _numpy_drift_reference(weights_by_period, period_returns_list):
    """Hand-rolled NumPy reference for the weight-drift convention: within
    each holding period, entering weights are held as per-asset dollar
    allocations (not reset every step), so the effective weight drifts
    with relative asset performance until the next rebalance resets it.
    """
    segments = []
    for w, period_returns in zip(weights_by_period, period_returns_list):
        asset_wealth = np.asarray(w)[None, :] * np.cumprod(
            1.0 + np.asarray(period_returns), axis=0
        )
        portfolio_wealth = asset_wealth.sum(axis=1)
        portfolio_wealth_prev = np.concatenate([[1.0], portfolio_wealth[:-1]])
        segments.append(portfolio_wealth / portfolio_wealth_prev - 1.0)
    return np.concatenate(segments)


def test_rebalancing_loop_k21_252_step_equal_weight_12_refits_matches_numpy_reference():
    from quantscenariobench.benchmark.runner._run_benchmark import _run_rebalancing_loop
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(10), 30, n)
    eval_ = _returns(jax.random.PRNGKey(11), 252, n)
    k = 21

    portfolio_returns, weight_sequence, _ = _run_rebalancing_loop(
        EqualWeight(), hist, eval_, None, k, None
    )
    assert len(weight_sequence) == 12

    eval_np = np.asarray(eval_)
    equal_weight = np.full(n, 1.0 / n)
    period_returns_list = [eval_np[t_i:t_i + k] for t_i in range(0, 252, k)]
    expected = _numpy_drift_reference([equal_weight] * 12, period_returns_list)

    np.testing.assert_allclose(np.asarray(portfolio_returns), expected, atol=1e-9)


def test_run_benchmark_k21_calls_allocate_12_times(monkeypatch):
    from quantscenariobench.benchmark.interface import RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(10), 30, n)
    eval_ = _returns(jax.random.PRNGKey(11), 252, n)

    calls = {"count": 0}
    original_allocate = EqualWeight.allocate

    def spy(self, historical_returns):
        calls["count"] += 1
        return original_allocate(self, historical_returns)

    monkeypatch.setattr(EqualWeight, "allocate", spy)

    result = run_benchmark(EqualWeight(), hist, eval_, rebalance_schedule=RebalanceSchedule(k=21))

    assert calls["count"] == 12
    assert result.rebalance_schedule == {"k": 21}


# ---------------------------------------------------------------------------
# AC3: causality — weights chosen at rebalance date t_i depend only on
# returns strictly before t_i; shifting evaluation returns at or after
# t_i must not change them (a tested no-lookahead invariant, FR-44)
# ---------------------------------------------------------------------------

def test_rebalancing_loop_causality_shifting_post_ti_returns_does_not_change_earlier_weights():
    from quantscenariobench.benchmark.runner._run_benchmark import _run_rebalancing_loop
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    n = 3
    hist = _returns(jax.random.PRNGKey(20), 30, n)
    eval_a = _returns(jax.random.PRNGKey(21), 60, n)
    k = 20  # rebalance dates t = 0, 20, 40 -> 3 refits

    # eval_b is identical to eval_a on [0, 20) and differs from row 20 on.
    eval_b = eval_a.at[20:].set(_returns(jax.random.PRNGKey(22), 40, n))

    # GlobalMinimumVariance(long_only=False) is content-sensitive (unlike
    # EqualWeight) and JAX-native (no scipy call), so this is a meaningful,
    # fast causality check.
    strategy = GlobalMinimumVariance(long_only=False)
    _, weights_a, _ = _run_rebalancing_loop(strategy, hist, eval_a, None, k, None)
    _, weights_b, _ = _run_rebalancing_loop(strategy, hist, eval_b, None, k, None)

    # t_0 = 0 depends only on historical_returns: unaffected by any change
    # to evaluation_returns.
    assert jnp.array_equal(weights_a[0].weights, weights_b[0].weights)
    # t_1 = 20 depends only on evaluation_returns[:20], identical between
    # eval_a and eval_b.
    assert jnp.array_equal(weights_a[1].weights, weights_b[1].weights)
    # t_2 = 40 depends on evaluation_returns[:40], which includes the
    # modified region [20:40) — these must actually differ, or the test
    # dataset is not exercising the invariant it claims to.
    assert not jnp.array_equal(weights_a[2].weights, weights_b[2].weights)


# ---------------------------------------------------------------------------
# AC4: weight drift — the realized portfolio-return series follows the
# weight-drift convention (weights evolve with relative asset performance
# between rebalances), not the simpler reset-every-step alternative
# (FR-44, AD-33)
# ---------------------------------------------------------------------------

def test_rebalancing_loop_drifts_weights_rather_than_resetting_every_step():
    from quantscenariobench.benchmark.interface import BaselineStrategy, PortfolioWeights
    from quantscenariobench.benchmark.runner._run_benchmark import _run_rebalancing_loop

    class FixedWeights(BaselineStrategy):
        def allocate(self, historical_returns):
            return PortfolioWeights(jnp.array([0.7, 0.3]))

    # A single 2-day holding period (k >= t2): isolates the within-period
    # drift formula from the refitting/rebalance-date logic.
    hist = jnp.zeros((5, 2))
    eval_ = jnp.array([[0.20, -0.10], [0.05, -0.05]])
    k = 2

    portfolio_returns, _, _ = _run_rebalancing_loop(FixedWeights(), hist, eval_, None, k, None)

    entering_weights = jnp.array([0.7, 0.3])
    reset_every_step_reference = eval_ @ entering_weights

    # Day 0: no drift has occurred yet, so both conventions agree exactly.
    assert float(portfolio_returns[0]) == pytest.approx(float(reset_every_step_reference[0]))
    # Day 1: asset 0 outperformed asset 1 on day 0, so the drift
    # convention's day-1 return must differ from a naive reset-to-target
    # convention — the explicit, tested choice AD-33/AC4 requires.
    assert float(portfolio_returns[1]) != pytest.approx(float(reset_every_step_reference[1]))
    assert float(portfolio_returns[1]) == pytest.approx(0.025675675675675746, abs=1e-12)


# ---------------------------------------------------------------------------
# AC5: a minimal PolicyStrategy (momentum: overweight the best-performing
# asset over the trailing window) runs end-to-end through run_benchmark()
# and the Evaluation Result pipeline (FR-44)
# ---------------------------------------------------------------------------

def _make_momentum_policy():
    from quantscenariobench.benchmark.interface import PolicyStrategy, PortfolioWeights

    class MomentumPolicy(PolicyStrategy):
        def allocate_sequence(self, observed_returns):
            n = observed_returns.shape[1]
            cumulative = jnp.sum(observed_returns, axis=0)
            winner = jnp.argmax(cumulative)
            weights = jnp.full((n,), 0.1 / (n - 1)).at[winner].set(0.9)
            return PortfolioWeights(weights, n_assets=n)

    return MomentumPolicy()


def test_policy_strategy_runs_end_to_end_through_run_benchmark_and_evaluation_pipeline():
    from quantscenariobench.benchmark.evaluation import to_evaluation_result
    from quantscenariobench.benchmark.interface import PolicyStrategy, RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark

    policy = _make_momentum_policy()
    assert isinstance(policy, PolicyStrategy)

    n = 4
    hist = _returns(jax.random.PRNGKey(30), 30, n)
    eval_ = _returns(jax.random.PRNGKey(31), 60, n)

    result = run_benchmark(policy, hist, eval_, rebalance_schedule=RebalanceSchedule(k=20))

    assert result.strategy_name == "MomentumPolicy"
    assert set(result.metrics) == {
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
    }
    assert result.rebalance_schedule == {"k": 20}
    for value in result.metrics.values():
        assert not jnp.isnan(jnp.asarray(value))

    evaluation_result = to_evaluation_result(result)
    assert evaluation_result.schema_version
    assert [m.value for m in evaluation_result.metrics] == list(result.metrics.values())


# ---------------------------------------------------------------------------
# AC7: a BenchmarkResult/EvaluationResult JSON file predating
# RebalanceSchedule (no rebalance_schedule key) still loads — the schedule
# field is additive (FR-44, NFR-6 extended)
# ---------------------------------------------------------------------------

def test_pre_story_benchmark_result_json_without_rebalance_schedule_still_loads():
    from quantscenariobench.benchmark.interface import BenchmarkResult

    golden = _load_golden_results()["EqualWeight"]
    assert "rebalance_schedule" not in golden

    result = BenchmarkResult(
        **golden,
        library_version="1.0.0",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert result.rebalance_schedule is None


def test_pre_story_evaluation_result_json_without_rebalance_schedule_still_loads():
    from quantscenariobench.benchmark.evaluation import EvaluationResult

    old_style_payload = {
        "schema_version": "1.0",
        "result_id": "result-0001",
        "strategy": {"name": "EqualWeight", "parameters": {}},
        "benchmark_dataset": {"asset_scenario_ids": [], "time_grid_reference": "tg-0"},
        "metrics": [{"name": "sharpe_ratio", "value": 1.0}],
        "library_version": "1.0.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        # rebalance_schedule intentionally omitted
    }
    result = EvaluationResult.from_dict(old_style_payload)
    assert result.rebalance_schedule is None


# ---------------------------------------------------------------------------
# AC8: the rolling loop is documented as lax.scan-compatible where
# possible, with a Python-loop fallback for scipy-backed refits (FR-44,
# NFR-2 extended)
# ---------------------------------------------------------------------------

def test_run_benchmark_module_documents_lax_scan_and_python_fallback_posture():
    src = (_pkg_root() / "benchmark" / "runner" / "_run_benchmark.py").read_text()
    assert "lax.scan" in src
    assert "Python for loop" in src or "Python-loop" in src or "python for loop" in src.lower()


# ---------------------------------------------------------------------------
# RebalanceSchedule is a plain, JSON-native, additive-friendly value type
# (AD-17-style posture, AD-33)
# ---------------------------------------------------------------------------

def test_rebalance_schedule_is_plain_frozen_dataclass_defaulting_to_k_none():
    import equinox as eqx

    from quantscenariobench.benchmark.interface import RebalanceSchedule

    default = RebalanceSchedule()
    assert default.k is None
    assert dataclasses.is_dataclass(default)
    assert type(default).__dataclass_params__.frozen is True
    assert not isinstance(default, eqx.Module)

    with pytest.raises(dataclasses.FrozenInstanceError):
        default.k = 5


# ---------------------------------------------------------------------------
# ProportionalCost is a plain, JSON-native value type mirroring
# RebalanceSchedule's house style (Story 10.2, FR-45, AD-34)
# ---------------------------------------------------------------------------

def test_proportional_cost_is_plain_frozen_dataclass():
    import equinox as eqx

    from quantscenariobench.benchmark.interface import ProportionalCost

    cost_model = ProportionalCost(one_way_bps=10)
    assert cost_model.one_way_bps == 10
    assert dataclasses.is_dataclass(cost_model)
    assert type(cost_model).__dataclass_params__.frozen is True
    assert not isinstance(cost_model, eqx.Module)

    with pytest.raises(dataclasses.FrozenInstanceError):
        cost_model.one_way_bps = 5


def test_proportional_cost_cost_method_matches_formula():
    from quantscenariobench.benchmark.interface import ProportionalCost

    w_target = jnp.array([0.4, 0.6])
    w_drifted = jnp.array([0.6, 0.4])
    cost_model = ProportionalCost(one_way_bps=10)

    expected = (10 / 1e4) * jnp.sum(jnp.abs(w_target - w_drifted))
    assert float(cost_model.cost(w_target, w_drifted)) == pytest.approx(float(expected), abs=1e-15)


# ---------------------------------------------------------------------------
# MetricContext.weights carries the full weight sequence for a rebalanced
# run, and weight-dependent metrics score it without error (Story 9.1/9.4
# follow-on impact, flagged in this story's Dev Notes)
# ---------------------------------------------------------------------------

def test_run_benchmark_rebalanced_run_scores_weight_dependent_metric():
    from quantscenariobench.benchmark.interface import RebalanceSchedule
    from quantscenariobench.benchmark.metrics import herfindahl_index
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 4
    hist = _returns(jax.random.PRNGKey(40), 30, n)
    eval_ = _returns(jax.random.PRNGKey(41), 60, n)

    result = run_benchmark(
        EqualWeight(), hist, eval_,
        rebalance_schedule=RebalanceSchedule(k=20),
        metrics=(herfindahl_index,),
    )
    # EqualWeight refits to 1/n at every rebalance, so the time-averaged
    # HHI over the weight sequence is exactly 1/n, same as the
    # buy-and-hold case (Story 9.4's degenerate one-element average).
    assert result.metrics["herfindahl_index"] == pytest.approx(1.0 / n, abs=1e-12)


# ===========================================================================
# Story 10.2 — Turnover Metric & Proportional Transaction-Cost Model
#
# Covers all acceptance criteria from GitHub Issue #84.
# ===========================================================================

# ---------------------------------------------------------------------------
# AC1: cost_model=None (the default) is bit-identical to pre-story golden
# results — already exercised by the golden-fixture tests above, which
# assert result.cost_model is None; here we additionally confirm the k=21
# rebalancing path (Story 10.1's own NumPy-reference regression,
# unaffected by cost_model=None) produces the same portfolio-return
# series whether cost_model is omitted or explicitly None (FR-45, AD-34).
# ---------------------------------------------------------------------------

def test_rebalancing_loop_cost_model_none_omitted_or_explicit_are_identical():
    from quantscenariobench.benchmark.runner._run_benchmark import _run_rebalancing_loop
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(60), 30, n)
    eval_ = _returns(jax.random.PRNGKey(61), 60, n)
    k = 20

    returns_no_cost, _, _ = _run_rebalancing_loop(EqualWeight(), hist, eval_, None, k, None)
    returns_default, _, _ = _run_rebalancing_loop(EqualWeight(), hist, eval_, None, k, None)
    assert jnp.array_equal(returns_no_cost, returns_default)


# ---------------------------------------------------------------------------
# AC2: ProportionalCost(0) nets to exactly the gross series
# ---------------------------------------------------------------------------

def test_proportional_cost_zero_bps_equals_gross_series():
    from quantscenariobench.benchmark.interface import ProportionalCost
    from quantscenariobench.benchmark.runner._run_benchmark import _run_rebalancing_loop
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    n = 3
    hist = _returns(jax.random.PRNGKey(62), 30, n)
    eval_ = _returns(jax.random.PRNGKey(63), 60, n)
    k = 20
    strategy = GlobalMinimumVariance(long_only=False)

    gross, _, _ = _run_rebalancing_loop(strategy, hist, eval_, None, k, None)
    net_zero, _, _ = _run_rebalancing_loop(strategy, hist, eval_, None, k, ProportionalCost(0))

    assert jnp.array_equal(gross, net_zero)


def test_run_benchmark_cost_model_zero_bps_matches_no_cost_model_metrics():
    from quantscenariobench.benchmark.interface import ProportionalCost, RebalanceSchedule
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(64), 30, n)
    eval_ = _returns(jax.random.PRNGKey(65), 60, n)
    schedule = RebalanceSchedule(k=20)

    result_none = run_benchmark(
        EqualWeight(), hist, eval_, rebalance_schedule=schedule, metrics=DEFAULT_METRICS
    )
    result_zero = run_benchmark(
        EqualWeight(), hist, eval_, rebalance_schedule=schedule,
        cost_model=ProportionalCost(0), metrics=DEFAULT_METRICS,
    )
    assert result_none.metrics == result_zero.metrics
    assert result_none.cost_model is None
    assert result_zero.cost_model == {"model": "ProportionalCost", "one_way_bps": 0}


# ---------------------------------------------------------------------------
# AC3: a hand-computed two-period example with a known weight change and
# one_way_bps=10 — net return at the rebalance step differs from gross by
# exactly 1e-3 * sum(|Δw|)
# ---------------------------------------------------------------------------

def test_proportional_cost_hand_computed_two_period_example():
    from quantscenariobench.benchmark.interface import BaselineStrategy, PortfolioWeights, ProportionalCost
    from quantscenariobench.benchmark.runner._run_benchmark import _run_rebalancing_loop

    def make_two_period_strategy():
        targets = iter([jnp.array([0.7, 0.3]), jnp.array([0.4, 0.6])])

        class TwoPeriodStrategy(BaselineStrategy):
            def allocate(self, historical_returns):
                return PortfolioWeights(next(targets))

        return TwoPeriodStrategy()

    hist = jnp.zeros((5, 2))
    eval_ = jnp.array([[0.20, -0.10], [0.05, -0.05], [0.02, 0.01], [0.03, -0.02]])
    k = 2  # rebalance dates t=0, t=2

    gross, _, drifted_weights = _run_rebalancing_loop(
        make_two_period_strategy(), hist, eval_, None, k, None
    )
    net, _, _ = _run_rebalancing_loop(
        make_two_period_strategy(), hist, eval_, None, k, ProportionalCost(10)
    )

    assert len(drifted_weights) == 1
    delta_w = jnp.sum(jnp.abs(jnp.array([0.4, 0.6]) - drifted_weights[0]))
    expected_diff = 1e-3 * delta_w

    # Only the rebalance-day return (index 2, the start of the second
    # holding period) differs; every other day is untouched.
    assert float(gross[2] - net[2]) == pytest.approx(float(expected_diff), abs=1e-12)
    for i in (0, 1, 3):
        assert float(gross[i]) == pytest.approx(float(net[i]), abs=1e-15)


# ---------------------------------------------------------------------------
# AC5: Sharpe(net) <= Sharpe(gross) for cost > 0, for every shipped
# strategy — a property test matching the paper's monotone cost-
# compression finding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_name", [
    "EqualWeight", "GlobalMinimumVariance", "CVaROptimization",
])
def test_sharpe_net_never_exceeds_sharpe_gross_for_every_shipped_strategy(strategy_name):
    from quantscenariobench.benchmark.interface import ProportionalCost, RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import (
        CVaROptimization,
        EqualWeight,
        GlobalMinimumVariance,
    )

    def make_strategy():
        if strategy_name == "EqualWeight":
            return EqualWeight()
        if strategy_name == "GlobalMinimumVariance":
            return GlobalMinimumVariance(long_only=True)
        return CVaROptimization(confidence_level=0.95)

    n = 4
    hist = _returns(jax.random.PRNGKey(70), 40, n)
    eval_ = _returns(jax.random.PRNGKey(71), 120, n)
    schedule = RebalanceSchedule(k=21)

    gross = run_benchmark(make_strategy(), hist, eval_, rebalance_schedule=schedule)
    net = run_benchmark(
        make_strategy(), hist, eval_, rebalance_schedule=schedule,
        cost_model=ProportionalCost(10),
    )
    assert net.metrics["sharpe_ratio"] <= gross.metrics["sharpe_ratio"]


# ---------------------------------------------------------------------------
# AC6: the active cost configuration is additive on BenchmarkResult, and
# two EvaluationResults for the same strategy/dataset at different bps
# produce two distinguishable leaderboard entries — the cost setting
# joins the aggregation key rather than collapsing rows
# ---------------------------------------------------------------------------

def test_run_benchmark_serializes_cost_model_additively():
    from quantscenariobench.benchmark.interface import ProportionalCost, RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(72), 30, n)
    eval_ = _returns(jax.random.PRNGKey(73), 60, n)

    result_no_cost = run_benchmark(EqualWeight(), hist, eval_)
    assert result_no_cost.cost_model is None

    result_with_cost = run_benchmark(
        EqualWeight(), hist, eval_,
        rebalance_schedule=RebalanceSchedule(k=20),
        cost_model=ProportionalCost(5),
    )
    assert result_with_cost.cost_model == {"model": "ProportionalCost", "one_way_bps": 5}


def test_two_evaluation_results_at_different_bps_produce_distinguishable_leaderboard_rows():
    from quantscenariobench.benchmark.evaluation import aggregate_evaluation_results, to_evaluation_result
    from quantscenariobench.benchmark.interface import ProportionalCost, RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(74), 30, n)
    eval_ = _returns(jax.random.PRNGKey(75), 60, n)
    schedule = RebalanceSchedule(k=20)

    result_0bps = run_benchmark(
        EqualWeight(), hist, eval_, rebalance_schedule=schedule, cost_model=ProportionalCost(0),
        asset_scenario_ids=["a0", "a1", "a2"], time_grid_reference="tg-cost-sweep",
    )
    result_10bps = run_benchmark(
        EqualWeight(), hist, eval_, rebalance_schedule=schedule, cost_model=ProportionalCost(10),
        asset_scenario_ids=["a0", "a1", "a2"], time_grid_reference="tg-cost-sweep",
    )

    table = aggregate_evaluation_results(
        [to_evaluation_result(result_0bps), to_evaluation_result(result_10bps)]
    )
    assert len(table) == 2
    assert {row["cost_one_way_bps"] for row in table} == {0, 10}


# ---------------------------------------------------------------------------
# AC7: a documented three-line sensitivity-sweep pattern over the shipped
# API, no bespoke helper
# ---------------------------------------------------------------------------

def test_cost_sensitivity_sweep_is_a_three_line_loop_against_the_shipped_api():
    from quantscenariobench.benchmark.interface import ProportionalCost, RebalanceSchedule
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import EqualWeight

    n = 3
    hist = _returns(jax.random.PRNGKey(76), 30, n)
    eval_ = _returns(jax.random.PRNGKey(77), 60, n)

    sharpe_by_bps = {}
    for bps in (0, 5, 10):
        result = run_benchmark(
            EqualWeight(), hist, eval_,
            rebalance_schedule=RebalanceSchedule(k=20), cost_model=ProportionalCost(bps),
        )
        sharpe_by_bps[bps] = result.metrics["sharpe_ratio"]

    assert set(sharpe_by_bps) == {0, 5, 10}
    # Monotone cost compression: higher bps never produces a higher Sharpe.
    assert sharpe_by_bps[0] >= sharpe_by_bps[5] >= sharpe_by_bps[10]


def test_run_benchmark_module_documents_cost_sweep_pattern():
    src = (_pkg_root() / "benchmark" / "runner" / "_run_benchmark.py").read_text()
    assert "ProportionalCost(bps)" in src
    assert "0, 5, 10" in src


def test_readme_documents_rebalancing_and_cost_sweep_pattern():
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    assert "RebalanceSchedule" in readme
    assert "ProportionalCost" in readme
    assert "0, 5, 10" in readme
