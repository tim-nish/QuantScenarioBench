"""
Story 4.3 — Portfolio Performance Metrics
(Sharpe, Sortino, Maximum Drawdown, Final Wealth Factor)

Covers all acceptance criteria from GitHub Issue #26.

Reference values below are hand-derived using plain Python (statistics/
math), never any of quantscenariobench's own jax.numpy implementation and
never a portfolio-analytics library (empyrical, quantstats, PyPortfolioOpt,
Riskfolio-Lib, or otherwise) — AD-10 amended.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test

_RETURNS = [0.10, -0.05, 0.02, 0.03, -0.01]
_TOL = 1e-9


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC: every MetricFn is written entirely in jax.numpy, jit-compatible, and
# never calls scipy or plain numpy (AD-18, AD-25)
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT = re.compile(r"(?:import|from)\s+(scipy|numpy)\b")


def test_metrics_module_never_imports_scipy_or_numpy():
    metrics_dir = _pkg_root() / "benchmark" / "metrics"
    violations = []
    for py_file in metrics_dir.rglob("*.py"):
        if _FORBIDDEN_IMPORT.search(py_file.read_text()):
            violations.append(str(py_file.relative_to(_pkg_root().parent)))
    assert not violations, (
        f"AD-18/AD-25 violation: benchmark.metrics must be jax.numpy-only: {violations}"
    )


@pytest.mark.parametrize("name", [
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
])
def test_metric_fn_is_jit_compatible(name):
    from quantscenariobench.benchmark import metrics as m
    fn = getattr(m, name)
    returns = jnp.array(_RETURNS)
    eager = fn(returns)
    jitted = jax.jit(fn)(returns)
    assert jnp.allclose(eager, jitted)


# ---------------------------------------------------------------------------
# AC: Sharpe Ratio matches a hand-derived reference (FR-16)
# ---------------------------------------------------------------------------

def test_sharpe_ratio_matches_hand_derived_reference():
    from quantscenariobench.benchmark.metrics import sharpe_ratio

    mean = statistics.fmean(_RETURNS)
    std = statistics.pstdev(_RETURNS)  # population stdev, ddof=0
    expected = mean / std

    actual = float(sharpe_ratio(jnp.array(_RETURNS)))
    assert actual == pytest.approx(expected, abs=_TOL)


def test_sharpe_ratio_returns_zero_for_constant_returns():
    from quantscenariobench.benchmark.metrics import sharpe_ratio

    constant = jnp.array([0.02, 0.02, 0.02, 0.02])
    result = sharpe_ratio(constant)
    assert float(result) == 0.0
    assert not bool(jnp.isnan(result))
    assert not bool(jnp.isinf(result))


# ---------------------------------------------------------------------------
# AC: Sortino Ratio matches a hand-derived reference (FR-17)
# ---------------------------------------------------------------------------

def test_sortino_ratio_matches_hand_derived_reference():
    from quantscenariobench.benchmark.metrics import sortino_ratio

    mean = statistics.fmean(_RETURNS)
    downside = [min(r, 0.0) for r in _RETURNS]
    downside_deviation = math.sqrt(statistics.fmean(d ** 2 for d in downside))
    expected = mean / downside_deviation

    actual = float(sortino_ratio(jnp.array(_RETURNS)))
    assert actual == pytest.approx(expected, abs=_TOL)


def test_sortino_ratio_returns_zero_when_no_negative_returns():
    from quantscenariobench.benchmark.metrics import sortino_ratio

    all_non_negative = jnp.array([0.01, 0.02, 0.0, 0.03])
    result = sortino_ratio(all_non_negative)
    assert float(result) == 0.0
    assert not bool(jnp.isnan(result))


# ---------------------------------------------------------------------------
# AC: Maximum Drawdown matches a hand-derived reference (FR-18)
# ---------------------------------------------------------------------------

def test_max_drawdown_matches_hand_derived_reference():
    from quantscenariobench.benchmark.metrics import max_drawdown

    wealth = []
    running = 1.0
    for r in _RETURNS:
        running *= (1.0 + r)
        wealth.append(running)

    peak = []
    running_peak = -math.inf
    for w in wealth:
        running_peak = max(running_peak, w)
        peak.append(running_peak)

    drawdowns = [(w - p) / p for w, p in zip(wealth, peak)]
    expected = min(drawdowns)

    actual = float(max_drawdown(jnp.array(_RETURNS)))
    assert actual == pytest.approx(expected, abs=_TOL)


# ---------------------------------------------------------------------------
# AC: Final Wealth Factor matches a hand-derived reference (FR-19)
# ---------------------------------------------------------------------------

def test_final_wealth_factor_matches_hand_derived_reference():
    from quantscenariobench.benchmark.metrics import final_wealth_factor

    expected = 1.0
    for r in _RETURNS:
        expected *= (1.0 + r)

    actual = float(final_wealth_factor(jnp.array(_RETURNS)))
    assert actual == pytest.approx(expected, abs=_TOL)


# ---------------------------------------------------------------------------
# AC: every MetricFn carries a .name: str attribute (AD-18)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
])
def test_metric_fn_has_name_attribute(name):
    from quantscenariobench.benchmark import metrics as m
    fn = getattr(m, name)
    assert isinstance(fn.name, str)
    assert fn.name == name


def test_metric_fn_accepts_return_series_and_returns_scalar():
    from quantscenariobench.benchmark.metrics import sharpe_ratio
    result = sharpe_ratio(jnp.array(_RETURNS))
    assert jnp.asarray(result).shape == ()


# ---------------------------------------------------------------------------
# AC: registry raises on duplicate .name rather than silently shadowing (AD-18)
# ---------------------------------------------------------------------------

def test_default_metrics_registry_has_no_duplicate_names():
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS, validate_metric_registry
    validate_metric_registry(DEFAULT_METRICS)  # must not raise
    names = [m.name for m in DEFAULT_METRICS]
    assert len(names) == len(set(names))


def test_validate_metric_registry_raises_on_duplicate_name():
    from quantscenariobench.benchmark.metrics import sharpe_ratio, validate_metric_registry

    def fake_metric(returns):
        return jnp.mean(returns)
    fake_metric.name = "sharpe_ratio"  # deliberately collides

    with pytest.raises(ValueError, match="sharpe_ratio"):
        validate_metric_registry((sharpe_ratio, fake_metric))


# ---------------------------------------------------------------------------
# AC: no Metrics/Baselines correctness test imports its reference value
# from a portfolio-analytics library (AD-10 amended)
# ---------------------------------------------------------------------------

_FORBIDDEN_REFERENCE_LIBS = re.compile(
    r"(?:import|from)\s+(empyrical|quantstats|pypfopt|PyPortfolioOpt|riskfolio|Riskfolio)\b",
    re.IGNORECASE,
)


def test_no_test_file_imports_a_portfolio_analytics_library():
    tests_dir = Path(__file__).parent
    violations = []
    for py_file in tests_dir.glob("test_*.py"):
        if _FORBIDDEN_REFERENCE_LIBS.search(py_file.read_text()):
            violations.append(py_file.name)
    assert not violations, (
        f"AD-10 (amended) violation: reference values must be hand-derived, "
        f"not imported from a portfolio-analytics library: {violations}"
    )


# ===========================================================================
# Story 9.1 — Context-Aware Metric Interface (MetricContext) & Metric
# Metadata
#
# Covers all acceptance criteria from GitHub Issue #79.
# ===========================================================================

def _make_context(returns, n_assets=1):
    from quantscenariobench.benchmark.interface import PortfolioWeights
    from quantscenariobench.benchmark.metrics import MetricContext

    return MetricContext(
        portfolio_returns=returns,
        weights=PortfolioWeights(jnp.full((n_assets,), 1.0 / n_assets)),
        evaluation_returns=jnp.ones((1, n_assets)),
    )


# ---------------------------------------------------------------------------
# AC1: MetricContext is a frozen value object carrying at minimum
# portfolio_returns, weights, evaluation_returns, and an extensible
# auxiliary field (FR-40, AD-31)
# ---------------------------------------------------------------------------

def test_metric_context_carries_required_fields():
    import dataclasses

    returns = jnp.array(_RETURNS)
    context = _make_context(returns)

    assert jnp.array_equal(context.portfolio_returns, returns)
    assert context.weights.weights.shape == (1,)
    assert context.evaluation_returns.shape == (1, 1)
    assert context.auxiliary == {}
    assert dataclasses.is_dataclass(context)
    assert type(context).__dataclass_params__.frozen is True


def test_metric_context_is_not_an_equinox_module():
    import equinox as eqx

    context = _make_context(jnp.array(_RETURNS))
    assert not isinstance(context, eqx.Module)


# ---------------------------------------------------------------------------
# AC2: the Metric protocol declares name, direction, params, and
# __call__(context) -> scalar (FR-40, AD-31)
# ---------------------------------------------------------------------------

def test_metric_protocol_is_runtime_checkable_and_matches_wrapped_metric():
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS, Metric

    for metric in DEFAULT_METRICS:
        assert isinstance(metric, Metric)


# ---------------------------------------------------------------------------
# AC3: a trivial weight-dependent metric implementing the new protocol
# scores correctly given a MetricContext (FR-40)
# ---------------------------------------------------------------------------

def test_weight_dependent_metric_reads_context_weights():
    class SumOfSquaredWeights:
        name = "sum_of_squared_weights"
        direction = "lower_is_better"
        params = None

        def __call__(self, context):
            return jnp.sum(context.weights.weights ** 2)

    context = _make_context(jnp.array(_RETURNS), n_assets=1)
    metric = SumOfSquaredWeights()
    assert float(metric(context)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC4: the four pre-existing metrics migrated via wrap_legacy_metric are
# bit-identical to calling the bare MetricFn directly, and DEFAULT_METRICS
# names are unchanged (FR-40, AD-31)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
])
def test_default_metrics_are_bit_identical_to_raw_metric_fn(name):
    from quantscenariobench.benchmark import metrics as m
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS

    raw_fn = getattr(m, name)
    wrapped = next(metric for metric in DEFAULT_METRICS if metric.name == name)

    returns = jnp.array(_RETURNS)
    context = _make_context(returns)
    assert jnp.array_equal(raw_fn(returns), wrapped(context))


def test_default_metrics_names_unchanged():
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS

    names = {metric.name for metric in DEFAULT_METRICS}
    assert names == {
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
    }


# ---------------------------------------------------------------------------
# AC5: wrap_legacy_metric adapts a bare MetricFn into the Metric protocol,
# preserving .name and forwarding context.portfolio_returns unchanged
# ---------------------------------------------------------------------------

def test_wrap_legacy_metric_preserves_name_and_forwards_portfolio_returns():
    from quantscenariobench.benchmark.metrics import wrap_legacy_metric

    def custom_metric(returns):
        return jnp.sum(returns)
    custom_metric.name = "total_return"

    wrapped = wrap_legacy_metric(custom_metric, direction="lower_is_better")
    assert wrapped.name == "total_return"
    assert wrapped.direction == "lower_is_better"
    assert wrapped.params is None

    returns = jnp.array(_RETURNS)
    context = _make_context(returns)
    assert wrapped(context) == custom_metric(returns)


# ---------------------------------------------------------------------------
# AC6: two Metric instances of a parametrized family with different
# params/names (e.g. cvar_0.95, cvar_0.99) coexist in one registry
# without tripping duplicate-name validation
# ---------------------------------------------------------------------------

def test_validate_metric_registry_allows_parametrized_metrics_with_distinct_names():
    from quantscenariobench.benchmark.metrics import validate_metric_registry

    class _ParamMetric:
        def __init__(self, alpha):
            self.name = f"cvar_{alpha}"
            self.direction = "lower_is_better"
            self.params = {"alpha": alpha}

        def __call__(self, context):
            return jnp.min(context.portfolio_returns)

    validate_metric_registry((_ParamMetric(0.95), _ParamMetric(0.99)))  # must not raise


# ---------------------------------------------------------------------------
# AC7: BenchmarkResult/EvaluationResult are unchanged by this story — a
# JSON file written by the previous library version (no direction/params
# fields) still loads, since the schema addition is deferred (FR-40, NFR-6)
# ---------------------------------------------------------------------------

def test_benchmark_result_schema_has_no_new_direction_or_params_fields():
    import dataclasses
    from quantscenariobench.benchmark.interface import BenchmarkResult

    field_names = {f.name for f in dataclasses.fields(BenchmarkResult)}
    assert "direction" not in field_names
    assert "params" not in field_names


def test_evaluation_metric_schema_has_no_new_direction_or_params_fields():
    import dataclasses
    from quantscenariobench.benchmark.evaluation import EvaluationMetric

    field_names = {f.name for f in dataclasses.fields(EvaluationMetric)}
    assert field_names == {"name", "value"}


def test_old_style_benchmark_result_json_still_loads():
    from quantscenariobench.benchmark.interface import BenchmarkResult

    old_style_payload = {
        "strategy_name": "EqualWeight",
        "strategy_parameters": {},
        "metrics": {"sharpe_ratio": 1.0},
        "asset_scenario_ids": [],
        "time_grid_reference": "tg-0",
        "library_version": "1.0.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    result = BenchmarkResult(**old_style_payload)
    assert result.metrics == {"sharpe_ratio": 1.0}


# ===========================================================================
# Story 9.2 — Tail-Risk Metrics: Value-at-Risk and Conditional Value-at-Risk
#
# Covers all acceptance criteria from GitHub Issue #80. The NumPy reference
# values below are computed in this test file only, never inside the
# library — quantscenariobench.benchmark.metrics itself stays jax.numpy-only
# (AD-18/AD-25, enforced by test_metrics_module_never_imports_scipy_or_numpy
# above, which already covers the new _tail_risk.py file).
# ===========================================================================

_TAIL_RISK_TOL = 1e-12

# A hand-computable 20-point return series (deterministic, fixed seed).
_TAIL_RETURNS_20 = [
    0.012, -0.034, 0.021, -0.008, 0.045, -0.062, 0.003, 0.017, -0.021, 0.009,
    -0.051, 0.028, 0.014, -0.009, 0.036, -0.073, 0.005, -0.018, 0.022, -0.004,
]


def _numpy_var_cvar_reference(returns, alpha):
    import numpy as np

    losses = -np.asarray(returns)
    nu = np.quantile(losses, alpha)
    tail_mean = np.mean(np.maximum(losses - nu, 0.0)) / (1.0 - alpha)
    return float(nu), float(nu + tail_mean)


def _tail_risk_context(returns):
    from quantscenariobench.benchmark.interface import PortfolioWeights
    from quantscenariobench.benchmark.metrics import MetricContext

    return MetricContext(
        portfolio_returns=jnp.asarray(returns),
        weights=PortfolioWeights(jnp.array([1.0])),
        evaluation_returns=jnp.ones((1, 1)),
    )


# ---------------------------------------------------------------------------
# AC1/AC2: value_at_risk(alpha)/conditional_value_at_risk(alpha) match a
# NumPy reference to 1e-12 on a hand-computable 20-point series (FR-41)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.90, 0.95])
def test_var_and_cvar_match_numpy_reference(alpha):
    # alpha is chosen so (1 - alpha) * 20 >= 1: the undersized-tail
    # fallback (AC4, tested separately below) does not trigger here, so
    # this test isolates the quantile/tail-mean arithmetic itself.
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )

    context = _tail_risk_context(_TAIL_RETURNS_20)
    expected_var, expected_cvar = _numpy_var_cvar_reference(_TAIL_RETURNS_20, alpha)

    actual_var = float(value_at_risk(alpha)(context))
    actual_cvar = float(conditional_value_at_risk(alpha)(context))

    assert actual_var == pytest.approx(expected_var, abs=_TAIL_RISK_TOL)
    assert actual_cvar == pytest.approx(expected_cvar, abs=_TAIL_RISK_TOL)


# ---------------------------------------------------------------------------
# AC3: cvar_0.95 >= var_0.95 holds across several return series (property)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cvar_is_never_less_than_var(seed):
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )

    key = jax.random.PRNGKey(seed)
    returns = jax.random.normal(key, (50,)) * 0.02
    context = _tail_risk_context(returns)

    var = float(value_at_risk(0.95)(context))
    cvar = float(conditional_value_at_risk(0.95)(context))
    assert cvar >= var - _TOL


def test_cvar_is_never_less_than_var_on_hand_computable_series():
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )

    context = _tail_risk_context(_TAIL_RETURNS_20)
    assert float(conditional_value_at_risk(0.95)(context)) >= float(
        value_at_risk(0.95)(context)
    )


# ---------------------------------------------------------------------------
# AC4: degenerate inputs return finite, documented values — never NaN/inf
# (all-zero returns, a single-step series, an undersized tail sample count)
# ---------------------------------------------------------------------------

def test_var_and_cvar_finite_for_all_zero_returns():
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )

    context = _tail_risk_context(jnp.zeros(20))
    for metric in (value_at_risk(0.95), conditional_value_at_risk(0.95)):
        value = metric(context)
        assert not bool(jnp.isnan(value))
        assert not bool(jnp.isinf(value))
        assert float(value) == pytest.approx(0.0)


def test_var_and_cvar_finite_for_single_step_series():
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )

    context = _tail_risk_context(jnp.array([0.05]))
    for metric in (value_at_risk(0.95), conditional_value_at_risk(0.95)):
        value = metric(context)
        assert not bool(jnp.isnan(value))
        assert not bool(jnp.isinf(value))


def test_var_and_cvar_fall_back_to_max_loss_when_tail_sample_count_undersized():
    from quantscenariobench.benchmark.metrics import (
        conditional_value_at_risk,
        value_at_risk,
    )

    # t=10, alpha=0.995 => (1 - alpha) * t = 0.05 < 1: undersized tail.
    returns = jnp.array(_TAIL_RETURNS_20[:10])
    context = _tail_risk_context(returns)
    expected_max_loss = float(jnp.max(-returns))

    var = float(value_at_risk(0.995)(context))
    cvar = float(conditional_value_at_risk(0.995)(context))
    assert var == pytest.approx(expected_max_loss, abs=_TAIL_RISK_TOL)
    assert cvar == pytest.approx(expected_max_loss, abs=_TAIL_RISK_TOL)
    assert not bool(jnp.isnan(var)) and not bool(jnp.isinf(var))
    assert not bool(jnp.isnan(cvar)) and not bool(jnp.isinf(cvar))


# ---------------------------------------------------------------------------
# AC7: docstrings state the sign convention (losses positive) and the
# quantile interpolation rule explicitly
# ---------------------------------------------------------------------------

def test_tail_risk_module_docstring_states_sign_convention_and_interpolation_rule():
    src = (_pkg_root() / "benchmark" / "metrics" / "_tail_risk.py").read_text()
    assert "positive" in src
    assert "linear" in src and "interpolat" in src


# ---------------------------------------------------------------------------
# AC8: DEFAULT_METRICS is unchanged — VaR/CVaR are opt-in only (AD-32)
# ---------------------------------------------------------------------------

def test_default_metrics_unchanged_by_tail_risk_metrics():
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS

    names = {metric.name for metric in DEFAULT_METRICS}
    assert names == {
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
    }
    assert not any(name.startswith("var_") or name.startswith("cvar_") for name in names)


# ---------------------------------------------------------------------------
# value_at_risk/conditional_value_at_risk are correctly named, parametrized
# Metric instances (name/direction/params shape, AD-31)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_value_at_risk_metric_shape(alpha):
    from quantscenariobench.benchmark.metrics import Metric, value_at_risk

    metric = value_at_risk(alpha)
    assert isinstance(metric, Metric)
    assert metric.name == f"var_{alpha}"
    assert metric.direction == "lower_is_better"
    assert metric.params == {"alpha": alpha}


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_conditional_value_at_risk_metric_shape(alpha):
    from quantscenariobench.benchmark.metrics import Metric, conditional_value_at_risk

    metric = conditional_value_at_risk(alpha)
    assert isinstance(metric, Metric)
    assert metric.name == f"cvar_{alpha}"
    assert metric.direction == "lower_is_better"
    assert metric.params == {"alpha": alpha}


# ---------------------------------------------------------------------------
# value_at_risk/conditional_value_at_risk stay jit-compatible (Review Focus)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory_name", ["value_at_risk", "conditional_value_at_risk"])
def test_tail_risk_metric_is_jit_compatible(factory_name):
    # MetricContext is deliberately not a pytree (AD-17 posture) and is
    # never itself passed across a jit boundary; PortfolioWeights' own
    # runtime validation also can't run under trace, so it is constructed
    # eagerly once and closed over — only the returns array is traced,
    # which is what "the metric body stays jax-native on the arrays it
    # carries" (Story 9.1 Dev Notes) actually requires.
    from quantscenariobench.benchmark import metrics as m
    from quantscenariobench.benchmark.interface import PortfolioWeights
    from quantscenariobench.benchmark.metrics import MetricContext

    metric = getattr(m, factory_name)(0.95)
    weights = PortfolioWeights(jnp.array([1.0]))

    def score(returns):
        context = MetricContext(
            portfolio_returns=returns,
            weights=weights,
            evaluation_returns=jnp.ones((1, 1)),
        )
        return metric(context)

    returns = jnp.asarray(_TAIL_RETURNS_20)
    eager = score(returns)
    jitted = jax.jit(score)(returns)
    assert jnp.allclose(eager, jitted)
