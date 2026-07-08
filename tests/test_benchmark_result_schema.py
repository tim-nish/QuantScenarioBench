"""
Story 6.3 — JSON-Serializable BenchmarkResult

Covers all acceptance criteria from GitHub Issue #32. Story 4.1 already
defines the BenchmarkResult type and Story 6.2 already builds
run_benchmark(); this story verifies the *actual output* of a completed
run_benchmark() call against AD-24's fixed minimum field set end to end,
mirroring how FR-15's dataset-card gate is verified by test assertion
rather than a separate runtime validator.
"""

from __future__ import annotations

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test

REQUIRED_BENCHMARK_RESULT_FIELDS = frozenset({
    "strategy_name",
    "strategy_parameters",
    "metrics",
    "asset_scenario_ids",
    "time_grid_reference",
    "library_version",
    "generated_at",
    "rebalance_schedule",
    "cost_model",
    "metrics_distribution",
})


def _returns(key, t, n):
    return jax.random.normal(key, (t, n)) * 0.01


_HIST = _returns(jax.random.PRNGKey(0), 20, 3)
_EVAL = _returns(jax.random.PRNGKey(1), 10, 3)


def _run(strategy, **kwargs):
    from quantscenariobench.benchmark.runner import run_benchmark
    return run_benchmark(strategy, _HIST, _EVAL, **kwargs)


# ---------------------------------------------------------------------------
# AC: a completed run_benchmark() call's BenchmarkResult carries at minimum
# all seven required fields (FR-29, AD-24)
# ---------------------------------------------------------------------------

def test_run_benchmark_result_has_all_required_fields():
    from quantscenariobench.benchmark.strategies import EqualWeight

    result = _run(EqualWeight())
    for name in REQUIRED_BENCHMARK_RESULT_FIELDS:
        assert hasattr(result, name), f"BenchmarkResult missing required field {name!r}"


def test_run_benchmark_result_metrics_is_flat_dict_keyed_by_metric_name():
    from quantscenariobench.benchmark.metrics import DEFAULT_METRICS
    from quantscenariobench.benchmark.strategies import EqualWeight

    result = _run(EqualWeight())
    assert isinstance(result.metrics, dict)
    assert set(result.metrics) == {m.name for m in DEFAULT_METRICS}
    for value in result.metrics.values():
        assert isinstance(value, float)
    # flat, never nested
    assert not any(isinstance(v, dict) for v in result.metrics.values())


# ---------------------------------------------------------------------------
# AC: a BenchmarkResult missing any required field fails review — mirrors
# FR-15's dataset-card gate (FR-29). BenchmarkResult's own dataclass field
# set is the enforcement mechanism: omitting a field at construction raises,
# and the field set never drifts from AD-24's fixed list.
# ---------------------------------------------------------------------------

def test_benchmark_result_dataclass_has_exactly_the_required_fields():
    from quantscenariobench.benchmark.interface import BenchmarkResult

    actual = {f.name for f in dataclasses.fields(BenchmarkResult)}
    assert actual == REQUIRED_BENCHMARK_RESULT_FIELDS, (
        f"BenchmarkResult field mismatch.\n"
        f"  Missing : {sorted(REQUIRED_BENCHMARK_RESULT_FIELDS - actual)}\n"
        f"  Extra   : {sorted(actual - REQUIRED_BENCHMARK_RESULT_FIELDS)}"
    )


def test_constructing_benchmark_result_without_a_required_field_raises():
    from quantscenariobench.benchmark.interface import BenchmarkResult

    with pytest.raises(TypeError):
        BenchmarkResult(
            strategy_name="EqualWeight",
            strategy_parameters={},
            metrics={},
            asset_scenario_ids=[],
            time_grid_reference="tg-0",
            library_version="1.0.0",
            # generated_at omitted
        )


# ---------------------------------------------------------------------------
# AC: a BenchmarkResult round-trips through json.dumps/json.loads without
# loss (FR-29, NFR-6)
# ---------------------------------------------------------------------------

def test_run_benchmark_result_round_trips_through_json():
    from quantscenariobench.benchmark.interface import BenchmarkResult
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    result = _run(GlobalMinimumVariance(long_only=True))

    payload = json.dumps(dataclasses.asdict(result))
    restored = BenchmarkResult(**json.loads(payload))

    assert restored == result


# ---------------------------------------------------------------------------
# AC: a BenchmarkResult is a plain immutable Python dataclass, not an
# equinox.Module, with only JSON-native field types (AD-17)
# ---------------------------------------------------------------------------

def test_run_benchmark_result_is_plain_frozen_dataclass_not_equinox_module():
    from quantscenariobench.benchmark.interface import BenchmarkResult
    from quantscenariobench.benchmark.strategies import EqualWeight

    result = _run(EqualWeight())

    assert dataclasses.is_dataclass(result)
    assert type(result).__dataclass_params__.frozen is True
    assert not isinstance(result, eqx.Module)

    # None is included for rebalance_schedule (FR-44, AD-33): an additive,
    # opt-in field whose JSON-native value is either a dict or null.
    allowed_json_native = (str, float, int, dict, list, type(None))
    for field in dataclasses.fields(BenchmarkResult):
        value = getattr(result, field.name)
        assert isinstance(value, allowed_json_native), (
            f"BenchmarkResult.{field.name} has non-JSON-native value {value!r}"
        )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.strategy_name = "SomethingElse"


# ---------------------------------------------------------------------------
# AC: a CVaROptimization run with confidence_level=0.95 has
# strategy_parameters including confidence_level: 0.95 (FR-22, AD-15)
# ---------------------------------------------------------------------------

def test_cvar_optimization_strategy_parameters_includes_confidence_level():
    from quantscenariobench.benchmark.strategies import CVaROptimization

    result = _run(CVaROptimization(confidence_level=0.95))
    assert result.strategy_parameters["confidence_level"] == 0.95
    assert result.strategy_name == "CVaROptimization"


# ---------------------------------------------------------------------------
# AC: a multi-asset run_benchmark() call's asset_scenario_ids identifies
# each constituent Scenario/dataset used, and time_grid_reference
# identifies the shared TimeGrid (AD-22, AD-24)
# ---------------------------------------------------------------------------

def test_multi_asset_run_benchmark_records_asset_scenario_ids_and_time_grid_reference():
    from quantscenariobench.benchmark.strategies import EqualWeight

    n_assets = _HIST.shape[1]
    scenario_ids = [f"scenario-asset-{i}" for i in range(n_assets)]
    time_grid_ref = "tg-daily-2026-07-02"

    result = _run(
        EqualWeight(),
        asset_scenario_ids=scenario_ids,
        time_grid_reference=time_grid_ref,
    )

    assert result.asset_scenario_ids == scenario_ids
    assert len(result.asset_scenario_ids) == n_assets
    assert result.time_grid_reference == time_grid_ref


# ---------------------------------------------------------------------------
# Story 9.4 (Issue #82) AC7: herfindahl_index/weight_entropy/
# effective_number_of_assets values appear in BenchmarkResult.metrics and
# survive the EvaluationResult JSON round-trip (FR-43)
# ---------------------------------------------------------------------------

def test_concentration_metrics_survive_evaluation_result_json_round_trip():
    from quantscenariobench.benchmark.evaluation import EvaluationResult, to_evaluation_result
    from quantscenariobench.benchmark.metrics import (
        effective_number_of_assets,
        herfindahl_index,
        weight_entropy,
    )
    from quantscenariobench.benchmark.strategies import EqualWeight

    result = _run(
        EqualWeight(),
        metrics=(herfindahl_index, weight_entropy, effective_number_of_assets),
    )
    assert set(result.metrics) == {
        "herfindahl_index", "weight_entropy", "effective_number_of_assets",
    }

    evaluation_result = to_evaluation_result(result)
    payload = json.dumps(dataclasses.asdict(evaluation_result))
    restored = EvaluationResult.from_dict(json.loads(payload))

    restored_metrics = {m.name: m.value for m in restored.metrics}
    assert restored_metrics == result.metrics
