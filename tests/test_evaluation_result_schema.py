"""
Story 7.1 — EvaluationResult Schema

Covers all acceptance criteria from GitHub Issue #46: a fixed, JSON-native
EvaluationResult schema derived from BenchmarkResult (FR-30, AD-26).
"""

from __future__ import annotations

import dataclasses
import json

import equinox as eqx
import pytest

REQUIRED_EVALUATION_RESULT_FIELDS = frozenset({
    "schema_version",
    "result_id",
    "strategy",
    "benchmark_dataset",
    "metrics",
    "library_version",
    "generated_at",
    "rebalance_schedule",
    "cost_model",
})


def _make_result():
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationMetric,
        EvaluationResult,
        EvaluationStrategy,
    )

    return EvaluationResult(
        schema_version="1.0",
        result_id="result-0001",
        strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
        benchmark_dataset=EvaluationBenchmarkDataset(
            asset_scenario_ids=["scenario-asset-0", "scenario-asset-1"],
            time_grid_reference="tg-daily-2026-07-02",
        ),
        metrics=[
            EvaluationMetric(name="sharpe_ratio", value=1.23),
            EvaluationMetric(name="max_drawdown", value=-0.12),
        ],
        library_version="1.0.0",
        generated_at="2026-07-03T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# AC: the EvaluationResult type carries at minimum schema_version, result_id,
# strategy (name, parameters), benchmark_dataset (asset_scenario_ids,
# time_grid_reference), metrics, library_version, generated_at (FR-30, AD-26)
# ---------------------------------------------------------------------------

def test_evaluation_result_has_all_required_fields():
    result = _make_result()
    for name in REQUIRED_EVALUATION_RESULT_FIELDS:
        assert hasattr(result, name), f"EvaluationResult missing required field {name!r}"
    assert result.strategy.name == "EqualWeight"
    assert result.strategy.parameters == {}
    assert result.benchmark_dataset.asset_scenario_ids == [
        "scenario-asset-0",
        "scenario-asset-1",
    ]
    assert result.benchmark_dataset.time_grid_reference == "tg-daily-2026-07-02"


def test_evaluation_result_dataclass_has_exactly_the_required_fields():
    from quantscenariobench.benchmark.evaluation import EvaluationResult

    actual = {f.name for f in dataclasses.fields(EvaluationResult)}
    assert actual == REQUIRED_EVALUATION_RESULT_FIELDS, (
        f"EvaluationResult field mismatch.\n"
        f"  Missing : {sorted(REQUIRED_EVALUATION_RESULT_FIELDS - actual)}\n"
        f"  Extra   : {sorted(actual - REQUIRED_EVALUATION_RESULT_FIELDS)}"
    )


# ---------------------------------------------------------------------------
# AC: EvaluationResult.metrics is an ordered list of {name, value} records —
# not BenchmarkResult's flat dict[str, float] (FR-30, AD-26)
# ---------------------------------------------------------------------------

def test_evaluation_result_metrics_is_ordered_list_of_name_value_records():
    result = _make_result()

    assert isinstance(result.metrics, list)
    assert [m.name for m in result.metrics] == ["sharpe_ratio", "max_drawdown"]
    for metric in result.metrics:
        assert isinstance(metric.value, float)
    assert not isinstance(result.metrics, dict)


# ---------------------------------------------------------------------------
# AC: an EvaluationResult missing any one required field fails review,
# mirroring FR-29's and FR-15's review gates (FR-30)
# ---------------------------------------------------------------------------

def test_constructing_evaluation_result_without_a_required_field_raises():
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationResult,
        EvaluationStrategy,
    )

    with pytest.raises(TypeError):
        EvaluationResult(
            schema_version="1.0",
            result_id="result-0001",
            strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=[], time_grid_reference="tg-0"
            ),
            metrics=[],
            library_version="1.0.0",
            # generated_at omitted
        )


# ---------------------------------------------------------------------------
# AC: an EvaluationResult round-trips through json.dumps/json.loads (or
# equivalent) without loss (FR-30, NFR-7)
# ---------------------------------------------------------------------------

def test_evaluation_result_round_trips_through_json():
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationMetric,
        EvaluationResult,
        EvaluationStrategy,
    )

    result = _make_result()

    payload = json.dumps(dataclasses.asdict(result))
    data = json.loads(payload)
    restored = EvaluationResult(
        schema_version=data["schema_version"],
        result_id=data["result_id"],
        strategy=EvaluationStrategy(**data["strategy"]),
        benchmark_dataset=EvaluationBenchmarkDataset(**data["benchmark_dataset"]),
        metrics=[EvaluationMetric(**m) for m in data["metrics"]],
        library_version=data["library_version"],
        generated_at=data["generated_at"],
    )

    assert restored == result


# ---------------------------------------------------------------------------
# AC: EvaluationResult is a plain immutable Python dataclass, not an
# equinox.Module, with only JSON-native field types (AD-26)
# ---------------------------------------------------------------------------

def test_evaluation_result_is_plain_frozen_dataclass_not_equinox_module():
    from quantscenariobench.benchmark.evaluation import EvaluationResult

    result = _make_result()

    assert dataclasses.is_dataclass(result)
    assert type(result).__dataclass_params__.frozen is True
    assert not isinstance(result, eqx.Module)

    # None is included for rebalance_schedule (FR-44, AD-33): an additive,
    # opt-in field whose JSON-native value is either a dict or null.
    allowed_json_native = (str, float, int, dict, list, type(None))
    for field in dataclasses.fields(EvaluationResult):
        value = getattr(result, field.name)
        if dataclasses.is_dataclass(value):
            continue
        assert isinstance(value, allowed_json_native), (
            f"EvaluationResult.{field.name} has non-JSON-native value {value!r}"
        )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.schema_version = "2.0"


def test_evaluation_result_nested_records_are_plain_frozen_dataclasses():
    result = _make_result()

    for nested in (result.strategy, result.benchmark_dataset, *result.metrics):
        assert dataclasses.is_dataclass(nested)
        assert type(nested).__dataclass_params__.frozen is True
        assert not isinstance(nested, eqx.Module)


# ---------------------------------------------------------------------------
# AC: BenchmarkResult's own schema (FR-29, AD-24) is unchanged by adding
# EvaluationResult — EvaluationResult is additive, never a replacement
# (AD-26)
# ---------------------------------------------------------------------------

def test_benchmark_result_schema_unchanged_by_evaluation_result():
    from quantscenariobench.benchmark.interface import BenchmarkResult

    actual = {f.name for f in dataclasses.fields(BenchmarkResult)}
    assert actual == {
        "strategy_name",
        "strategy_parameters",
        "metrics",
        "asset_scenario_ids",
        "time_grid_reference",
        "library_version",
        "generated_at",
        "rebalance_schedule",
        "cost_model",
    }
