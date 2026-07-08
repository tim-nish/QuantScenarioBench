from __future__ import annotations

import hashlib
import json

from ..interface import BenchmarkResult
from ._evaluation_result import (
    EvaluationBenchmarkDataset,
    EvaluationMetric,
    EvaluationResult,
    EvaluationStrategy,
)

_SCHEMA_VERSION = "1.0"


def _result_id(result: BenchmarkResult) -> str:
    """A deterministic id derived from result's own content (FR-31).

    Hashing the BenchmarkResult's fields — rather than a random uuid —
    is what gives to_evaluation_result its determinism guarantee: the
    same BenchmarkResult always yields the same result_id, mirroring
    FR-27's guarantee for BenchmarkResult itself.
    """
    canonical = json.dumps(
        {
            "strategy_name": result.strategy_name,
            "strategy_parameters": result.strategy_parameters,
            "metrics": result.metrics,
            "asset_scenario_ids": result.asset_scenario_ids,
            "time_grid_reference": result.time_grid_reference,
            "library_version": result.library_version,
            "generated_at": result.generated_at,
            "rebalance_schedule": result.rebalance_schedule,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def to_evaluation_result(result: BenchmarkResult) -> EvaluationResult:
    """Convert a BenchmarkResult into its published EvaluationResult (FR-31, AD-26).

    A single, pure function: it never mutates or subclasses result, and
    requires no changes to run_benchmark() or any Epic 6 module — it
    only reads fields already present on an already-produced
    BenchmarkResult.
    """
    return EvaluationResult(
        schema_version=_SCHEMA_VERSION,
        result_id=_result_id(result),
        strategy=EvaluationStrategy(
            name=result.strategy_name,
            parameters=result.strategy_parameters,
        ),
        benchmark_dataset=EvaluationBenchmarkDataset(
            asset_scenario_ids=result.asset_scenario_ids,
            time_grid_reference=result.time_grid_reference,
        ),
        metrics=[
            EvaluationMetric(name=name, value=value)
            for name, value in result.metrics.items()
        ],
        library_version=result.library_version,
        generated_at=result.generated_at,
        rebalance_schedule=result.rebalance_schedule,
    )
