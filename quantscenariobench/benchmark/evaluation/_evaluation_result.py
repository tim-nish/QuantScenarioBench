from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class EvaluationStrategy:
    """The `strategy` sub-record of an EvaluationResult (FR-30, AD-26)."""

    name: str
    parameters: dict


@dataclasses.dataclass(frozen=True)
class EvaluationBenchmarkDataset:
    """The `benchmark_dataset` sub-record of an EvaluationResult (FR-30, AD-26)."""

    asset_scenario_ids: list
    time_grid_reference: str


@dataclasses.dataclass(frozen=True)
class EvaluationMetric:
    """One `{name, value}` record in EvaluationResult.metrics (FR-30, AD-26).

    Deliberately a list of records rather than BenchmarkResult's flat
    dict[str, float] (AD-24) — matches the Hugging Face
    model-index.results[].metrics[] convention consumed by Hub rendering
    and Leaderboard aggregation (FR-34).
    """

    name: str
    value: float


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    """The publication-layer schema derived from BenchmarkResult (AD-26).

    A plain immutable dataclass — never an equinox.Module — with only
    JSON-native field types, the same posture AD-17 fixes for
    BenchmarkResult. EvaluationResult is additive: BenchmarkResult (AD-17,
    AD-24) remains the sole runtime representation produced by
    run_benchmark() and is unchanged by this type.
    """

    schema_version: str
    result_id: str
    strategy: EvaluationStrategy
    benchmark_dataset: EvaluationBenchmarkDataset
    metrics: list[EvaluationMetric]
    library_version: str
    generated_at: str
    rebalance_schedule: Optional[dict] = None

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """Reconstruct an EvaluationResult from its dataclasses.asdict() shape.

        The exact inverse of json.loads(json.dumps(dataclasses.asdict(result))),
        needed by any reader of a published EvaluationResult (e.g. Story 7.5's
        Leaderboard aggregation).

        rebalance_schedule uses .get(...), defaulting to None, so an
        EvaluationResult JSON file published before this field existed
        still loads (FR-44, AC7) — the schema addition is additive only.
        """
        return cls(
            schema_version=data["schema_version"],
            result_id=data["result_id"],
            strategy=EvaluationStrategy(**data["strategy"]),
            benchmark_dataset=EvaluationBenchmarkDataset(**data["benchmark_dataset"]),
            metrics=[EvaluationMetric(**m) for m in data["metrics"]],
            library_version=data["library_version"],
            generated_at=data["generated_at"],
            rebalance_schedule=data.get("rebalance_schedule"),
        )
