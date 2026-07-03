from __future__ import annotations

import dataclasses


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
