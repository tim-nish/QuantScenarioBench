"""Leaderboard aggregation over published EvaluationResults (FR-34).

Generic purely because EvaluationResult's schema (AD-26) is fixed — no
strategy-specific or dataset-specific branching anywhere in this module,
mirroring AD-18's extensibility guarantee for Metrics one layer up.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Sequence

from ._evaluation_result import EvaluationResult
from ._hub_publish import _RESULTS_PREFIX


def load_evaluation_results(root: str | Path) -> list[EvaluationResult]:
    """Load every EvaluationResult from a local collection (Story 7.3's layout).

    root is the same directory write_evaluation_result() writes under —
    walked recursively for every *.json file, regardless of the
    Benchmark Dataset/strategy subdirectories it's organized into.
    """
    root = Path(root)
    return [
        EvaluationResult.from_dict(json.loads(path.read_text()))
        for path in sorted(root.rglob("*.json"))
    ]


def load_evaluation_results_from_hub(
    repo_id: str, *, token: str | None = None
) -> list[EvaluationResult]:
    """Load every EvaluationResult published to a shared Hugging Face dataset repo.

    Downloads the repo's results/ tree (Story 7.4's upload layout) to a
    local snapshot and reuses load_evaluation_results() to parse it — the
    same "equivalent local collection" this reader's contract allows.
    """
    try:
        import huggingface_hub as hf
    except ImportError as exc:
        raise ImportError(
            "load_evaluation_results_from_hub requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    snapshot_dir = hf.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{_RESULTS_PREFIX}/**"],
        token=token,
    )
    return load_evaluation_results(Path(snapshot_dir) / _RESULTS_PREFIX)


def _parse_generated_at(generated_at: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(generated_at)


def _cost_one_way_bps(result: EvaluationResult) -> float | None:
    """The active cost setting, or None when no cost model was used (FR-45,
    AD-34) — joins the aggregation key so two EvaluationResults for the
    same strategy/dataset at different bps produce distinguishable rows
    instead of collapsing into the same "latest wins" cell.
    """
    if result.cost_model is None:
        return None
    return result.cost_model.get("one_way_bps")


def aggregate_evaluation_results(results: Sequence[EvaluationResult]) -> list[dict]:
    """Build a ranked Leaderboard table from every published EvaluationResult (FR-34).

    Returns a plain list of dict rows — one row per strategy x Benchmark
    Dataset x cost-setting combination, one key per Metric name (plus
    "strategy", "benchmark_dataset", "cost_one_way_bps") — with no
    dependency on any UI framework. When more than one EvaluationResult
    exists for the same combination (an append-only history, Story
    7.3/7.4), the most recently generated one wins the row — unchanged
    from before the cost setting joined the key (FR-45, AD-34): this
    merge rule only ever collapses genuine repeats (same strategy,
    dataset, *and* cost setting), never results that differ by bps.
    """
    latest_by_key: dict[tuple[str, str, float | None], EvaluationResult] = {}
    for result in results:
        key = (
            result.strategy.name,
            result.benchmark_dataset.time_grid_reference,
            _cost_one_way_bps(result),
        )
        current = latest_by_key.get(key)
        if current is None or _parse_generated_at(result.generated_at) > _parse_generated_at(
            current.generated_at
        ):
            latest_by_key[key] = result

    rows = []
    for (strategy_name, dataset, cost_bps), result in sorted(
        latest_by_key.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2] is None, item[0][2] or 0.0),
    ):
        row: dict = {
            "strategy": strategy_name,
            "benchmark_dataset": dataset,
            "cost_one_way_bps": cost_bps,
        }
        for metric in result.metrics:
            row[metric.name] = metric.value
        rows.append(row)
    return rows
