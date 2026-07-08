"""
Story 7.5 — Leaderboard Aggregation

Covers all acceptance criteria from GitHub Issue #50: a generic reader
that loads every published EvaluationResult and ranks them in one table
(FR-34).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _make_evaluation_result(**overrides):
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationMetric,
        EvaluationResult,
        EvaluationStrategy,
    )

    defaults = dict(
        schema_version="1.0",
        result_id="result-0001",
        strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
        benchmark_dataset=EvaluationBenchmarkDataset(
            asset_scenario_ids=["scenario-asset-0"],
            time_grid_reference="tg-a",
        ),
        metrics=[
            EvaluationMetric(name="sharpe_ratio", value=1.23),
            EvaluationMetric(name="max_drawdown", value=-0.12),
        ],
        library_version="1.0.0",
        generated_at="2026-07-03T00:00:00+00:00",
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC: a collection containing EvaluationResults for multiple strategies and
# Benchmark Datasets aggregates into a table with one row per strategy x
# Benchmark Dataset combination and one column per Metric name (FR-34)
# ---------------------------------------------------------------------------

def test_aggregate_evaluation_results_one_row_per_strategy_dataset_combination():
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationStrategy,
        aggregate_evaluation_results,
    )

    results = [
        _make_evaluation_result(
            strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a0"], time_grid_reference="tg-a"
            ),
        ),
        _make_evaluation_result(
            strategy=EvaluationStrategy(name="GlobalMinimumVariance", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a0"], time_grid_reference="tg-a"
            ),
        ),
        _make_evaluation_result(
            strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a0"], time_grid_reference="tg-b"
            ),
        ),
    ]

    table = aggregate_evaluation_results(results)

    assert len(table) == 3
    combos = {(row["strategy"], row["benchmark_dataset"]) for row in table}
    assert combos == {
        ("EqualWeight", "tg-a"),
        ("GlobalMinimumVariance", "tg-a"),
        ("EqualWeight", "tg-b"),
    }
    for row in table:
        assert row["sharpe_ratio"] == 1.23
        assert row["max_drawdown"] == -0.12


def test_aggregate_evaluation_results_columns_are_metric_names():
    from quantscenariobench.benchmark.evaluation import EvaluationMetric, aggregate_evaluation_results

    result = _make_evaluation_result(
        metrics=[
            EvaluationMetric(name="sortino_ratio", value=2.0),
            EvaluationMetric(name="final_wealth_factor", value=1.5),
        ]
    )

    table = aggregate_evaluation_results([result])

    assert len(table) == 1
    row = table[0]
    assert set(row) == {
        "strategy", "benchmark_dataset", "cost_one_way_bps",
        "sortino_ratio", "final_wealth_factor",
    }


def test_aggregate_evaluation_results_uses_latest_result_for_duplicate_combination():
    from quantscenariobench.benchmark.evaluation import EvaluationMetric, aggregate_evaluation_results

    older = _make_evaluation_result(
        generated_at="2026-07-01T00:00:00+00:00",
        metrics=[EvaluationMetric(name="sharpe_ratio", value=0.5)],
    )
    newer = _make_evaluation_result(
        generated_at="2026-07-02T00:00:00+00:00",
        metrics=[EvaluationMetric(name="sharpe_ratio", value=1.5)],
    )

    table = aggregate_evaluation_results([older, newer])

    assert len(table) == 1
    assert table[0]["sharpe_ratio"] == 1.5


# ---------------------------------------------------------------------------
# AC: a newly published EvaluationResult for a strategy/Benchmark Dataset
# combination not previously seen appears as a new row when the reader is
# re-run, with zero changes to the aggregation function's source (FR-34)
# ---------------------------------------------------------------------------

def test_aggregate_evaluation_results_adds_new_row_for_newly_seen_combination():
    from quantscenariobench.benchmark.evaluation import EvaluationBenchmarkDataset, EvaluationStrategy, aggregate_evaluation_results

    first_run = [_make_evaluation_result()]
    table_before = aggregate_evaluation_results(first_run)
    assert len(table_before) == 1

    second_run = first_run + [
        _make_evaluation_result(
            strategy=EvaluationStrategy(name="CVaROptimization", parameters={"confidence_level": 0.95}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a0"], time_grid_reference="tg-new-dataset"
            ),
        )
    ]
    table_after = aggregate_evaluation_results(second_run)

    assert len(table_after) == 2
    assert ("CVaROptimization", "tg-new-dataset") in {
        (row["strategy"], row["benchmark_dataset"]) for row in table_after
    }


# ---------------------------------------------------------------------------
# AC: the aggregation reader's return value is a plain, headlessly-testable
# table/data structure with no dependency on any UI framework (FR-34)
# ---------------------------------------------------------------------------

def test_aggregate_evaluation_results_returns_plain_list_of_dicts():
    from quantscenariobench.benchmark.evaluation import aggregate_evaluation_results

    table = aggregate_evaluation_results([_make_evaluation_result()])

    assert isinstance(table, list)
    for row in table:
        assert isinstance(row, dict)
        for value in row.values():
            # None is included for cost_one_way_bps (FR-45, AD-34): the
            # value is None when the row's EvaluationResult used no cost
            # model. dict is included for metrics_distribution (FR-46,
            # AD-35): present only on a distributional row.
            assert isinstance(value, (str, float, int, dict, type(None)))


def test_leaderboard_module_has_no_ui_framework_dependency():
    src = (_pkg_root() / "benchmark" / "evaluation" / "_leaderboard.py").read_text()
    for forbidden in ("streamlit", "gradio", "flask", "dash", "plotly", "matplotlib", "pandas"):
        assert forbidden not in src.lower()


# ---------------------------------------------------------------------------
# AC: the aggregation reader's source code contains no strategy-specific or
# dataset-specific branching — it is generic purely because
# EvaluationResult's schema (AD-26) is fixed (FR-34)
# ---------------------------------------------------------------------------

def test_leaderboard_module_has_no_strategy_or_dataset_specific_literals():
    src = (_pkg_root() / "benchmark" / "evaluation" / "_leaderboard.py").read_text()
    strategy_names = ("EqualWeight", "GlobalMinimumVariance", "CVaROptimization")
    for name in strategy_names:
        assert name not in src, f"{name} hardcoded in leaderboard module"


def test_aggregate_evaluation_results_generic_over_arbitrary_strategy_and_metric_names():
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationMetric,
        EvaluationStrategy,
        aggregate_evaluation_results,
    )

    exotic_result = _make_evaluation_result(
        strategy=EvaluationStrategy(name="SomeFutureStrategy", parameters={}),
        benchmark_dataset=EvaluationBenchmarkDataset(
            asset_scenario_ids=["a0"], time_grid_reference="some-future-dataset"
        ),
        metrics=[EvaluationMetric(name="some_future_metric", value=42.0)],
    )

    table = aggregate_evaluation_results([exotic_result])

    assert table == [
        {
            "strategy": "SomeFutureStrategy",
            "benchmark_dataset": "some-future-dataset",
            "cost_one_way_bps": None,
            "some_future_metric": 42.0,
        }
    ]


# ---------------------------------------------------------------------------
# load_evaluation_results: reads a local collection (Story 7.3's layout)
# ---------------------------------------------------------------------------

def test_load_evaluation_results_reads_back_what_was_written(tmp_path):
    from quantscenariobench.benchmark.evaluation import (
        load_evaluation_results,
        write_evaluation_result,
    )

    result = _make_evaluation_result()
    write_evaluation_result(result, root=tmp_path)

    loaded = load_evaluation_results(tmp_path)

    assert loaded == [result]


def test_load_and_aggregate_end_to_end(tmp_path):
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationStrategy,
        aggregate_evaluation_results,
        load_evaluation_results,
        write_evaluation_result,
    )

    write_evaluation_result(
        _make_evaluation_result(
            strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a0"], time_grid_reference="tg-a"
            ),
        ),
        root=tmp_path,
    )
    write_evaluation_result(
        _make_evaluation_result(
            strategy=EvaluationStrategy(name="GlobalMinimumVariance", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a0"], time_grid_reference="tg-a"
            ),
        ),
        root=tmp_path,
    )

    table = aggregate_evaluation_results(load_evaluation_results(tmp_path))

    assert len(table) == 2


# ---------------------------------------------------------------------------
# load_evaluation_results_from_hub: reads a shared HF Evaluation Results
# repo (mocked to avoid real Hub traffic, mirroring Story 3.2/7.4's approach)
# ---------------------------------------------------------------------------

def test_load_evaluation_results_from_hub_reads_snapshot(tmp_path):
    from quantscenariobench.benchmark.evaluation import (
        load_evaluation_results_from_hub,
        write_evaluation_result,
    )

    results_dir = tmp_path / "results"
    result = _make_evaluation_result()
    write_evaluation_result(result, root=results_dir)

    with mock.patch(
        "huggingface_hub.snapshot_download", return_value=str(tmp_path)
    ) as mock_snapshot:
        loaded = load_evaluation_results_from_hub("org/eval-results", token="tok")

    assert mock_snapshot.called
    assert loaded == [result]


# ---------------------------------------------------------------------------
# Story 10.3 (Issue #85) AC6: an old EvaluationResult JSON with no
# distribution block still loads and renders alongside new distributional
# rows — mixed old/new rendering, mean±std where present (FR-46)
# ---------------------------------------------------------------------------

def test_aggregate_evaluation_results_renders_mixed_old_and_distributional_rows():
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationStrategy,
        aggregate_evaluation_results,
    )

    old_style_result = _make_evaluation_result(
        strategy=EvaluationStrategy(name="EqualWeight", parameters={}),
        benchmark_dataset=EvaluationBenchmarkDataset(
            asset_scenario_ids=["a0"], time_grid_reference="tg-old"
        ),
    )
    assert old_style_result.metrics_distribution is None

    distributional_result = _make_evaluation_result(
        strategy=EvaluationStrategy(name="GlobalMinimumVariance", parameters={}),
        benchmark_dataset=EvaluationBenchmarkDataset(
            asset_scenario_ids=["a0"], time_grid_reference="tg-new"
        ),
        metrics_distribution={
            "sharpe_ratio": {
                "mean": 1.23, "std": 0.1, "ci_low": 1.0, "ci_high": 1.4,
                "n_repeats": 32, "values": [1.2, 1.3],
            },
        },
    )

    table = aggregate_evaluation_results([old_style_result, distributional_result])

    assert len(table) == 2
    rows_by_dataset = {row["benchmark_dataset"]: row for row in table}
    assert "metrics_distribution" not in rows_by_dataset["tg-old"]
    assert rows_by_dataset["tg-new"]["metrics_distribution"]["sharpe_ratio"]["mean"] == 1.23
    # The plain scalar (mean) column is unaffected either way — ranking
    # keeps using it regardless of whether a distribution is attached.
    assert rows_by_dataset["tg-new"]["sharpe_ratio"] == 1.23


def test_old_evaluation_result_json_without_distribution_block_still_loads():
    from quantscenariobench.benchmark.evaluation import EvaluationResult

    old_style_payload = {
        "schema_version": "1.0",
        "result_id": "result-0001",
        "strategy": {"name": "EqualWeight", "parameters": {}},
        "benchmark_dataset": {"asset_scenario_ids": [], "time_grid_reference": "tg-0"},
        "metrics": [{"name": "sharpe_ratio", "value": 1.0}],
        "library_version": "1.0.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        # rebalance_schedule/cost_model/metrics_distribution intentionally omitted
    }
    result = EvaluationResult.from_dict(old_style_payload)
    assert result.metrics_distribution is None
