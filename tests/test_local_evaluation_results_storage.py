"""
Story 7.3 — Local Evaluation Results Storage

Covers all acceptance criteria from GitHub Issue #48: writing an
EvaluationResult to a local, organized, append-only file layout (FR-32).
"""

from __future__ import annotations

import json


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
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# AC: an EvaluationResult for a given Benchmark Dataset and strategy is
# saved as a timestamped JSON file under a directory keyed by Benchmark
# Dataset and strategy name (FR-32)
# ---------------------------------------------------------------------------

def test_write_evaluation_result_creates_json_file_under_dataset_and_strategy_directory(tmp_path):
    from quantscenariobench.benchmark.evaluation import write_evaluation_result

    result = _make_evaluation_result()

    path = write_evaluation_result(result, root=tmp_path)

    assert path.exists()
    assert path.suffix == ".json"
    assert path.name.startswith("result_")
    assert path.parent == tmp_path / "tg-daily-2026-07-02" / "EqualWeight"


def test_write_evaluation_result_content_round_trips():
    import dataclasses
    import tempfile
    from pathlib import Path

    from quantscenariobench.benchmark.evaluation import write_evaluation_result

    result = _make_evaluation_result()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_evaluation_result(result, root=Path(tmpdir))
        written = json.loads(path.read_text())

    assert written == dataclasses.asdict(result)


# ---------------------------------------------------------------------------
# AC: two EvaluationResults written for the same Benchmark Dataset/strategy
# combination produce two separate files — neither write overwrites the
# other (FR-32)
# ---------------------------------------------------------------------------

def test_write_evaluation_result_twice_for_same_combination_creates_two_files(tmp_path):
    from quantscenariobench.benchmark.evaluation import write_evaluation_result

    first_result = _make_evaluation_result(result_id="result-0001")
    second_result = _make_evaluation_result(result_id="result-0002")

    first_path = write_evaluation_result(first_result, root=tmp_path)
    second_path = write_evaluation_result(second_result, root=tmp_path)

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert json.loads(first_path.read_text())["result_id"] == "result-0001"
    assert json.loads(second_path.read_text())["result_id"] == "result-0002"

    directory = tmp_path / "tg-daily-2026-07-02" / "EqualWeight"
    assert len(list(directory.glob("*.json"))) == 2


def test_write_evaluation_result_never_collides_even_for_identical_content(tmp_path):
    from quantscenariobench.benchmark.evaluation import write_evaluation_result

    result = _make_evaluation_result()

    paths = {write_evaluation_result(result, root=tmp_path) for _ in range(5)}

    assert len(paths) == 5
    for path in paths:
        assert path.exists()


# ---------------------------------------------------------------------------
# AC: the local file layout requires no reorganization to be uploaded as-is
# to the Hugging Face path of Story 7.4 (FR-32) — verified by asserting the
# directory tree is exactly <root>/<dataset>/<strategy>/*.json with no
# extra nesting or nondeterministic structure
# ---------------------------------------------------------------------------

def test_local_layout_is_flat_dataset_then_strategy_then_result_files(tmp_path):
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationStrategy,
        write_evaluation_result,
    )

    write_evaluation_result(
        _make_evaluation_result(
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["scenario-asset-0"],
                time_grid_reference="tg-alpha",
            ),
            strategy=EvaluationStrategy(name="GlobalMinimumVariance", parameters={}),
        ),
        root=tmp_path,
    )

    all_files = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    assert len(all_files) == 1
    relative_path = all_files[0]
    assert relative_path.parts[0] == "tg-alpha"
    assert relative_path.parts[1] == "GlobalMinimumVariance"
    assert len(relative_path.parts) == 3
    assert relative_path.suffix == ".json"
