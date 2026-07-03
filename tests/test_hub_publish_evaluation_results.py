"""
Story 7.4 — Hugging Face Evaluation Results Publishing

Covers all acceptance criteria from GitHub Issue #49: publishing one or
more EvaluationResults to a shared Hugging Face dataset repo (FR-33).

Mirrors Story 3.2's approach: HfApi calls are mocked to avoid real Hub
traffic while still exercising the full upload/card-generation codepath.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

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


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC: one or more EvaluationResults published to a target HF dataset repo
# ID are each uploaded as their own file under the repo's results layout
# (mirroring Story 7.3's local organization), and none of the repo's
# previously published results is overwritten (FR-33)
# ---------------------------------------------------------------------------

def test_publish_evaluation_results_uploads_one_file_per_result():
    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    results = [
        _make_evaluation_result(result_id="result-0001"),
        _make_evaluation_result(result_id="result-0002"),
    ]

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch("huggingface_hub.HfApi.list_repo_files", return_value=[]),
        mock.patch("huggingface_hub.HfApi.upload_file") as mock_upload,
    ):
        publish_evaluation_results(results, "org/eval-results", token="tok")

    # 2 results + 1 README
    assert mock_upload.call_count == 3
    uploaded_paths = [kw["path_in_repo"] for _, kw in mock_upload.call_args_list]
    result_paths = [p for p in uploaded_paths if p != "README.md"]
    assert len(result_paths) == 2
    assert len(set(result_paths)) == 2  # each result gets its own distinct path


def test_publish_evaluation_results_mirrors_local_directory_organization():
    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    result = _make_evaluation_result()

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch("huggingface_hub.HfApi.list_repo_files", return_value=[]),
        mock.patch("huggingface_hub.HfApi.upload_file") as mock_upload,
    ):
        publish_evaluation_results([result], "org/eval-results", token="tok")

    uploaded_paths = [kw["path_in_repo"] for _, kw in mock_upload.call_args_list]
    result_path = next(p for p in uploaded_paths if p != "README.md")

    assert result_path.startswith("results/tg-daily-2026-07-02/EqualWeight/result_")
    assert result_path.endswith(".json")


def test_publish_evaluation_results_does_not_overwrite_previously_published_results():
    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    previously_published = [
        "results/tg-daily-2026-07-02/EqualWeight/result_20260701T000000000000_aaaaaaaa.json",
    ]

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch(
            "huggingface_hub.HfApi.list_repo_files", return_value=list(previously_published)
        ),
        mock.patch("huggingface_hub.HfApi.upload_file") as mock_upload,
    ):
        publish_evaluation_results([_make_evaluation_result()], "org/eval-results", token="tok")

    uploaded_paths = [kw["path_in_repo"] for _, kw in mock_upload.call_args_list]
    # the new result's path must differ from the pre-existing one — never
    # re-uploaded under the same path_in_repo (i.e. never overwritten)
    new_result_path = next(p for p in uploaded_paths if p != "README.md")
    assert new_result_path not in previously_published


# ---------------------------------------------------------------------------
# AC: after a publish call completes, the repo's README/card is
# regenerated to reflect the current set of published results (FR-33, FR-15)
# ---------------------------------------------------------------------------

def test_publish_evaluation_results_uploads_a_readme():
    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch("huggingface_hub.HfApi.list_repo_files", return_value=[]),
        mock.patch("huggingface_hub.HfApi.upload_file") as mock_upload,
    ):
        publish_evaluation_results([_make_evaluation_result()], "org/eval-results", token="tok")

    readme_calls = [kw for _, kw in mock_upload.call_args_list if kw["path_in_repo"] == "README.md"]
    assert len(readme_calls) == 1


def test_readme_reflects_previously_published_results_plus_newly_published_ones():
    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    previously_published = [
        "results/tg-daily-2026-07-02/GlobalMinimumVariance/result_20260701T000000000000_aaaaaaaa.json",
    ]

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch(
            "huggingface_hub.HfApi.list_repo_files", return_value=list(previously_published)
        ),
        mock.patch("huggingface_hub.HfApi.upload_file") as mock_upload,
    ):
        publish_evaluation_results([_make_evaluation_result()], "org/eval-results", token="tok")

    readme_body = next(
        kw["path_or_fileobj"]
        for _, kw in mock_upload.call_args_list
        if kw["path_in_repo"] == "README.md"
    ).decode()

    assert "GlobalMinimumVariance" in readme_body  # pre-existing result
    assert "EqualWeight" in readme_body            # newly published result
    assert "tg-daily-2026-07-02" in readme_body


def test_generate_evaluation_results_card_summarizes_dataset_and_strategy_counts():
    from quantscenariobench.benchmark.evaluation import generate_evaluation_results_card

    files = [
        "results/tg-a/EqualWeight/result_1.json",
        "results/tg-a/EqualWeight/result_2.json",
        "results/tg-a/GlobalMinimumVariance/result_1.json",
    ]

    card = generate_evaluation_results_card(files)

    assert isinstance(card, str)
    assert "tg-a" in card
    assert "EqualWeight" in card
    assert "GlobalMinimumVariance" in card
    assert "2" in card  # EqualWeight has 2 published results


def test_generate_evaluation_results_card_handles_empty_repo():
    from quantscenariobench.benchmark.evaluation import generate_evaluation_results_card

    card = generate_evaluation_results_card([])
    assert isinstance(card, str)
    assert len(card) > 0


# ---------------------------------------------------------------------------
# AC: the Hugging Face publishing function's source code consumes
# EvaluationResult exclusively — it never reads a BenchmarkResult
# directly (AD-26)
# ---------------------------------------------------------------------------

def test_hub_publish_module_never_references_benchmark_result():
    import ast

    src = (_pkg_root() / "benchmark" / "evaluation" / "_hub_publish.py").read_text()
    tree = ast.parse(src)

    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "BenchmarkResult" not in referenced_names
    assert "BenchmarkResult" not in imported_names


def test_publish_evaluation_results_returns_hub_url():
    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch("huggingface_hub.HfApi.list_repo_files", return_value=[]),
        mock.patch("huggingface_hub.HfApi.upload_file"),
    ):
        url = publish_evaluation_results(
            [_make_evaluation_result()], "my-org/eval-results", token="tok"
        )

    assert url == "https://huggingface.co/datasets/my-org/eval-results"


def test_publish_evaluation_results_requires_huggingface_hub_installed(monkeypatch):
    import builtins

    from quantscenariobench.benchmark.evaluation import publish_evaluation_results

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError):
        publish_evaluation_results([_make_evaluation_result()], "org/eval-results")
