"""
Story 8.1 — Leaderboard Space Scaffold, Data Loading & Table Rendering

Covers all acceptance criteria from GitHub Issue #58: a hosted Gradio
Space that renders Feature 4.9's Leaderboard aggregation (FR-34) as a
live table, presentation-layer-only (FR-36), staying current without a
redeploy (FR-37, AD-28).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test
from quantscenariobench.benchmark.evaluation import (
    EvaluationBenchmarkDataset,
    EvaluationMetric,
    EvaluationResult,
    EvaluationStrategy,
    aggregate_evaluation_results,
)


def _repo_root() -> Path:
    return Path(__file__).parent.parent


def _space_dir() -> Path:
    return _repo_root() / "spaces" / "leaderboard"


def _app_path() -> Path:
    return _space_dir() / "app.py"


def _load_app_module():
    """Load spaces/leaderboard/app.py as a standalone module.

    Not a package under quantscenariobench (AD-27), so it can't be
    imported via a dotted path — loaded directly from its file instead.
    """
    path = _app_path()
    spec = importlib.util.spec_from_file_location("leaderboard_space_app", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_evaluation_result(**overrides) -> EvaluationResult:
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
        library_version="1.1.0",
        generated_at="2026-07-03T00:00:00+00:00",
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# AC: spaces/leaderboard/ exists at the project root, sibling to
# quantscenariobench/ — not a submodule of it — containing app.py and
# requirements.txt (AD-27)
# ---------------------------------------------------------------------------

def test_space_directory_is_sibling_of_package_not_nested_inside_it():
    space_dir = _space_dir()
    assert space_dir.is_dir()
    # spaces/ (space_dir's parent) is the sibling of quantscenariobench/,
    # not space_dir itself — spaces/leaderboard/ is one level deeper.
    assert space_dir.parent.name == "spaces"
    assert space_dir.parent.parent == _repo_root()
    assert not (_repo_root() / "quantscenariobench" / "spaces").exists()


def test_space_directory_contains_app_and_requirements():
    assert (_space_dir() / "app.py").is_file()
    assert (_space_dir() / "requirements.txt").is_file()


# ---------------------------------------------------------------------------
# AC: requirements.txt pins gradio>=6.19 and quantscenariobench; gradio
# does not appear in the installable quantscenariobench package's own
# dependency list (AD-27, Stack)
# ---------------------------------------------------------------------------

def test_space_requirements_pin_gradio_and_quantscenariobench():
    text = (_space_dir() / "requirements.txt").read_text()
    assert "gradio" in text
    assert "quantscenariobench" in text


def test_gradio_absent_from_installable_package_dependencies():
    pyproject_text = (_repo_root() / "pyproject.toml").read_text()
    assert "gradio" not in pyproject_text.lower()


# ---------------------------------------------------------------------------
# AC: app.py's import statements import only
# quantscenariobench.benchmark.evaluation — never .strategies, .solver,
# .metrics, .returns, or .runner (FR-36, AD-27)
# ---------------------------------------------------------------------------

def _imported_module_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_app_imports_only_evaluation_submodule_of_benchmark_layer():
    tree = ast.parse(_app_path().read_text())
    imported = _imported_module_names(tree)

    forbidden_prefixes = (
        "quantscenariobench.benchmark.strategies",
        "quantscenariobench.benchmark.solver",
        "quantscenariobench.benchmark.metrics",
        "quantscenariobench.benchmark.returns",
        "quantscenariobench.benchmark.runner",
    )
    for name in imported:
        assert not name.startswith(forbidden_prefixes), (
            f"app.py must not import {name} — only benchmark.evaluation "
            "is permitted (FR-36, AD-27)"
        )

    assert "quantscenariobench.benchmark.evaluation" in imported, (
        "app.py must import quantscenariobench.benchmark.evaluation"
    )


# ---------------------------------------------------------------------------
# AC: the rendered table matches FR-34's aggregation shape exactly — no
# reshaping or reinterpretation (FR-35)
# ---------------------------------------------------------------------------

def test_build_leaderboard_dataframe_matches_fr34_aggregation_shape():
    app = _load_app_module()
    results = [
        _make_evaluation_result(),
        _make_evaluation_result(
            result_id="result-0002",
            strategy=EvaluationStrategy(name="GlobalMinimumVariance", parameters={}),
            benchmark_dataset=EvaluationBenchmarkDataset(
                asset_scenario_ids=["a1"], time_grid_reference="tg-b"
            ),
        ),
    ]

    expected_rows = aggregate_evaluation_results(results)
    df = app.build_leaderboard_dataframe(results)

    assert list(df.to_dict(orient="records")) == expected_rows
    assert set(df.columns) == set(expected_rows[0].keys())


# ---------------------------------------------------------------------------
# AC: a newly published EvaluationResult appears on the next Gradio
# session load, with no code change or manual redeploy (FR-37, AD-28) —
# and the loading code carries no server-side cache (AD-28)
# ---------------------------------------------------------------------------

def test_load_leaderboard_table_reflects_latest_data_each_call_no_caching():
    app = _load_app_module()

    first_batch = [_make_evaluation_result()]
    second_batch = first_batch + [
        _make_evaluation_result(
            result_id="result-0002",
            strategy=EvaluationStrategy(name="GlobalMinimumVariance", parameters={}),
        )
    ]
    call_results = [first_batch, second_batch]

    def fake_fetch(repo_id, *, token=None):
        return call_results.pop(0)

    with mock.patch.object(app, "fetch_evaluation_results", side_effect=fake_fetch):
        first_table = app.load_leaderboard_table("org/eval-results")
        second_table = app.load_leaderboard_table("org/eval-results")

    assert len(first_table) == 1
    assert len(second_table) == 2, (
        "load_leaderboard_table must re-fetch on every call — a cached "
        "result would still show 1 row (AD-28)"
    )


def test_leaderboard_data_loading_has_no_cache_or_scheduler():
    src = _app_path().read_text()
    for forbidden in ("lru_cache", "cachetools", "@cache", "schedule.every", "cron"):
        assert forbidden not in src, (
            f"app.py must not introduce a cache/scheduler ('{forbidden}' found) — "
            "v1.2 data currency is read-on-session-load only (AD-28)"
        )


def test_fetch_evaluation_results_delegates_to_hub_loader():
    app = _load_app_module()

    # Patch the name as bound inside app.py's own namespace (a `from ... import`
    # captures the function object at import time) — not the origin module,
    # which patching there would not affect (classic "patch where it's used").
    with mock.patch.object(
        app, "load_evaluation_results_from_hub",
        return_value=[_make_evaluation_result()],
    ) as mock_loader:
        results = app.fetch_evaluation_results("org/eval-results", token="tok")

    mock_loader.assert_called_once_with("org/eval-results", token="tok")
    assert len(results) == 1


def test_eval_results_repo_overridable_via_env_var(monkeypatch):
    app = _load_app_module()
    monkeypatch.setenv("QSB_EVAL_RESULTS_REPO", "someorg/some-eval-results")
    assert app._eval_results_repo() == "someorg/some-eval-results"


def test_eval_results_repo_has_a_default_when_env_var_unset(monkeypatch):
    app = _load_app_module()
    monkeypatch.delenv("QSB_EVAL_RESULTS_REPO", raising=False)
    assert app._eval_results_repo() == app.DEFAULT_EVAL_RESULTS_REPO


# ---------------------------------------------------------------------------
# AC: app.py's rendering code contains no aggregation, ranking, or
# scoring computation of its own — every value shown is read directly
# from the aggregation function's output (FR-36)
# ---------------------------------------------------------------------------

def test_app_has_no_reimplemented_ranking_or_scoring_logic():
    src = _app_path().read_text().lower()
    forbidden_terms = ("sharpe", "sortino", "drawdown", "wealth_factor", "sort_values", ".rank(")
    for term in forbidden_terms:
        assert term not in src, (
            f"app.py must not reimplement scoring/ranking ('{term}' found) — "
            "it must consume FR-34's aggregation output only (FR-36)"
        )


def test_build_app_returns_gradio_blocks_with_dataframe_and_load_hook():
    import gradio as gr

    app = _load_app_module()
    assert isinstance(app.demo, gr.Blocks)
