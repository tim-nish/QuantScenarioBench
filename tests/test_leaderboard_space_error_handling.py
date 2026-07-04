"""
Leaderboard Space — graceful handling of a missing/empty/private/
inaccessible Evaluation Results repo.

Bug: the Space crashed with `RepositoryNotFoundError` when the default
(or configured) Evaluation Results repo did not exist on the Hub. This
covers `load_leaderboard_safely()`, which wraps `load_leaderboard_table()`
so the Space always renders — an empty table plus a clear, user-facing
message — instead of raising.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import httpx
import pandas as pd
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test
from quantscenariobench.benchmark.evaluation import (
    EvaluationBenchmarkDataset,
    EvaluationMetric,
    EvaluationResult,
    EvaluationStrategy,
)


def _app_path() -> Path:
    return Path(__file__).parent.parent / "spaces" / "leaderboard" / "app.py"


def _load_app_module():
    path = _app_path()
    spec = importlib.util.spec_from_file_location("leaderboard_space_app_errors", path)
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
            asset_scenario_ids=["a0"], time_grid_reference="tg-a"
        ),
        metrics=[EvaluationMetric(name="sharpe_ratio", value=1.23)],
        library_version="1.1.0",
        generated_at="2026-07-03T00:00:00+00:00",
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


def _hf_http_error(cls, message: str):
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/org/repo")
    response = httpx.Response(status_code=401, request=request)
    return cls(message, response=response)


# ---------------------------------------------------------------------------
# Bug: RepositoryNotFoundError (missing or unauthenticated-private repo)
# must not crash the Space — it must render an empty table plus a clear
# message instead.
# ---------------------------------------------------------------------------

def test_missing_repo_renders_empty_table_with_message_instead_of_raising():
    app = _load_app_module()

    with mock.patch.object(
        app,
        "fetch_evaluation_results",
        side_effect=_hf_http_error(RepositoryNotFoundError, "Repository Not Found"),
    ):
        result = app.load_leaderboard_safely("quantscenariobench/evaluation-results")

    assert result.table.empty
    assert result.message is not None
    assert "quantscenariobench/evaluation-results" in result.message


# ---------------------------------------------------------------------------
# Gated/private repos raise GatedRepoError (a RepositoryNotFoundError
# subclass) — must be handled the same way, not just the base class.
# ---------------------------------------------------------------------------

def test_gated_or_private_repo_renders_empty_table_with_message_instead_of_raising():
    app = _load_app_module()

    with mock.patch.object(
        app,
        "fetch_evaluation_results",
        side_effect=_hf_http_error(GatedRepoError, "Access to this repo is restricted"),
    ):
        result = app.load_leaderboard_safely("some-org/private-eval-results")

    assert result.table.empty
    assert result.message is not None


# ---------------------------------------------------------------------------
# A repo that is reachable but has zero published results must also
# render gracefully with an explanatory message, not a silently blank
# table that looks broken.
# ---------------------------------------------------------------------------

def test_empty_repo_renders_empty_table_with_message():
    app = _load_app_module()

    with mock.patch.object(app, "fetch_evaluation_results", return_value=[]):
        result = app.load_leaderboard_safely("org/eval-results")

    assert result.table.empty
    assert result.message is not None
    assert "no published" in result.message.lower()


# ---------------------------------------------------------------------------
# Requirement: preserve existing behavior when the repo is reachable and
# has results — same data as load_leaderboard_table, no message.
# ---------------------------------------------------------------------------

def test_reachable_repo_with_results_returns_data_and_no_message():
    app = _load_app_module()
    fixture_results = [_make_evaluation_result()]

    with mock.patch.object(app, "fetch_evaluation_results", return_value=fixture_results):
        safe_result = app.load_leaderboard_safely("org/eval-results")
        direct_table = app.load_leaderboard_table("org/eval-results")

    assert safe_result.message is None
    assert not safe_result.table.empty
    assert safe_result.table.to_dict(orient="records") == direct_table.to_dict(orient="records")


def test_load_leaderboard_table_still_raises_directly_unwrapped():
    # load_leaderboard_table() itself is unchanged — it still raises.
    # Only the new load_leaderboard_safely() wrapper catches errors, so
    # existing callers/tests of the unwrapped function keep working.
    app = _load_app_module()

    with mock.patch.object(
        app,
        "fetch_evaluation_results",
        side_effect=_hf_http_error(RepositoryNotFoundError, "Repository Not Found"),
    ):
        try:
            app.load_leaderboard_table("org/missing-repo")
            raised = False
        except RepositoryNotFoundError:
            raised = True
    assert raised, "load_leaderboard_table must still raise — only the safe wrapper catches"


# ---------------------------------------------------------------------------
# The Space's load hook must use the safe wrapper, and never crash
# rebuilding the app itself when the configured repo is unreachable.
# ---------------------------------------------------------------------------

def test_app_uses_safe_loader_not_the_raising_one_in_its_load_hook():
    src = _app_path().read_text()
    assert "load_leaderboard_safely" in src
    # The demo.load(...) wiring should reference the safe function, not
    # call the raising load_leaderboard_table directly as the load hook.
    assert "fn=_on_load" in src


def test_build_app_still_builds_a_valid_blocks_app():
    import gradio as gr

    app = _load_app_module()
    assert isinstance(app.demo, gr.Blocks)


def test_status_message_component_exists_and_is_hidden_by_default():
    import gradio as gr

    app = _load_app_module()
    markdown_components = [c for c in app.demo.blocks.values() if isinstance(c, gr.Markdown)]
    # One Markdown for the page title, one for the status message.
    assert len(markdown_components) == 2
    assert any(c.visible is False for c in markdown_components), (
        "the status message should start hidden and only appear when "
        "load_leaderboard_safely reports a problem or empty result"
    )


# ---------------------------------------------------------------------------
# FR-36: this fix is resilience/UX only — no new aggregation, ranking,
# or scoring logic was introduced to handle these cases.
# ---------------------------------------------------------------------------

def test_error_handling_introduces_no_reimplemented_ranking_or_scoring_logic():
    src = _app_path().read_text().lower()
    forbidden_terms = ("sharpe", "sortino", "drawdown", "wealth_factor", "sort_values", ".rank(")
    for term in forbidden_terms:
        assert term not in src


def test_pandas_dataframe_empty_check_used_not_a_bespoke_emptiness_check():
    # Sanity: `table.empty` (pandas' own semantics) drives the "no
    # results yet" branch, not a hand-rolled len()==0 check that could
    # drift from real empty-DataFrame semantics.
    src = _app_path().read_text()
    assert "table.empty" in src or ".empty" in src
