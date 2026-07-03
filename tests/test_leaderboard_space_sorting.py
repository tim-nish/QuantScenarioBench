"""
Story 8.2 — Leaderboard Sorting

Covers all acceptance criteria from GitHub Issue #59: sorting the
displayed Leaderboard table by any column (FR-38).

Gradio's `Dataframe` component provides column-header click-to-sort as a
built-in, client-side browser feature — Gradio's own docs state
"sorting the columns in the browser will not affect the values passed
to this function". That means: (a) no Python sort logic is needed to
satisfy FR-38's row-reordering and ascending/descending behavior, and
(b) the underlying aggregated data is untouched by construction, not
just by convention. This story's tests therefore verify absence of a
bespoke reimplementation (AD-27) rather than a custom sort function's
correctness — there is no custom sort function to test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _app_path() -> Path:
    return Path(__file__).parent.parent / "spaces" / "leaderboard" / "app.py"


def _load_app_module():
    path = _app_path()
    spec = importlib.util.spec_from_file_location("leaderboard_space_app_sorting", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# AC: app.py's sorting implementation uses Gradio Dataframe's native
# column-sort — no bespoke sort algorithm or aggregation-layer change
# was introduced to implement it (AD-27)
# ---------------------------------------------------------------------------

def test_app_has_no_bespoke_sort_implementation():
    src = _app_path().read_text()
    # Scoped to reimplementing the *table's* sort order specifically —
    # not a blanket ban on Python's sorted()/list.sort(), which Story
    # 8.3's filter-choice population legitimately uses for presenting
    # dropdown options alphabetically (an unrelated UI concern).
    forbidden = ("sort_values(", "def sort_leaderboard", "def sort_table")
    for term in forbidden:
        assert term not in src, (
            f"app.py must not reimplement table sorting ('{term}' found) — "
            "FR-38 is satisfied by Gradio Dataframe's native column-header "
            "sort, not custom Python logic (AD-27)"
        )


def test_app_documents_native_sort_reliance():
    src = _app_path().read_text()
    # Guards against a future edit silently deleting this rationale and
    # reintroducing a bespoke sort without a reviewer noticing why one
    # was never needed in the first place.
    assert "AD-27" in src and "sort" in src.lower(), (
        "app.py should document that FR-38's sorting relies on Gradio's "
        "native Dataframe behavior, not a custom implementation"
    )


# ---------------------------------------------------------------------------
# AC: sorting only reorders the display — the underlying aggregated data
# (FR-34's output) is unchanged (FR-38). Verified at the data layer: the
# Dataframe's displayed value is exactly Story 8.1's unmodified output,
# since no code path in app.py mutates or reorders it server-side.
# ---------------------------------------------------------------------------

def test_leaderboard_dataframe_component_still_present_and_unmodified_by_sorting():
    import gradio as gr

    app = _load_app_module()
    assert isinstance(app.demo, gr.Blocks)
    # build_leaderboard_dataframe (Story 8.1) remains the sole source of
    # the displayed table's contents; no post-processing step reorders
    # or reshapes its output as part of this story.
    from quantscenariobench.benchmark.evaluation import (
        EvaluationBenchmarkDataset,
        EvaluationMetric,
        EvaluationResult,
        EvaluationStrategy,
        aggregate_evaluation_results,
    )

    result = EvaluationResult(
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
    expected = aggregate_evaluation_results([result])
    df = app.build_leaderboard_dataframe([result])
    assert list(df.to_dict(orient="records")) == expected
