"""
Story 8.3 — Leaderboard Filtering

Covers all acceptance criteria from GitHub Issue #60: filtering the
displayed Leaderboard table by Benchmark Dataset, Strategy, and Metric,
independently or in combination (FR-39), without touching the
underlying aggregated data (FR-34's output) or introducing new
aggregation/ranking logic (FR-36, AD-27).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _app_path() -> Path:
    return Path(__file__).parent.parent / "spaces" / "leaderboard" / "app.py"


def _load_app_module():
    path = _app_path()
    spec = importlib.util.spec_from_file_location("leaderboard_space_app_filtering", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy": "EqualWeight", "benchmark_dataset": "tg-a", "sharpe_ratio": 1.0, "max_drawdown": -0.1},
            {"strategy": "GlobalMinimumVariance", "benchmark_dataset": "tg-a", "sharpe_ratio": 1.5, "max_drawdown": -0.05},
            {"strategy": "EqualWeight", "benchmark_dataset": "tg-b", "sharpe_ratio": 0.8, "max_drawdown": -0.2},
        ]
    )


# ---------------------------------------------------------------------------
# AC: selecting a Benchmark Dataset filter narrows the table to rows for
# that Benchmark Dataset only (FR-39)
# ---------------------------------------------------------------------------

def test_filter_by_benchmark_dataset_narrows_rows():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(df, benchmark_datasets=["tg-a"])

    assert len(filtered) == 2
    assert set(filtered["benchmark_dataset"]) == {"tg-a"}


# ---------------------------------------------------------------------------
# AC: selecting a Strategy filter narrows the table to rows for that
# Strategy only (FR-39)
# ---------------------------------------------------------------------------

def test_filter_by_strategy_narrows_rows():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(df, strategies=["EqualWeight"])

    assert len(filtered) == 2
    assert set(filtered["strategy"]) == {"EqualWeight"}


# ---------------------------------------------------------------------------
# AC: selecting a Metric filter narrows the displayed columns to the
# selected Metric(s) only (FR-39)
# ---------------------------------------------------------------------------

def test_filter_by_metric_narrows_columns_but_keeps_row_key_columns():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(df, metrics=["sharpe_ratio"])

    assert list(filtered.columns) == ["strategy", "benchmark_dataset", "sharpe_ratio"]
    assert len(filtered) == len(df)  # metric filter narrows columns, not rows


# ---------------------------------------------------------------------------
# AC: a Benchmark Dataset filter and a Strategy filter together show only
# rows satisfying both simultaneously (FR-39)
# ---------------------------------------------------------------------------

def test_combined_dataset_and_strategy_filters_intersect():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(
        df, benchmark_datasets=["tg-a"], strategies=["EqualWeight"]
    )

    assert len(filtered) == 1
    row = filtered.iloc[0]
    assert row["benchmark_dataset"] == "tg-a"
    assert row["strategy"] == "EqualWeight"


def test_all_three_filter_dimensions_combine():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(
        df,
        benchmark_datasets=["tg-a"],
        strategies=["EqualWeight", "GlobalMinimumVariance"],
        metrics=["max_drawdown"],
    )

    assert list(filtered.columns) == ["strategy", "benchmark_dataset", "max_drawdown"]
    assert set(filtered["strategy"]) == {"EqualWeight", "GlobalMinimumVariance"}
    assert set(filtered["benchmark_dataset"]) == {"tg-a"}


# ---------------------------------------------------------------------------
# AC: with any combination of active filters, the underlying aggregated
# data (FR-34's output) is unchanged — filtering is a view, not a
# mutation (FR-39)
# ---------------------------------------------------------------------------

def test_filtering_never_mutates_the_source_dataframe():
    app = _load_app_module()
    df = _sample_table()
    original = df.copy(deep=True)

    app.filter_leaderboard(df, benchmark_datasets=["tg-a"])
    app.filter_leaderboard(df, strategies=["EqualWeight"])
    app.filter_leaderboard(df, metrics=["sharpe_ratio"])

    pd.testing.assert_frame_equal(df, original)


def test_no_active_filters_returns_full_table_unchanged():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(df)

    pd.testing.assert_frame_equal(filtered, df)


# ---------------------------------------------------------------------------
# Edge case: a filter combination matching no rows renders cleanly
# rather than erroring
# ---------------------------------------------------------------------------

def test_filter_combination_matching_no_rows_returns_empty_dataframe_not_error():
    app = _load_app_module()
    df = _sample_table()

    filtered = app.filter_leaderboard(
        df, benchmark_datasets=["tg-a"], strategies=["StrategyThatDoesNotExist"]
    )

    assert len(filtered) == 0
    assert list(filtered.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# AC: filters are implemented as Gradio Dropdown inputs re-slicing the
# already-loaded DataFrame locally in app.py — no aggregation-layer
# change was introduced (FR-36, AD-27)
# ---------------------------------------------------------------------------

def test_app_has_no_hardcoded_strategy_or_dataset_names_in_filter_logic():
    src = _app_path().read_text()
    hardcoded_strategy_names = ("EqualWeight", "GlobalMinimumVariance", "CVaROptimization")
    for name in hardcoded_strategy_names:
        assert name not in src, (
            f"app.py's filtering must stay generic — {name} must not be "
            "hardcoded (mirrors FR-34's own genericity convention)"
        )


def test_filter_controls_are_gradio_dropdowns():
    src = _app_path().read_text()
    assert "gr.Dropdown" in src
    assert "multiselect=True" in src


def test_build_app_wires_filters_to_the_same_table_component_sort_operates_on():
    import gradio as gr

    app = _load_app_module()
    assert isinstance(app.demo, gr.Blocks)
    # Confirms there's exactly one rendered Dataframe (the same component
    # both the filter callbacks and Gradio's native sort act on) — a
    # second, separate render path would let sort and filter diverge.
    dataframe_components = [
        c for c in app.demo.blocks.values() if isinstance(c, gr.Dataframe)
    ]
    assert len(dataframe_components) == 1
