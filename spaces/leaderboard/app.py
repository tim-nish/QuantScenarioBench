"""QuantScenarioBench Leaderboard Space (PRD Feature 4.10, Epic 8, Story 8.1).

A Hugging Face Space rendering Feature 4.9's Leaderboard aggregation
(FR-34) as a live, browsable table. Presentation layer only (FR-36):
this file consumes quantscenariobench.benchmark.evaluation's existing
aggregation and Hub-loading functions and adds no aggregation, ranking,
or data-model logic of its own (AD-27).

Data currency (FR-37, AD-28): the Leaderboard table is loaded fresh on
every Gradio session load -- no server-side cache, scheduled job, or
webhook in v1.2 (see Architecture Spine Deferred section for the
upgrade path if traffic or the Evaluation Results repo's size grows).
"""
from __future__ import annotations

import os
from typing import Sequence

import gradio as gr
import pandas as pd

from quantscenariobench.benchmark.evaluation import (
    EvaluationResult,
    aggregate_evaluation_results,
    load_evaluation_results_from_hub,
)

# [ASSUMPTION] Hugging Face namespace/naming convention for the shared
# Evaluation Results repo is still undecided (PRD Open Questions 18, 22;
# Architecture Spine Deferred). Overridable via QSB_EVAL_RESULTS_REPO so
# this Space needs no code change once a real namespace is chosen.
DEFAULT_EVAL_RESULTS_REPO = "quantscenariobench/evaluation-results"


def _eval_results_repo() -> str:
    return os.environ.get("QSB_EVAL_RESULTS_REPO", DEFAULT_EVAL_RESULTS_REPO)


def build_leaderboard_dataframe(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Turn already-loaded EvaluationResults into the displayed table.

    A thin pandas.DataFrame wrapper around FR-34's
    aggregate_evaluation_results -- no reshaping, no derived columns, no
    ranking logic of its own (FR-36).
    """
    rows = aggregate_evaluation_results(results)
    return pd.DataFrame(rows)


def fetch_evaluation_results(
    repo_id: str, *, token: str | None = None
) -> list[EvaluationResult]:
    """Thin wrapper around Feature 4.9's Hub-loading function.

    Kept as its own function (rather than inlined) so tests can
    substitute a fixture loader without making real Hub network calls.
    """
    return load_evaluation_results_from_hub(repo_id, token=token)


def load_leaderboard_table(
    repo_id: str | None = None, *, token: str | None = None
) -> pd.DataFrame:
    """Load the current Leaderboard fresh -- called once per Gradio session load (FR-37, AD-28).

    No caching: every call re-fetches from the shared Evaluation Results
    repo, so a newly published EvaluationResult is visible on the very
    next session without redeploying this Space.
    """
    results = fetch_evaluation_results(repo_id or _eval_results_repo(), token=token)
    return build_leaderboard_dataframe(results)


_ROW_KEY_COLUMNS = ("strategy", "benchmark_dataset")


def filter_leaderboard(
    df: pd.DataFrame,
    benchmark_datasets: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Narrow the displayed table by Benchmark Dataset, Strategy, and/or Metric (FR-39).

    A pure view over `df` -- never mutates it (a new DataFrame is always
    returned) and never re-derives or re-ranks any value, only selects
    rows/columns already present in `df` (FR-36). Active filters combine
    with AND: selecting a Benchmark Dataset and a Strategy narrows to
    rows satisfying both, not either. Generic over whatever Benchmark
    Dataset/Strategy/Metric values are actually present -- no
    strategy/dataset/metric name is hardcoded here.
    """
    result = df
    if benchmark_datasets:
        result = result[result["benchmark_dataset"].isin(benchmark_datasets)]
    if strategies:
        result = result[result["strategy"].isin(strategies)]
    if metrics:
        keep = [c for c in _ROW_KEY_COLUMNS if c in result.columns]
        keep += [c for c in metrics if c in result.columns]
        result = result[keep]
    return result.reset_index(drop=True)


def _unique_sorted_values(series: pd.Series | None) -> list[str]:
    if series is None or series.empty:
        return []
    return sorted(series.unique().tolist())


def _metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _ROW_KEY_COLUMNS]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="QuantScenarioBench Leaderboard") as demo:
        gr.Markdown("# QuantScenarioBench Leaderboard")

        # The unfiltered table loaded this session (FR-37, AD-28) -- filter
        # controls below narrow the *display* only; this snapshot is never
        # mutated, so re-filtering always starts from the full result set.
        table_state = gr.State(pd.DataFrame())

        with gr.Row():
            dataset_filter = gr.Dropdown(
                label="Benchmark Dataset", multiselect=True, choices=[]
            )
            strategy_filter = gr.Dropdown(label="Strategy", multiselect=True, choices=[])
            metric_filter = gr.Dropdown(label="Metric", multiselect=True, choices=[])

        # Sorting (FR-38) is Gradio Dataframe's built-in, client-side
        # column-header sort -- clicking a header reorders the displayed
        # rows in the browser without a Python round-trip, and never
        # touches the underlying data (Gradio's own docs: "sorting the
        # columns in the browser will not affect the values passed to
        # this function"). No bespoke sort algorithm or callback is
        # introduced here (AD-27) -- interactive=False only disables
        # cell editing, not this native sort affordance. Because the
        # filtered view below re-renders into this same component, a
        # header-click sort applies to whatever is currently displayed
        # (the filtered rows/columns), not the full unfiltered table.
        table = gr.Dataframe(interactive=False, wrap=True)

        def _on_load():
            df = load_leaderboard_table()
            return (
                df,
                df,
                gr.update(choices=_unique_sorted_values(df.get("benchmark_dataset"))),
                gr.update(choices=_unique_sorted_values(df.get("strategy"))),
                gr.update(choices=_metric_columns(df)),
            )

        demo.load(
            fn=_on_load,
            inputs=None,
            outputs=[table_state, table, dataset_filter, strategy_filter, metric_filter],
        )

        def _on_filter_change(df, datasets, strategies, metrics):
            return filter_leaderboard(df, datasets, strategies, metrics)

        for control in (dataset_filter, strategy_filter, metric_filter):
            control.change(
                fn=_on_filter_change,
                inputs=[table_state, dataset_filter, strategy_filter, metric_filter],
                outputs=table,
            )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
