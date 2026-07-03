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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="QuantScenarioBench Leaderboard") as demo:
        gr.Markdown("# QuantScenarioBench Leaderboard")
        table = gr.Dataframe(interactive=False, wrap=True)
        demo.load(fn=load_leaderboard_table, inputs=None, outputs=table)
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
