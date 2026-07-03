"""Hugging Face Evaluation Results publishing (FR-33, AD-26).

Consumes EvaluationResult exclusively — never reads a BenchmarkResult
directly (AD-26); BenchmarkResult -> EvaluationResult conversion is
Story 7.2's concern, not this module's.
"""
from __future__ import annotations

import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ._evaluation_result import EvaluationResult
from ._local_storage import write_evaluation_result

_RESULTS_PREFIX = "results"

_RESULT_PATH_RE = re.compile(
    rf"^{_RESULTS_PREFIX}/(?P<dataset>[^/]+)/(?P<strategy>[^/]+)/[^/]+\.json$"
)

_CARD_TEMPLATE = """\
---
license: mit
tags:
  - finance
  - benchmark
  - evaluation-results
  - leaderboard
  - quantscenariobench
---

# QuantScenarioBench Evaluation Results

Published, versioned `EvaluationResult`s (FR-30–FR-33), each derived from a
`BenchmarkResult` produced by [QuantScenarioBench](https://github.com/tim-nish/QuantScenarioBench)'s
`run_benchmark()`. This repository accumulates an append-only history —
publishing never overwrites a previously published result.

## Published Results

| Benchmark Dataset | Strategy | Published Results |
|--------------------|----------|--------------------|
{summary_table_rows}
"""


def _summarize_result_files(result_files: Sequence[str]) -> str:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for path in result_files:
        match = _RESULT_PATH_RE.match(path)
        if match is None:
            continue
        counts[(match.group("dataset"), match.group("strategy"))] += 1

    if not counts:
        return "| _(none yet)_ | | |"

    rows = [
        f"| `{dataset}` | `{strategy}` | {count} |"
        for (dataset, strategy), count in sorted(counts.items())
    ]
    return "\n".join(rows)


def generate_evaluation_results_card(result_files: Sequence[str]) -> str:
    """Generate the shared Evaluation Results repo's README.md (FR-33, FR-15).

    result_files is the complete set of `results/**/*.json` paths currently
    published to the repo — the card always reflects the current set of
    published results, mirroring generate_dataset_card's role for Benchmark
    Datasets.
    """
    return _CARD_TEMPLATE.format(summary_table_rows=_summarize_result_files(result_files))


def publish_evaluation_results(
    results: Sequence[EvaluationResult],
    repo_id: str,
    *,
    token: str | None = None,
    commit_message: str = "Upload QuantScenarioBench evaluation results",
) -> str:
    """Publish one or more EvaluationResults to a shared Hugging Face dataset repo (FR-33).

    Each result is uploaded as its own file under the repo's results/ layout
    (mirroring write_evaluation_result's local organization, Story 7.3); no
    previously published result is ever overwritten. The repo's README is
    regenerated after every publish to reflect the current, complete set of
    published results.

    Parameters
    ----------
    results:
        One or more EvaluationResults to publish.
    repo_id:
        Hugging Face Hub repository ID, e.g. ``"my-org/evaluation-results"``.
    token:
        HF Hub API token. Falls back to ``HF_TOKEN`` environment variable or
        a prior ``huggingface-cli login`` session if ``None``.
    commit_message:
        Commit message recorded in the Hub repository.

    Returns
    -------
    str
        URL of the published Evaluation Results repo on the Hugging Face Hub.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    """
    try:
        import huggingface_hub as hf
    except ImportError as exc:
        raise ImportError(
            "publish_evaluation_results requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    api = hf.HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

    existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

    uploaded_paths_in_repo: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for result in results:
            local_path = write_evaluation_result(result, root=tmp_root)
            relative = local_path.relative_to(tmp_root)
            path_in_repo = f"{_RESULTS_PREFIX}/{relative.as_posix()}"

            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                token=token,
            )
            uploaded_paths_in_repo.append(path_in_repo)

    all_result_files = sorted(existing_files | set(uploaded_paths_in_repo))
    card_text = generate_evaluation_results_card(all_result_files)
    api.upload_file(
        path_or_fileobj=card_text.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token,
    )

    return f"https://huggingface.co/datasets/{repo_id}"
