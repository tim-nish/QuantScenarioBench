from __future__ import annotations

import dataclasses
import datetime
import json
import uuid
from pathlib import Path

from ._evaluation_result import EvaluationResult

_DEFAULT_ROOT = "results"


def write_evaluation_result(
    result: EvaluationResult, root: str | Path = _DEFAULT_ROOT
) -> Path:
    """Write result to a local, organized, append-only file layout (FR-32).

    Saved as its own timestamped JSON file under
    ``<root>/<benchmark_dataset.time_grid_reference>/<strategy.name>/``.
    The timestamp is suffixed with a short random token so that two
    writes for the same Benchmark Dataset/strategy combination never
    collide, even if they land in the same microsecond — no write ever
    overwrites a prior one. This directory tree is exactly the layout
    Story 7.4 uploads to the Hub unchanged, with no reorganization.
    """
    directory = Path(root) / result.benchmark_dataset.time_grid_reference / result.strategy.name
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    unique_suffix = uuid.uuid4().hex[:8]
    path = directory / f"result_{timestamp}_{unique_suffix}.json"

    path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    return path
