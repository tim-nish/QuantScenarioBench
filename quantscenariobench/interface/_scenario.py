from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx

from ._time_grid import TimeGrid


@dataclasses.dataclass
class Metadata:
    """Provenance record attached to every Scenario.

    AD-8 fixes the minimum guaranteed field set. All ten fields are
    required; no Market Model may omit or rename any of them.
    """

    seed: int
    prng_key_info: str
    model_name: str
    model_version: str
    parameters: Any          # The Market Model's own eqx.Module instance
    time_grid: TimeGrid
    n_paths: int
    library_version: str
    dataset_version: str
    generated_at: str


class Scenario(eqx.Module):
    """The object returned by simulate().

    observation and latent_state are dynamic (traced) pytree leaves.
    metadata is static pytree aux_data and never a traced leaf (AD-2).
    The top-level field set is identical across all Market Models (FR-2).
    """

    observation: Any
    latent_state: Any
    metadata: Metadata = eqx.field(static=True)
