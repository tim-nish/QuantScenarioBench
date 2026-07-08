"""Parquet export of Scenario batches (FR-12, FR-13, FR-15, AD-2, AD-5, AD-9).

Generic over the Scenario schema — no concrete Market Model is ever imported.
Columns are derived by pytree-flattening the dynamic fields (observation,
latent_state) of each Scenario (AD-2, AD-5).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..interface import BasketMetadata, Scenario

# The ten Metadata fields mandated by AD-8.
_METADATA_FIELDS = (
    "seed",
    "prng_key_info",
    "model_name",
    "model_version",
    "parameters",
    "time_grid",
    "n_paths",
    "library_version",
    "dataset_version",
    "generated_at",
)


def _serialize_parameters(params: object) -> str:
    """JSON-encode a MarketModel eqx.Module's scalar parameters."""
    try:
        field_items = {
            f.name: getattr(params, f.name)
            for f in dataclasses.fields(params)  # type: ignore[arg-type]
        }
    except TypeError:
        field_items = vars(params)
    serializable: dict = {}
    for k, v in field_items.items():
        if hasattr(v, "item"):
            serializable[k] = v.item()
        elif hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    return json.dumps(serializable)


def _serialize_time_grid(tg: object) -> str:
    """JSON-encode a TimeGrid's time-point array."""
    return json.dumps(np.array(tg.t).tolist())  # type: ignore[attr-defined]


def _serialize_rho(rho: object) -> str:
    """JSON-encode a basket correlation matrix."""
    return json.dumps(np.asarray(rho).tolist())


def export_parquet(
    scenarios: Sequence[Scenario],
    path: str | Path,
    basket_metadata: Optional[BasketMetadata] = None,
) -> None:
    """Export a batch of Scenarios to a single Parquet file.

    One row per simulation path. Columns:

    - ``observation``: ``list<float64>`` — the asset-price (or observation) path
    - ``latent_state``: ``list<float64>`` — the latent-state path; empty list
      for models with no latent process (e.g. Black-Scholes)
    - Ten Metadata columns (AD-8): ``seed``, ``prng_key_info``, ``model_name``,
      ``model_version``, ``parameters``, ``time_grid``, ``n_paths``,
      ``library_version``, ``dataset_version``, ``generated_at``

    The column set is identical across all v1 Market Models (FR-13) — only the
    content of ``latent_state`` and ``parameters`` differs between models.

    basket_metadata is additive (FR-47, AD-36): when supplied (a
    quantscenariobench.interface.BasketMetadata from
    simulate_correlated_basket()), three further columns are added —
    ``basket_rho`` (JSON-encoded correlation matrix), ``basket_seed``, and
    ``basket_constituents`` (JSON-encoded constituent identifier list) —
    the same basket-wide value repeated on every row, mirroring how
    per-scenario Metadata is already repeated per path. When
    basket_metadata is None (the default), the exported column set is
    byte-for-byte unchanged from before this parameter existed.

    Parameters
    ----------
    scenarios:
        One or more :class:`~quantscenariobench.interface.Scenario` objects,
        as returned by :func:`~quantscenariobench.api.simulate` or
        :func:`~quantscenariobench.api.simulate_correlated_basket`.
    path:
        Destination file path for the Parquet output.
    basket_metadata:
        Optional basket-level provenance record to embed additively.
    """
    obs_rows: list[list[float]] = []
    lat_rows: list[list[float]] = []
    meta_cols: dict[str, list] = {f: [] for f in _METADATA_FIELDS}
    basket_cols: dict[str, list] = {
        "basket_rho": [], "basket_seed": [], "basket_constituents": [],
    }

    if basket_metadata is not None:
        basket_rho_json = _serialize_rho(basket_metadata.rho)
        basket_constituents_json = json.dumps(basket_metadata.constituents)

    for scenario in scenarios:
        obs = np.array(scenario.observation)    # (n_paths, T)
        lat = np.array(scenario.latent_state)   # (n_paths, T') or (n_paths, 0)
        meta = scenario.metadata
        n = obs.shape[0]

        obs_rows.extend(row.tolist() for row in obs)
        lat_rows.extend(row.tolist() for row in lat)

        params_json = _serialize_parameters(meta.parameters)
        tg_json = _serialize_time_grid(meta.time_grid)

        for _ in range(n):
            meta_cols["seed"].append(meta.seed)
            meta_cols["prng_key_info"].append(meta.prng_key_info)
            meta_cols["model_name"].append(meta.model_name)
            meta_cols["model_version"].append(meta.model_version)
            meta_cols["parameters"].append(params_json)
            meta_cols["time_grid"].append(tg_json)
            meta_cols["n_paths"].append(meta.n_paths)
            meta_cols["library_version"].append(meta.library_version)
            meta_cols["dataset_version"].append(meta.dataset_version)
            meta_cols["generated_at"].append(meta.generated_at)
            if basket_metadata is not None:
                basket_cols["basket_rho"].append(basket_rho_json)
                basket_cols["basket_seed"].append(basket_metadata.basket_seed)
                basket_cols["basket_constituents"].append(basket_constituents_json)

    columns = {
        "observation": pa.array(obs_rows, type=pa.list_(pa.float64())),
        "latent_state": pa.array(lat_rows, type=pa.list_(pa.float64())),
        "seed": pa.array(meta_cols["seed"], type=pa.int64()),
        "prng_key_info": pa.array(meta_cols["prng_key_info"], type=pa.string()),
        "model_name": pa.array(meta_cols["model_name"], type=pa.string()),
        "model_version": pa.array(meta_cols["model_version"], type=pa.string()),
        "parameters": pa.array(meta_cols["parameters"], type=pa.string()),
        "time_grid": pa.array(meta_cols["time_grid"], type=pa.string()),
        "n_paths": pa.array(meta_cols["n_paths"], type=pa.int64()),
        "library_version": pa.array(meta_cols["library_version"], type=pa.string()),
        "dataset_version": pa.array(meta_cols["dataset_version"], type=pa.string()),
        "generated_at": pa.array(meta_cols["generated_at"], type=pa.string()),
    }
    if basket_metadata is not None:
        columns["basket_rho"] = pa.array(basket_cols["basket_rho"], type=pa.string())
        columns["basket_seed"] = pa.array(basket_cols["basket_seed"], type=pa.int64())
        columns["basket_constituents"] = pa.array(
            basket_cols["basket_constituents"], type=pa.string()
        )

    table = pa.table(columns)
    pq.write_table(table, Path(path))
