"""
Story 3.1 — Parquet Export of Scenario Batches

Covers all acceptance criteria from GitHub Issue #9.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quantscenariobench  # noqa: F401 — enables x64 before any test

from quantscenariobench.export import export_parquet
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes, Heston


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

import jax.numpy as jnp

_TG = TimeGrid(jnp.linspace(0.0, 1.0, 13))   # 12 monthly steps
_N = 8
_SEED = 0

_BS_MODEL = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
_H_MODEL = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)

_METADATA_COLUMNS = {
    "seed", "prng_key_info", "model_name", "model_version",
    "parameters", "time_grid", "n_paths", "library_version",
    "dataset_version", "generated_at",
}
_DATA_COLUMNS = {"observation", "latent_state"}
_ALL_COLUMNS = _METADATA_COLUMNS | _DATA_COLUMNS


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def _bs_scenario():
    return _simulate(_BS_MODEL, _TG, _N, _SEED)


def _heston_scenario():
    return _simulate(_H_MODEL, _TG, _N, _SEED)


# ---------------------------------------------------------------------------
# AC 1: export source imports only from interface (AD-5, AD-9)
# ---------------------------------------------------------------------------

def test_export_does_not_import_models():
    for py in (_pkg_root() / "export").rglob("*.py"):
        source = py.read_text()
        assert "quantscenariobench.models" not in source, py
        assert "from ..models" not in source, py


def test_export_does_not_import_solver():
    for py in (_pkg_root() / "export").rglob("*.py"):
        source = py.read_text()
        assert "quantscenariobench.solver" not in source, py
        assert "from ..solver" not in source, py


def test_export_does_not_import_testing():
    for py in (_pkg_root() / "export").rglob("*.py"):
        source = py.read_text()
        assert "quantscenariobench.testing" not in source, py
        assert "from ..testing" not in source, py


def test_export_ad9_dependency_direction():
    """Covered by the existing test_interface AD-9 scan; extra guard here."""
    pkg_root = _pkg_root()
    allowed_qsb_imports = {"interface"}
    violations: list[str] = []
    for py in (pkg_root / "export").rglob("*.py"):
        source = py.read_text()
        imported = {
            m.group(1)
            for m in re.finditer(r"from \.\.((\w+))", source)
        }
        bad = imported - allowed_qsb_imports
        if bad:
            violations.append(f"{py.name}: imports {bad}")
    assert not violations, "\n".join(violations)


# ---------------------------------------------------------------------------
# AC 2: export_parquet writes Parquet with correct columns (AD-2, AD-5, FR-12)
# ---------------------------------------------------------------------------

def test_export_parquet_creates_file():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_export_parquet_has_expected_columns_blackscholes():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert set(table.schema.names) == _ALL_COLUMNS


def test_export_parquet_has_expected_columns_heston():
    s = _heston_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert set(table.schema.names) == _ALL_COLUMNS


def test_export_parquet_row_count_equals_n_paths():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert len(table) == _N


def test_export_parquet_multiple_scenarios_stacked():
    s1 = _bs_scenario()
    s2 = _simulate(_BS_MODEL, _TG, _N * 2, seed=1)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s1, s2], path)
    table = pq.read_table(path)
    assert len(table) == _N + _N * 2


def test_observation_column_type_is_list_of_float():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    obs_type = table.schema.field("observation").type
    assert pa.types.is_list(obs_type)
    assert pa.types.is_floating(obs_type.value_type)


def test_latent_state_column_type_is_list_of_float():
    s = _heston_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    lat_type = table.schema.field("latent_state").type
    assert pa.types.is_list(lat_type)
    assert pa.types.is_floating(lat_type.value_type)


# ---------------------------------------------------------------------------
# AC 3: BlackScholes and Heston share the same column names (FR-13)
# ---------------------------------------------------------------------------

def test_shared_column_schema_across_models():
    bs_s = _bs_scenario()
    h_s = _heston_scenario()
    with (
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f_bs,
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f_h,
    ):
        export_parquet([bs_s], f_bs.name)
        export_parquet([h_s], f_h.name)
        bs_cols = set(pq.read_table(f_bs.name).schema.names)
        h_cols = set(pq.read_table(f_h.name).schema.names)

    assert bs_cols == h_cols, (
        f"Column mismatch: BS-only={bs_cols - h_cols}, Heston-only={h_cols - bs_cols}"
    )


def test_column_names_differ_in_content_not_names():
    bs_s = _bs_scenario()
    h_s = _heston_scenario()
    with (
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f_bs,
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f_h,
    ):
        export_parquet([bs_s], f_bs.name)
        export_parquet([h_s], f_h.name)
        bs_table = pq.read_table(f_bs.name)
        h_table = pq.read_table(f_h.name)

    # latent_state differs in content (empty vs T-length) not in column presence
    bs_lat = bs_table["latent_state"][0].as_py()
    h_lat = h_table["latent_state"][0].as_py()
    assert bs_lat == []   # BlackScholes: empty latent state
    assert len(h_lat) == len(_TG)   # Heston: T-length variance path


def test_model_name_differs_between_models():
    bs_s = _bs_scenario()
    h_s = _heston_scenario()
    with (
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f_bs,
        tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f_h,
    ):
        export_parquet([bs_s], f_bs.name)
        export_parquet([h_s], f_h.name)
        bs_table = pq.read_table(f_bs.name)
        h_table = pq.read_table(f_h.name)

    assert bs_table["model_name"][0].as_py() == "BlackScholes"
    assert h_table["model_name"][0].as_py() == "Heston"


# ---------------------------------------------------------------------------
# AC 4: Round-trip observation and latent_state numerically identical (FR-12)
# ---------------------------------------------------------------------------

def test_round_trip_observation_via_pyarrow():
    s = _bs_scenario()
    orig_obs = np.array(s.observation)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    rows = [table["observation"][i].as_py() for i in range(len(table))]
    rt_obs = np.array(rows)
    np.testing.assert_array_equal(rt_obs, orig_obs)


def test_round_trip_latent_state_via_pyarrow_heston():
    s = _heston_scenario()
    orig_lat = np.array(s.latent_state)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    rows = [table["latent_state"][i].as_py() for i in range(len(table))]
    rt_lat = np.array(rows)
    np.testing.assert_array_equal(rt_lat, orig_lat)


def test_round_trip_observation_via_pandas():
    s = _bs_scenario()
    orig_obs = np.array(s.observation)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    df = pd.read_parquet(path)
    rt_obs = np.stack(df["observation"].tolist())
    np.testing.assert_array_equal(rt_obs, orig_obs)


def test_round_trip_latent_state_via_pandas_heston():
    s = _heston_scenario()
    orig_lat = np.array(s.latent_state)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    df = pd.read_parquet(path)
    rt_lat = np.stack(df["latent_state"].tolist())
    np.testing.assert_array_equal(rt_lat, orig_lat)


def test_round_trip_empty_latent_state_blackscholes():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    for i in range(len(table)):
        assert table["latent_state"][i].as_py() == []


# ---------------------------------------------------------------------------
# AC 5: All ten Metadata fields appear as columns (AD-8, FR-15)
# ---------------------------------------------------------------------------

def test_all_ten_metadata_fields_present():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    col_names = set(table.schema.names)
    for field in _METADATA_COLUMNS:
        assert field in col_names, f"Missing metadata column: {field}"


def test_seed_column_has_correct_value():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert table["seed"][0].as_py() == _SEED


def test_model_name_column_value():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert table["model_name"][0].as_py() == "BlackScholes"


def test_n_paths_column_value():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert table["n_paths"][0].as_py() == _N


def test_parameters_column_is_parseable_json():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    params_str = table["parameters"][0].as_py()
    parsed = json.loads(params_str)
    assert "mu" in parsed
    assert "sigma" in parsed
    assert "S0" in parsed


def test_time_grid_column_is_parseable_json():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    tg_str = table["time_grid"][0].as_py()
    parsed = json.loads(tg_str)
    assert isinstance(parsed, list)
    assert len(parsed) == len(_TG)
    np.testing.assert_allclose(parsed, np.array(_TG.t).tolist())


def test_generated_at_is_nonempty_string():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    assert table["generated_at"][0].as_py() != ""


def test_metadata_repeated_for_each_path():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    seeds = table["seed"].to_pylist()
    assert all(v == _SEED for v in seeds), "seed must be same for all rows in one Scenario"


# ---------------------------------------------------------------------------
# Additional / structural checks
# ---------------------------------------------------------------------------

def test_export_parquet_is_importable():
    from quantscenariobench.export import export_parquet  # noqa: F401


def test_observation_list_length_equals_time_grid():
    s = _bs_scenario()
    T = len(_TG)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    for i in range(len(table)):
        assert len(table["observation"][i].as_py()) == T


def test_heston_latent_state_list_length_equals_time_grid():
    s = _heston_scenario()
    T = len(_TG)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    table = pq.read_table(path)
    for i in range(len(table)):
        assert len(table["latent_state"][i].as_py()) == T


def test_empty_scenarios_list_writes_empty_parquet():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([], path)
    table = pq.read_table(path)
    assert len(table) == 0
    assert set(table.schema.names) == _ALL_COLUMNS


def test_export_path_accepts_pathlib_path():
    s = _bs_scenario()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.parquet"
        export_parquet([s], path)
        assert path.exists()


def test_export_path_accepts_string():
    s = _bs_scenario()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    export_parquet([s], path)
    assert Path(path).exists()
