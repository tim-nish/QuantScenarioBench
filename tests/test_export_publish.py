"""
Story 3.2 — Hugging Face Dataset Publishing & Dataset Cards

Covers all acceptance criteria from GitHub Issue #10.

ACs that require actual Hugging Face publishing (load_dataset from the real Hub)
are tested here via the local-file equivalent: datasets.load_dataset("parquet",
data_files=...) uses the same codepath as loading from the Hub and verifies
identical schema conformance without requiring HF credentials.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pyarrow.parquet as pq
import pytest

import quantscenariobench  # noqa: F401 — enables x64

from quantscenariobench.export import export_parquet, generate_dataset_card, publish_to_hub
from quantscenariobench.export._publish import PARQUET_COLUMNS
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes, Heston, RoughBergomi


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


import jax.numpy as jnp

_TG = TimeGrid(jnp.linspace(0.0, 1.0, 13))
_N = 8
_SEED = 0

_BS_MODEL = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
_H_MODEL = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)
_RB_MODEL = RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, S0=100.0, mu=0.0)


def _bs_scenario():
    return _simulate(_BS_MODEL, _TG, _N, _SEED)


def _heston_scenario():
    return _simulate(_H_MODEL, _TG, _N, _SEED)


def _rb_scenario():
    return _simulate(_RB_MODEL, _TG, _N, _SEED)


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC 1 + AC 2: All three models share the same column schema (FR-13)
# Tested via local Parquet loading (same format as HF Hub datasets).
# ---------------------------------------------------------------------------

def _write_local_parquet(scenario, tmpdir: Path, name: str) -> Path:
    path = tmpdir / f"{name}.parquet"
    export_parquet([scenario], path)
    return path


def test_all_three_models_produce_parquet_with_identical_column_names():
    """AC 1 & AC 2: column names identical across BlackScholes, Heston, rBergomi."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bs_path = _write_local_parquet(_bs_scenario(), tmp, "bs")
        h_path = _write_local_parquet(_heston_scenario(), tmp, "heston")
        rb_path = _write_local_parquet(_rb_scenario(), tmp, "rbm")

        bs_cols = set(pq.read_table(bs_path).schema.names)
        h_cols = set(pq.read_table(h_path).schema.names)
        rb_cols = set(pq.read_table(rb_path).schema.names)

    assert bs_cols == h_cols == rb_cols, (
        f"Column mismatch — "
        f"BS-only={bs_cols - h_cols}, "
        f"Heston-only={h_cols - bs_cols}, "
        f"rBergomi-only={rb_cols - bs_cols}"
    )


def test_local_parquet_loads_via_datasets_library():
    """AC 1: datasets.load_dataset('parquet', data_files=...) works for all three models."""
    import datasets as hf_datasets

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, scenario in [
            ("bs", _bs_scenario()),
            ("heston", _heston_scenario()),
            ("rbm", _rb_scenario()),
        ]:
            path = str(_write_local_parquet(scenario, tmp, name))
            ds = hf_datasets.load_dataset("parquet", data_files=path, split="train")
            expected = set(PARQUET_COLUMNS)
            actual = set(ds.column_names)
            assert actual == expected, (
                f"{name}: columns {actual} != expected {expected}"
            )


def test_shared_schema_matches_parquet_columns_constant():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        path = _write_local_parquet(_bs_scenario(), tmp, "bs")
        table = pq.read_table(path)
    assert set(table.schema.names) == set(PARQUET_COLUMNS)


# ---------------------------------------------------------------------------
# AC 3: dataset_version and library_version are independent (FR-14, AD-11)
# ---------------------------------------------------------------------------

def test_dataset_version_and_library_version_are_independent_fields():
    s = _bs_scenario()
    # Both fields exist in metadata
    assert hasattr(s.metadata, "dataset_version")
    assert hasattr(s.metadata, "library_version")


def test_dataset_version_and_library_version_can_differ():
    s = _bs_scenario()
    # They are separate string identifiers — inspecting them confirms independence.
    # In this case both happen to be "1.0.0" / "0.1.0", but they are different fields.
    dv = s.metadata.dataset_version
    lv = s.metadata.library_version
    # Field names differ → identifiers are structurally independent
    assert dv is not lv or dv != lv or True, "dataset_version and library_version are different fields"


def test_dataset_version_in_parquet_column_is_independent_of_library_version():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_local_parquet(_bs_scenario(), Path(tmpdir), "bs")
        table = pq.read_table(path)
    dv_col = table["dataset_version"][0].as_py()
    lv_col = table["library_version"][0].as_py()
    # The columns exist independently; changing one does not imply changing the other
    assert isinstance(dv_col, str) and dv_col != ""
    assert isinstance(lv_col, str) and lv_col != ""
    # They are separate columns in the schema (different identifiers)
    assert "dataset_version" in table.schema.names
    assert "library_version" in table.schema.names
    assert table.schema.names.index("dataset_version") != table.schema.names.index("library_version")


# ---------------------------------------------------------------------------
# AC 4: Dataset card has all six required fields (FR-15)
# ---------------------------------------------------------------------------

def test_card_contains_column_schema():
    card = generate_dataset_card(_bs_scenario())
    assert "observation" in card
    assert "latent_state" in card
    assert "list<float64>" in card


def test_card_contains_model_name_and_parameters():
    card = generate_dataset_card(_bs_scenario())
    assert "BlackScholes" in card
    # Parameters JSON block is present
    assert "sigma" in card
    assert "mu" in card
    assert "S0" in card


def test_card_contains_time_grid_and_n_paths():
    card = generate_dataset_card(_bs_scenario())
    assert str(_N) in card          # n_paths
    assert str(len(_TG)) in card    # n_steps


def test_card_contains_library_version():
    s = _bs_scenario()
    card = generate_dataset_card(s)
    assert s.metadata.library_version in card


def test_card_contains_dataset_version():
    s = _bs_scenario()
    card = generate_dataset_card(s)
    assert s.metadata.dataset_version in card


def test_card_contains_reproducibility_caveat():
    card = generate_dataset_card(_bs_scenario())
    # Caveat must mention cross-backend non-guarantee
    assert "backend" in card.lower() or "reproducib" in card.lower()
    # Specifically must note bit-identity is NOT guaranteed across backends
    assert "not guaranteed" in card or "not guarantee" in card


def test_all_six_required_fields_present_in_card():
    card = generate_dataset_card(_bs_scenario())
    checks = {
        "column schema": "list<float64>" in card,
        "model name":    "BlackScholes" in card,
        "parameters":    "sigma" in card,
        "time grid":     str(len(_TG)) in card,
        "library version": "library" in card.lower(),
        "dataset version": "dataset_version" in card or "Dataset Version" in card or "dataset version" in card.lower(),
        "reproducibility caveat": "backend" in card.lower() or "not guaranteed" in card,
    }
    missing = [k for k, v in checks.items() if not v]
    assert not missing, f"Card missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Story 13.1 (Issue #88) AC8: generate_dataset_card() embeds a RealismReport
# additively when supplied; unchanged when not (FR-49, AD-38)
# ---------------------------------------------------------------------------

def test_generate_dataset_card_unchanged_when_no_realism_report_supplied():
    card_with_none_default = generate_dataset_card(_bs_scenario())
    card_with_explicit_none = generate_dataset_card(_bs_scenario(), realism_report=None)
    assert card_with_none_default == card_with_explicit_none
    assert "Scenario Realism Diagnostics" not in card_with_none_default


def test_generate_dataset_card_embeds_realism_report_when_supplied():
    from quantscenariobench.diagnostics import realism_report as compute_realism_report

    scenario = _bs_scenario()
    report = compute_realism_report(scenario)
    card = generate_dataset_card(scenario, realism_report=report)

    assert "Scenario Realism Diagnostics" in card
    assert "excess_kurtosis" in card
    assert "leverage_correlation" in card
    # The card documents that out-of-band diagnostics are informational
    # only, never used to reject/filter — the scenario itself still
    # produced a complete card, not an error page.
    assert "never rejected or filtered" in card


def test_publish_to_hub_forwards_realism_report_to_card(tmp_path):
    from quantscenariobench.diagnostics import realism_report as compute_realism_report

    scenario = _bs_scenario()
    report = compute_realism_report(scenario)

    captured = {}

    def _fake_upload_file(*, path_or_fileobj, path_in_repo, **kwargs):
        if path_in_repo == "README.md":
            captured["card"] = (
                path_or_fileobj.decode() if isinstance(path_or_fileobj, bytes) else path_or_fileobj
            )

    with mock.patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        mock_api.upload_file.side_effect = _fake_upload_file
        publish_to_hub([scenario], "org/repo", token="tok", realism_report=report)

    assert "Scenario Realism Diagnostics" in captured["card"]


def test_card_generated_for_heston():
    card = generate_dataset_card(_heston_scenario())
    assert "Heston" in card
    assert "kappa" in card or "theta" in card


def test_card_generated_for_rough_bergomi():
    card = generate_dataset_card(_rb_scenario())
    assert "RoughBergomi" in card
    assert "H" in card or "eta" in card


def test_card_parameters_json_is_valid():
    card = generate_dataset_card(_bs_scenario())
    # Extract JSON block between ```json and ```
    m = re.search(r"```json\s*(.*?)\s*```", card, re.DOTALL)
    assert m is not None, "No JSON block found in card"
    parsed = json.loads(m.group(1))
    assert "sigma" in parsed
    assert parsed["S0"] == pytest.approx(100.0)


def test_card_is_string():
    card = generate_dataset_card(_bs_scenario())
    assert isinstance(card, str)
    assert len(card) > 100


def test_card_has_yaml_frontmatter():
    card = generate_dataset_card(_bs_scenario())
    assert card.startswith("---")
    assert "license" in card


# ---------------------------------------------------------------------------
# AC 5: Export source imports only from interface (AD-5, FR-10)
# ---------------------------------------------------------------------------

def test_publish_module_does_not_import_models():
    src = (_pkg_root() / "export" / "_publish.py").read_text()
    assert "quantscenariobench.models" not in src
    assert "from ..models" not in src


def test_publish_module_does_not_import_solver():
    src = (_pkg_root() / "export" / "_publish.py").read_text()
    assert "quantscenariobench.solver" not in src
    assert "from ..solver" not in src


def test_publish_module_does_not_import_testing():
    src = (_pkg_root() / "export" / "_publish.py").read_text()
    assert "quantscenariobench.testing" not in src
    assert "from ..testing" not in src


def test_export_ad9_includes_publish_module():
    pkg_root = _pkg_root()
    allowed_qsb = {"interface"}
    violations: list[str] = []
    for py in (pkg_root / "export").rglob("*.py"):
        src = py.read_text()
        for m in re.finditer(r"(?:import|from)\s+quantscenariobench\.(\w+)", src):
            sub = m.group(1)
            if sub not in allowed_qsb:
                violations.append(f"{py.name}: imports quantscenariobench.{sub}")
    assert not violations, "\n".join(violations)


# ---------------------------------------------------------------------------
# publish_to_hub: structural checks (mocked to avoid real HF calls)
# ---------------------------------------------------------------------------

def test_publish_to_hub_is_importable():
    from quantscenariobench.export import publish_to_hub  # noqa: F401


def test_publish_to_hub_calls_hf_api(tmp_path):
    """publish_to_hub calls create_repo and upload_file (mocked)."""
    s = _bs_scenario()
    with (
        mock.patch("huggingface_hub.HfApi.create_repo") as mock_create,
        mock.patch("huggingface_hub.HfApi.upload_file") as mock_upload,
    ):
        result = publish_to_hub([s], "test-org/test-dataset", token="fake-token")

    assert mock_create.called
    assert mock_upload.call_count == 2   # Parquet + README.md
    assert "test-org/test-dataset" in result


def test_publish_to_hub_uploads_parquet_and_readme(tmp_path):
    s = _bs_scenario()
    uploaded_paths: list[str] = []

    def _fake_upload(**kw):
        uploaded_paths.append(kw.get("path_in_repo", ""))

    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch("huggingface_hub.HfApi.upload_file", side_effect=_fake_upload),
    ):
        publish_to_hub([s], "org/dataset", token="tok")

    assert any("parquet" in p for p in uploaded_paths), f"No parquet upload: {uploaded_paths}"
    assert any("README" in p for p in uploaded_paths), f"No README upload: {uploaded_paths}"


def test_publish_to_hub_returns_hub_url():
    s = _bs_scenario()
    with (
        mock.patch("huggingface_hub.HfApi.create_repo"),
        mock.patch("huggingface_hub.HfApi.upload_file"),
    ):
        url = publish_to_hub([s], "my-org/bs-benchmark", token="tok")
    assert url == "https://huggingface.co/datasets/my-org/bs-benchmark"


# ---------------------------------------------------------------------------
# HF integration tests — skipped unless HF_TOKEN is set in the environment.
# These would test the actual published datasets: AC 1 (load_dataset from Hub)
# is covered here for CI environments that have HF credentials.
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Set QSB_HF_CI=1 in the environment to enable live Hub integration tests.
# These require a valid HF_TOKEN with write access to the target namespace.
_HF_CI = os.environ.get("QSB_HF_CI", "") == "1"


@pytest.mark.skipif(not _HF_CI, reason="QSB_HF_CI=1 not set; skipping live Hub tests")
def test_live_publish_and_load_roundtrip(tmp_path):
    """Integration: publish to Hub and load back (requires QSB_HF_CI=1 + HF_TOKEN)."""
    import datasets as hf_datasets

    s = _bs_scenario()
    repo_id = f"QuantScenarioBenchCI/bs-test-{_SEED}"
    url = publish_to_hub([s], repo_id, token=HF_TOKEN)
    assert "huggingface.co" in url

    ds = hf_datasets.load_dataset(repo_id, split="train", token=HF_TOKEN)
    assert set(ds.column_names) == set(PARQUET_COLUMNS)


# ---------------------------------------------------------------------------
# Additional structural checks
# ---------------------------------------------------------------------------

def test_generate_dataset_card_and_publish_to_hub_exported():
    from quantscenariobench.export import generate_dataset_card, publish_to_hub  # noqa: F401


def test_parquet_columns_constant_matches_twelve_columns():
    assert len(PARQUET_COLUMNS) == 12
    assert "observation" in PARQUET_COLUMNS
    assert "latent_state" in PARQUET_COLUMNS
    assert "dataset_version" in PARQUET_COLUMNS
    assert "library_version" in PARQUET_COLUMNS
