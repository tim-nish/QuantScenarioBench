"""
Story 1.6 — State-Space Interface Conformance Suite

Covers all acceptance criteria from GitHub Issue #6.
"""
from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — x64 enabled before any test

from quantscenariobench.interface import TimeGrid, QuantScenarioBenchValidationWarning
from quantscenariobench.models import BlackScholes
from quantscenariobench.testing import (
    DummyModel,
    assert_market_model_conforms,
    assert_scenario_schema,
    assert_reproducible,
    assert_validation_behaviour,
)


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


_TG = TimeGrid(jnp.linspace(0.0, 1.0, 13))
_N = 8
_SEED = 0

_BLACK_SCHOLES = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
_DUMMY = DummyModel(alpha=0.02, sigma=0.15, S0=1.0)


# ---------------------------------------------------------------------------
# AC 1: testing imports only from interface and test tooling (AD-9)
# ---------------------------------------------------------------------------

def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def test_testing_does_not_import_models():
    violations: list[str] = []
    for py_file in (_pkg_root() / "testing").rglob("*.py"):
        source = py_file.read_text()
        if re.search(r"quantscenariobench\.models", source):
            violations.append(py_file.name)
        if re.search(r"from \.\.(models)", source):
            violations.append(py_file.name)
    assert not violations, (
        "testing must not import from models (AD-9): " + ", ".join(violations)
    )


def test_testing_does_not_import_solver():
    violations: list[str] = []
    for py_file in (_pkg_root() / "testing").rglob("*.py"):
        source = py_file.read_text()
        if re.search(r"quantscenariobench\.solver|from \.\.(solver)", source):
            violations.append(py_file.name)
    assert not violations, (
        "testing must not import from solver (AD-9): " + ", ".join(violations)
    )


def test_testing_does_not_import_export():
    violations: list[str] = []
    for py_file in (_pkg_root() / "testing").rglob("*.py"):
        source = py_file.read_text()
        if re.search(r"quantscenariobench\.export|from \.\.(export)", source):
            violations.append(py_file.name)
    assert not violations, (
        "testing must not import from export (AD-9): " + ", ".join(violations)
    )


def test_testing_does_not_import_api():
    violations: list[str] = []
    for py_file in (_pkg_root() / "testing").rglob("*.py"):
        source = py_file.read_text()
        if re.search(r"quantscenariobench\.api|from \.\.(api)", source):
            violations.append(py_file.name)
    assert not violations, (
        "testing must not import from api (AD-9): " + ", ".join(violations)
    )


def test_testing_does_not_import_diffrax():
    violations: list[str] = []
    for py_file in (_pkg_root() / "testing").rglob("*.py"):
        source = py_file.read_text()
        if "diffrax" in source:
            violations.append(py_file.name)
    assert not violations, (
        "testing must not import diffrax (AD-9): " + ", ".join(violations)
    )


# ---------------------------------------------------------------------------
# AC 2: conformance suite passes against BlackScholes (FR-10, FR-11)
# ---------------------------------------------------------------------------

def test_black_scholes_passes_full_conformance_suite():
    assert_market_model_conforms(
        _BLACK_SCHOLES, _TG, _N, _SEED, simulate_fn=_simulate
    )


def test_black_scholes_scenario_schema():
    scenario = _simulate(_BLACK_SCHOLES, _TG, _N, _SEED)
    assert_scenario_schema(scenario)


def test_black_scholes_reproducibility():
    s1 = _simulate(_BLACK_SCHOLES, _TG, _N, _SEED)
    s2 = _simulate(_BLACK_SCHOLES, _TG, _N, _SEED)
    assert_reproducible(s1, s2)


# ---------------------------------------------------------------------------
# AC 3: conformance suite passes against DummyModel; core unchanged (FR-10, FR-11)
# ---------------------------------------------------------------------------

def test_dummy_model_passes_full_conformance_suite():
    assert_market_model_conforms(
        _DUMMY, _TG, _N, _SEED, simulate_fn=_simulate
    )


def test_dummy_model_scenario_schema():
    scenario = _simulate(_DUMMY, _TG, _N, _SEED)
    assert_scenario_schema(scenario)


def test_dummy_model_reproducibility():
    s1 = _simulate(_DUMMY, _TG, _N, _SEED)
    s2 = _simulate(_DUMMY, _TG, _N, _SEED)
    assert_reproducible(s1, s2)


# ---------------------------------------------------------------------------
# AC 4: DummyModel is NOT in models package and not exported from non-testing
# ---------------------------------------------------------------------------

def test_dummy_model_not_in_models_package():
    import quantscenariobench.models as models_pkg
    assert not hasattr(models_pkg, "DummyModel"), (
        "DummyModel must not appear in quantscenariobench.models (FR-11)"
    )


def test_dummy_model_not_in_top_level_package():
    import quantscenariobench as qsb
    assert not hasattr(qsb, "DummyModel"), (
        "DummyModel must not be exported from the top-level package (FR-11)"
    )


def test_dummy_model_only_in_testing_source():
    pkg_root = _pkg_root()
    exposures: list[str] = []
    for sub in ("models", "api", "solver", "export", "interface"):
        for py_file in (pkg_root / sub).rglob("*.py"):
            if "DummyModel" in py_file.read_text():
                exposures.append(str(py_file.relative_to(pkg_root.parent)))
    assert not exposures, (
        "DummyModel must exist only inside quantscenariobench.testing (FR-11):\n"
        + "\n".join(exposures)
    )


# ---------------------------------------------------------------------------
# AC 5: reproducibility test in the suite (FR-4, NFR-1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [_BLACK_SCHOLES, _DUMMY], ids=["BlackScholes", "DummyModel"])
def test_reproducibility_parametrized(model):
    s1 = _simulate(model, _TG, _N, _SEED)
    s2 = _simulate(model, _TG, _N, _SEED)
    assert_reproducible(s1, s2)


# ---------------------------------------------------------------------------
# AC 6: Scenario shape test (FR-2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", [_BLACK_SCHOLES, _DUMMY], ids=["BlackScholes", "DummyModel"])
def test_scenario_always_has_three_fields(model):
    scenario = _simulate(model, _TG, _N, _SEED)
    assert_scenario_schema(scenario)


@pytest.mark.parametrize("model", [_BLACK_SCHOLES, _DUMMY], ids=["BlackScholes", "DummyModel"])
def test_latent_state_never_absent(model):
    scenario = _simulate(model, _TG, _N, _SEED)
    assert scenario.latent_state is not None
    assert hasattr(scenario.latent_state, "shape")


# ---------------------------------------------------------------------------
# AC 7: Validation behaviour test via conformance suite (FR-6)
# ---------------------------------------------------------------------------

def test_dummy_model_validation_behaviour():
    assert_validation_behaviour(
        lambda: DummyModel(alpha=0.02, sigma=-0.5, S0=1.0)
    )


def test_black_scholes_validation_behaviour():
    assert_validation_behaviour(
        lambda: BlackScholes(mu=0.05, sigma=-0.2, S0=100.0)
    )


def test_validation_warning_does_not_prevent_simulate():
    """simulate() must complete even when model emitted a warning (FR-6)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuantScenarioBenchValidationWarning)
        model = DummyModel(alpha=0.02, sigma=-0.5, S0=1.0)
    scenario = _simulate(model, _TG, _N, _SEED)
    assert_scenario_schema(scenario)


# ---------------------------------------------------------------------------
# Additional: DummyModel structural checks
# ---------------------------------------------------------------------------

def test_dummy_model_is_market_model_subclass():
    from quantscenariobench.interface import MarketModel
    import equinox as eqx
    assert issubclass(DummyModel, MarketModel)
    assert issubclass(DummyModel, eqx.Module)


def test_dummy_model_initial_state():
    model = DummyModel(alpha=0.0, sigma=0.1, S0=42.0)
    y0 = model.initial_state()
    assert jnp.allclose(y0, jnp.array(42.0))


def test_conformance_suite_is_importable_from_testing():
    from quantscenariobench.testing import (  # noqa: F401
        assert_market_model_conforms,
        assert_scenario_schema,
        assert_reproducible,
        assert_validation_behaviour,
        DummyModel,
    )
