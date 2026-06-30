"""
Story 1.5 — Black-Scholes Market Model

Covers all acceptance criteria from GitHub Issue #5.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — x64 enabled before any test

from quantscenariobench.interface import (
    MarketModel,
    QuantScenarioBenchValidationWarning,
    TimeGrid,
)
from quantscenariobench.models import BlackScholes


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


# ---------------------------------------------------------------------------
# AC 1: BlackScholes is an equinox.Module subclass of MarketModel (AD-1, AD-6)
# ---------------------------------------------------------------------------

def test_black_scholes_is_market_model_subclass():
    assert issubclass(BlackScholes, MarketModel)


def test_black_scholes_is_equinox_module_subclass():
    assert issubclass(BlackScholes, eqx.Module)


def test_black_scholes_instantiates_successfully():
    model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    assert isinstance(model, BlackScholes)
    assert isinstance(model, MarketModel)
    assert isinstance(model, eqx.Module)


# ---------------------------------------------------------------------------
# AC 2: _drift and _diffusion implement GBM dynamics in JAX (NFR-2)
# ---------------------------------------------------------------------------

def test_drift_is_mu_times_state():
    model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    state = jnp.array(50.0)
    result = model._drift(0.0, state)
    expected = 0.05 * 50.0
    assert jnp.allclose(result, expected)


def test_diffusion_is_sigma_times_state():
    model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    state = jnp.array(50.0)
    result = model._diffusion(0.0, state)
    expected = 0.2 * 50.0
    assert jnp.allclose(result, expected)


def test_drift_is_jax_traceable():
    """_drift must be JAX-traceable (no Python-level loops)."""
    model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    # jax.jit traces the function through abstract values; raises if not traceable
    jit_drift = jax.jit(lambda s: model._drift(0.0, s))
    result = jit_drift(jnp.array(100.0))
    assert jnp.isfinite(result)


def test_diffusion_is_jax_traceable():
    """_diffusion must be JAX-traceable (no Python-level loops)."""
    model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    jit_diff = jax.jit(lambda s: model._diffusion(0.0, s))
    result = jit_diff(jnp.array(100.0))
    assert jnp.isfinite(result)


def test_drift_scales_linearly_with_state():
    model = BlackScholes(mu=0.1, sigma=0.2, S0=100.0)
    s1, s2 = jnp.array(50.0), jnp.array(200.0)
    # GBM drift is linear in state
    assert jnp.allclose(model._drift(0.0, s2) / model._drift(0.0, s1), s2 / s1)


def test_diffusion_scales_linearly_with_state():
    model = BlackScholes(mu=0.05, sigma=0.3, S0=100.0)
    s1, s2 = jnp.array(50.0), jnp.array(200.0)
    assert jnp.allclose(model._diffusion(0.0, s2) / model._diffusion(0.0, s1), s2 / s1)


# ---------------------------------------------------------------------------
# AC 3: latent_state is explicitly empty for Black-Scholes (FR-7)
# ---------------------------------------------------------------------------

_MODEL = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
_TG = TimeGrid(jnp.linspace(0.0, 1.0, 13))
_N = 32
_SEED = 7


def test_latent_state_is_empty_for_black_scholes():
    scenario = _simulate(_MODEL, _TG, _N, _SEED)
    assert scenario.latent_state.size == 0, (
        f"Expected empty latent_state, got shape {scenario.latent_state.shape}"
    )


def test_latent_state_leading_axis_is_n_paths():
    scenario = _simulate(_MODEL, _TG, _N, _SEED)
    assert scenario.latent_state.shape[0] == _N


# ---------------------------------------------------------------------------
# AC 4 & 5: Closed-form correctness — log-return mean and std (FR-7, NFR-3, AD-10)
#
# Uses 252 Euler steps over T=1 to ensure Euler-Maruyama converges
# to the exact GBM log-normal distribution within the 3 SE tolerance.
# ---------------------------------------------------------------------------

_MU = 0.05
_SIGMA = 0.20
_S0 = 100.0
_T = 1.0
_N_PATHS = 10_000
_CORRECTNESS_SEED = 0
_CORRECTNESS_MODEL = BlackScholes(mu=_MU, sigma=_SIGMA, S0=_S0)
_CORRECTNESS_TG = TimeGrid(jnp.linspace(0.0, _T, 252))


def _run_correctness_scenario():
    return _simulate(_CORRECTNESS_MODEL, _CORRECTNESS_TG, _N_PATHS, _CORRECTNESS_SEED)


def test_log_return_mean_within_3_se_of_closed_form():
    scenario = _run_correctness_scenario()
    S_T = scenario.observation[:, -1]
    log_returns = jnp.log(S_T / _S0)

    sample_mean = float(jnp.mean(log_returns))
    sample_std = float(jnp.std(log_returns))

    closed_form_mean = (_MU - 0.5 * _SIGMA ** 2) * _T
    se = sample_std / jnp.sqrt(_N_PATHS)
    assert abs(sample_mean - closed_form_mean) < 3 * se, (
        f"mean(log(S_T/S_0))={sample_mean:.6f} not within 3 SE "
        f"of closed-form {closed_form_mean:.6f} (3 SE={3*se:.6f})"
    )


def test_log_return_std_within_3_se_of_closed_form():
    scenario = _run_correctness_scenario()
    S_T = scenario.observation[:, -1]
    log_returns = jnp.log(S_T / _S0)

    sample_std = float(jnp.std(log_returns))
    closed_form_std = _SIGMA * jnp.sqrt(_T)
    se_of_std = closed_form_std / jnp.sqrt(2 * _N_PATHS)
    assert abs(sample_std - closed_form_std) < 3 * se_of_std, (
        f"std(log(S_T/S_0))={sample_std:.6f} not within 3 SE "
        f"of closed-form {closed_form_std:.6f} (3 SE={3*se_of_std:.6f})"
    )


# ---------------------------------------------------------------------------
# AC 6: BlackScholes imports only from interface and equinox (AD-9)
# ---------------------------------------------------------------------------

def test_black_scholes_does_not_import_solver():
    src = Path(__file__).parent.parent / "quantscenariobench" / "models" / "_black_scholes.py"
    source = src.read_text()
    assert "quantscenariobench.solver" not in source
    assert "from ..solver" not in source


def test_black_scholes_does_not_import_api():
    src = Path(__file__).parent.parent / "quantscenariobench" / "models" / "_black_scholes.py"
    source = src.read_text()
    assert "quantscenariobench.api" not in source
    assert "from ..api" not in source


def test_black_scholes_does_not_import_diffrax():
    src = Path(__file__).parent.parent / "quantscenariobench" / "models" / "_black_scholes.py"
    source = src.read_text()
    assert "diffrax" not in source


def test_models_package_dependency_direction():
    pkg_root = Path(__file__).parent.parent / "quantscenariobench"
    violations: list[str] = []
    allowed = {"interface"}
    for py_file in (pkg_root / "models").rglob("*.py"):
        source = py_file.read_text()
        for m in re.finditer(r"from \.\.((\w+))", source):
            sub = m.group(1)
            if sub not in allowed:
                violations.append(
                    f"{py_file.name}: imports from ..{sub} (not allowed by AD-9)"
                )
    assert not violations, "\n".join(violations)


# ---------------------------------------------------------------------------
# AC 7: sigma < 0 emits QuantScenarioBenchValidationWarning (FR-6)
# ---------------------------------------------------------------------------

def test_negative_sigma_emits_validation_warning():
    with pytest.warns(QuantScenarioBenchValidationWarning):
        BlackScholes(mu=0.05, sigma=-0.1, S0=100.0)


def test_negative_sigma_warning_is_not_an_exception():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", QuantScenarioBenchValidationWarning)
            model = BlackScholes(mu=0.05, sigma=-0.1, S0=100.0)
        assert isinstance(model, BlackScholes)
    except Exception as exc:
        pytest.fail(f"BlackScholes with sigma < 0 must not raise; got {exc!r}")


def test_negative_sigma_simulate_still_completes():
    """simulate() must succeed even with sigma < 0 (FR-6)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuantScenarioBenchValidationWarning)
        model = BlackScholes(mu=0.05, sigma=-0.1, S0=100.0)
    tg = TimeGrid(jnp.linspace(0.0, 1.0, 13))
    scenario = _simulate(model, tg, 8, 42)
    from quantscenariobench.interface import Scenario
    assert isinstance(scenario, Scenario)


def test_positive_sigma_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", QuantScenarioBenchValidationWarning)
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0)  # should not raise


def test_zero_sigma_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", QuantScenarioBenchValidationWarning)
        BlackScholes(mu=0.05, sigma=0.0, S0=100.0)  # boundary: not negative, no warn


# ---------------------------------------------------------------------------
# AC 8: All codebase validation warnings use QuantScenarioBenchValidationWarning
#        — no bare UserWarning, no model-specific subclasses (FR-6 Consistency)
# ---------------------------------------------------------------------------

def test_no_bare_user_warning_in_models():
    pkg_root = Path(__file__).parent.parent / "quantscenariobench"
    violations: list[str] = []
    for py_file in (pkg_root / "models").rglob("*.py"):
        source = py_file.read_text()
        if re.search(r'warnings\.warn\(.*UserWarning', source, re.DOTALL):
            violations.append(str(py_file.name))
    assert not violations, (
        "Bare UserWarning in models (must use QuantScenarioBenchValidationWarning):\n"
        + "\n".join(violations)
    )


def test_no_model_specific_warning_subclasses():
    pkg_root = Path(__file__).parent.parent / "quantscenariobench"
    violations: list[str] = []
    for py_file in (pkg_root / "models").rglob("*.py"):
        source = py_file.read_text()
        # Any class that inherits from Warning variants other than
        # QuantScenarioBenchValidationWarning itself is a violation
        for m in re.finditer(
            r"class\s+(\w+)\s*\(.*Warning.*\)", source
        ):
            cls_name = m.group(1)
            if cls_name != "QuantScenarioBenchValidationWarning":
                violations.append(f"{py_file.name}: {cls_name}")
    assert not violations, (
        "Model-specific Warning subclasses found (FR-6 requires only "
        "QuantScenarioBenchValidationWarning):\n" + "\n".join(violations)
    )


def test_validation_warning_class_is_exactly_one():
    pkg_root = Path(__file__).parent.parent / "quantscenariobench"
    warning_defs: list[str] = []
    for py_file in pkg_root.rglob("*.py"):
        source = py_file.read_text()
        for m in re.finditer(r"class\s+(\w+Warning)\s*\(", source):
            warning_defs.append(f"{py_file.relative_to(pkg_root.parent)}: {m.group(1)}")
    # Exactly one warning class must exist, and it must be the canonical one
    assert len(warning_defs) == 1, (
        f"Expected exactly 1 warning class, found {len(warning_defs)}:\n"
        + "\n".join(warning_defs)
    )
    assert "QuantScenarioBenchValidationWarning" in warning_defs[0]


# ---------------------------------------------------------------------------
# Additional: initial_state() returns S0 as a JAX array
# ---------------------------------------------------------------------------

def test_initial_state_returns_jax_array():
    model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    y0 = model.initial_state()
    assert hasattr(y0, "shape")


def test_initial_state_value_matches_s0():
    model = BlackScholes(mu=0.05, sigma=0.2, S0=123.45)
    y0 = model.initial_state()
    assert jnp.allclose(y0, jnp.array(123.45))


def test_simulate_without_explicit_y0_uses_s0():
    model = BlackScholes(mu=0.05, sigma=0.2, S0=50.0)
    tg = TimeGrid(jnp.linspace(0.0, 1.0, 13))
    scenario = _simulate(model, tg, 8, 42)
    # All paths must start at S0
    assert jnp.allclose(scenario.observation[:, 0], 50.0)
