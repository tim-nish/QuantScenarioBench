"""
Story 2.2 — Rough Bergomi Model

Covers all acceptance criteria from GitHub Issue #8.
"""
from __future__ import annotations

import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quantscenariobench  # noqa: F401 — enables x64 before any test

from quantscenariobench.interface import MarketModel, TimeGrid
from quantscenariobench.models import BlackScholes, RoughBergomi
from quantscenariobench.testing import assert_market_model_conforms


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_H = 0.1
_ETA = 1.5
_RHO = -0.7
_XI0 = 0.04
_S0 = 100.0
_MU = 0.0

_MODEL = RoughBergomi(H=_H, eta=_ETA, rho=_RHO, xi0=_XI0, S0=_S0, mu=_MU)
_TG = TimeGrid(jnp.linspace(0.0, 1.0, 53))   # ~weekly, fast
_N = 16
_SEED = 42


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC 1: RoughBergomi is equinox.Module subclass of MarketModel (AD-1, AD-6)
# ---------------------------------------------------------------------------

def test_rough_bergomi_is_market_model_subclass():
    assert issubclass(RoughBergomi, MarketModel)


def test_rough_bergomi_is_equinox_module():
    assert issubclass(RoughBergomi, eqx.Module)


def test_rough_bergomi_instantiates():
    m = RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, S0=100.0, mu=0.0)
    assert isinstance(m, RoughBergomi)


# ---------------------------------------------------------------------------
# AC 2: _drift/_diffusion implement rBergomi dynamics in JAX (NFR-2, FR-9)
# ---------------------------------------------------------------------------

def test_drift_shape():
    state = jnp.array([100.0, 0.04])
    d = _MODEL._drift(0.5, state)
    assert d.shape == (2,)


def test_drift_asset_is_mu_times_S():
    state = jnp.array([100.0, 0.04])
    d = _MODEL._drift(0.5, state)
    assert jnp.allclose(d[0], _MU * 100.0)


def test_drift_variance_component_is_zero():
    # Variance has zero drift in the local Itô sense
    state = jnp.array([100.0, 0.04])
    d = _MODEL._drift(0.5, state)
    assert jnp.allclose(d[1], 0.0)


def test_diffusion_shape_is_2x2():
    state = jnp.array([100.0, 0.04])
    sigma = _MODEL._diffusion(0.5, state)
    assert sigma.shape == (2, 2)


def test_diffusion_variance_row_references_H_exponent():
    state = jnp.array([100.0, 0.04])
    sigma = _MODEL._diffusion(0.5, state)
    # Row 1 encodes the Volterra kernel K(t, 0) = t^{H - 1/2}
    k_t = 0.5 ** (_H - 0.5)
    expected_v_col0 = _ETA * 0.04 * k_t
    assert jnp.allclose(sigma[1, 0], expected_v_col0, rtol=1e-5)


def test_diffusion_encodes_rho_for_asset():
    state = jnp.array([100.0, 0.04])
    sigma = _MODEL._diffusion(0.5, state)
    sv = float(jnp.sqrt(0.04))
    # Row 0, col 0: sigma * S * rho
    assert jnp.allclose(sigma[0, 0], sv * 100.0 * _RHO, rtol=1e-5)


def test_drift_is_jax_traceable():
    fn = jax.jit(lambda s: _MODEL._drift(0.5, s))
    r = fn(jnp.array([100.0, 0.04]))
    assert r.shape == (2,)
    assert jnp.all(jnp.isfinite(r))


def test_diffusion_is_jax_traceable():
    fn = jax.jit(lambda s: _MODEL._diffusion(0.5, s))
    r = fn(jnp.array([100.0, 0.04]))
    assert r.shape == (2, 2)
    assert jnp.all(jnp.isfinite(r))


def test_model_source_references_volterra_kernel():
    src = (_pkg_root() / "models" / "_rough_bergomi.py").read_text()
    # Source must mention H - 0.5 (Volterra exponent) explicitly
    assert "H - 0.5" in src or "H-0.5" in src or "H - 1/2" in src or "H-1/2" in src


def test_model_source_references_fractional_BM():
    src = (_pkg_root() / "models" / "_rough_bergomi.py").read_text()
    # Source must reference the fBM / Volterra mechanism
    assert "Volterra" in src or "fBM" in src or "fractional" in src.lower()


# ---------------------------------------------------------------------------
# AC 3: latent_state shape is (n_paths, len(time_grid)) (FR-9)
# ---------------------------------------------------------------------------

def test_latent_state_shape():
    s = _simulate(_MODEL, _TG, _N, _SEED)
    T = len(_TG)
    assert s.latent_state.shape == (_N, T)


def test_observation_shape():
    s = _simulate(_MODEL, _TG, _N, _SEED)
    T = len(_TG)
    assert s.observation.shape == (_N, T)


def test_observation_starts_at_S0():
    s = _simulate(_MODEL, _TG, _N, _SEED)
    assert jnp.allclose(s.observation[:, 0], _S0, atol=1e-6)


def test_latent_state_starts_at_xi0():
    # V_0 = xi0 * exp(eta * 0 - 0.5 * eta^2 * 0) = xi0
    s = _simulate(_MODEL, _TG, _N, _SEED)
    assert jnp.allclose(s.latent_state[:, 0], _XI0, atol=1e-6)


def test_latent_state_is_positive():
    s = _simulate(_MODEL, _TG, _N, _SEED)
    # Variance process should stay positive (xi0 * exp(...) > 0 always)
    assert jnp.all(s.latent_state > 0)


# ---------------------------------------------------------------------------
# AC 4: Identical Scenario field names as BlackScholes; simulate() unchanged (FR-2, FR-10)
# ---------------------------------------------------------------------------

def test_scenario_field_names_identical_to_black_scholes():
    bs = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    bs_s = _simulate(bs, _TG, _N, _SEED)
    rb_s = _simulate(_MODEL, _TG, _N, _SEED)
    for field in ("observation", "latent_state", "metadata"):
        assert hasattr(bs_s, field)
        assert hasattr(rb_s, field)


def test_simulate_source_not_model_specific():
    import inspect
    from quantscenariobench.api import simulate
    src = inspect.getsource(simulate)
    assert "RoughBergomi" not in src
    assert "Heston" not in src
    assert "BlackScholes" not in src
    assert "quantscenariobench.models" not in src


# ---------------------------------------------------------------------------
# AC 5: H=0.5 Markovian limit test — mean and variance of log(S_T/S0) (NFR-3, AD-10)
#
# At H=0.5, eta=0, rho=0, mu=0: V_t = xi0 (constant) → GBM with sigma^2=xi0.
# log(S_T/S0) ~ N(-0.5*xi0*T, xi0*T).
# ---------------------------------------------------------------------------

_MARKOV_N = 10_000
_MARKOV_T = 1.0
_MARKOV_XI0 = 0.04    # constant variance when eta=0
_MARKOV_MODEL = RoughBergomi(H=0.5, eta=0.0, rho=0.0, xi0=_MARKOV_XI0, S0=100.0, mu=0.0)
_MARKOV_TG = TimeGrid(jnp.linspace(0.0, _MARKOV_T, 253))   # daily


def test_markovian_limit_log_return_mean():
    s = _simulate(_MARKOV_MODEL, _MARKOV_TG, _MARKOV_N, seed=0)
    log_ret = jnp.log(s.observation[:, -1] / 100.0)
    mc_mean = float(jnp.mean(log_ret))
    mc_se = float(jnp.std(log_ret)) / (_MARKOV_N ** 0.5)

    expected_mean = -0.5 * _MARKOV_XI0 * _MARKOV_T   # = -0.02
    assert abs(mc_mean - expected_mean) < 3.0 * mc_se, (
        f"MC mean {mc_mean:.5f} vs expected {expected_mean:.5f}, 3*SE={3*mc_se:.5f}"
    )


def test_markovian_limit_log_return_variance():
    s = _simulate(_MARKOV_MODEL, _MARKOV_TG, _MARKOV_N, seed=0)
    log_ret = jnp.log(s.observation[:, -1] / 100.0)
    mc_var = float(jnp.var(log_ret))

    expected_var = _MARKOV_XI0 * _MARKOV_T   # = 0.04
    # SE of sample variance ≈ var * sqrt(2 / (N-1))
    se_var = expected_var * float(np.sqrt(2.0 / (_MARKOV_N - 1)))
    assert abs(mc_var - expected_var) < 3.0 * se_var, (
        f"MC var {mc_var:.5f} vs expected {expected_var:.5f}, 3*SE={3*se_var:.5f}"
    )


# ---------------------------------------------------------------------------
# AC 6: Smaller H → steeper skew with rho < 0 (NFR-3, FR-9, AD-10)
#
# With negative rho, the log-return distribution is negatively skewed.
# Rougher volatility (smaller H < 0.5) amplifies this skew.
# We test: skewness(H=0.1) < skewness(H=0.3) (both negative).
# ---------------------------------------------------------------------------

_SKEW_N = 8_000
_SKEW_T = 0.5     # shorter maturity: rough vol skew is more pronounced
_SKEW_TG = TimeGrid(jnp.linspace(0.0, _SKEW_T, 63))
_SKEW_ETA = 1.5
_SKEW_RHO = -0.7   # negative: produces left-skewed log-returns
_SKEW_XI0 = 0.04
_SKEW_S0 = 100.0

_MODEL_H01 = RoughBergomi(H=0.1, eta=_SKEW_ETA, rho=_SKEW_RHO, xi0=_SKEW_XI0, S0=_SKEW_S0, mu=0.0)
_MODEL_H03 = RoughBergomi(H=0.3, eta=_SKEW_ETA, rho=_SKEW_RHO, xi0=_SKEW_XI0, S0=_SKEW_S0, mu=0.0)


def _skewness(x: jnp.ndarray) -> float:
    mu = jnp.mean(x)
    sigma = jnp.std(x)
    return float(jnp.mean(((x - mu) / sigma) ** 3))


def test_rougher_H_produces_more_negative_skew():
    s_h01 = _simulate(_MODEL_H01, _SKEW_TG, _SKEW_N, seed=7)
    s_h03 = _simulate(_MODEL_H03, _SKEW_TG, _SKEW_N, seed=7)
    log_ret_h01 = jnp.log(s_h01.observation[:, -1] / _SKEW_S0)
    log_ret_h03 = jnp.log(s_h03.observation[:, -1] / _SKEW_S0)
    skew_h01 = _skewness(log_ret_h01)
    skew_h03 = _skewness(log_ret_h03)
    assert skew_h01 < skew_h03, (
        f"Expected skew(H=0.1)={skew_h01:.4f} < skew(H=0.3)={skew_h03:.4f} "
        "(rougher vol should produce steeper negative skew with rho < 0)"
    )


def test_negative_skew_is_present_for_negative_rho():
    # Both H=0.1 and H=0.3 should produce negative skew with rho < 0
    s_h01 = _simulate(_MODEL_H01, _SKEW_TG, _SKEW_N, seed=7)
    log_ret = jnp.log(s_h01.observation[:, -1] / _SKEW_S0)
    assert _skewness(log_ret) < 0, "Expected negative skewness with rho < 0"


def test_skew_steepness_monotone_in_H():
    # Three H values: skewness should increase (become less negative) with H
    models = [
        RoughBergomi(H=h, eta=_SKEW_ETA, rho=_SKEW_RHO, xi0=_SKEW_XI0, S0=_SKEW_S0, mu=0.0)
        for h in (0.1, 0.3, 0.49)
    ]
    skews = []
    for m in models:
        s = _simulate(m, _SKEW_TG, _SKEW_N, seed=13)
        log_ret = jnp.log(s.observation[:, -1] / _SKEW_S0)
        skews.append(_skewness(log_ret))
    assert skews[0] < skews[1] < skews[2], (
        f"Skewness should increase with H: {skews}"
    )


# ---------------------------------------------------------------------------
# AC 7: Conformance suite passes (FR-10, FR-11)
# ---------------------------------------------------------------------------

def test_rough_bergomi_passes_conformance_suite():
    assert_market_model_conforms(
        _MODEL, _TG, _N, _SEED, simulate_fn=_simulate
    )


# ---------------------------------------------------------------------------
# AC 8: Source imports only interface + equinox (AD-9)
# ---------------------------------------------------------------------------

def test_rough_bergomi_does_not_import_solver():
    src = (_pkg_root() / "models" / "_rough_bergomi.py").read_text()
    assert "solver" not in src
    assert "diffrax" not in src


def test_rough_bergomi_does_not_import_api():
    src = (_pkg_root() / "models" / "_rough_bergomi.py").read_text()
    assert "quantscenariobench.api" not in src
    assert "from ..api" not in src


def test_models_dependency_direction_includes_rough_bergomi():
    pkg_root = _pkg_root()
    violations: list[str] = []
    allowed = {"interface"}
    for py_file in (pkg_root / "models").rglob("*.py"):
        source = py_file.read_text()
        for m in re.finditer(r"from \.\.((\w+))", source):
            sub = m.group(1)
            if sub not in allowed:
                violations.append(f"{py_file.name}: imports from ..{sub}")
    assert not violations, "AD-9 violation in models:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Additional structural checks
# ---------------------------------------------------------------------------

def test_rough_bergomi_in_models_namespace():
    import quantscenariobench.models as m
    assert hasattr(m, "RoughBergomi")


def test_initial_state_shape():
    y0 = _MODEL.initial_state()
    assert y0.shape == (2,)


def test_initial_state_values():
    y0 = _MODEL.initial_state()
    assert jnp.allclose(y0[0], _S0)
    assert jnp.allclose(y0[1], _XI0)


def test_split_state():
    n, T = 4, 5
    ys = jnp.ones((n, T, 2)) * jnp.array([[[50.0, 0.05]]])
    obs, lat = _MODEL.split_state(ys)
    assert obs.shape == (n, T)
    assert lat.shape == (n, T)
    assert jnp.allclose(obs, 50.0)
    assert jnp.allclose(lat, 0.05)


def test_rough_bergomi_reproducible():
    s1 = _simulate(_MODEL, _TG, _N, _SEED)
    s2 = _simulate(_MODEL, _TG, _N, _SEED)
    assert jnp.array_equal(s1.observation, s2.observation)
    assert jnp.array_equal(s1.latent_state, s2.latent_state)


def test_different_seeds_differ():
    s1 = _simulate(_MODEL, _TG, _N, 0)
    s2 = _simulate(_MODEL, _TG, _N, 1)
    assert not jnp.array_equal(s1.observation, s2.observation)


def test_different_H_differ():
    m1 = RoughBergomi(H=0.1, eta=_ETA, rho=_RHO, xi0=_XI0, S0=_S0, mu=_MU)
    m2 = RoughBergomi(H=0.4, eta=_ETA, rho=_RHO, xi0=_XI0, S0=_S0, mu=_MU)
    s1 = _simulate(m1, _TG, _N, _SEED)
    s2 = _simulate(m2, _TG, _N, _SEED)
    assert not jnp.array_equal(s1.latent_state, s2.latent_state)


def test_variance_paths_finite():
    s = _simulate(_MODEL, _TG, _N, _SEED)
    assert jnp.all(jnp.isfinite(s.latent_state))
    assert jnp.all(jnp.isfinite(s.observation))
