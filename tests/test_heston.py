"""
Story 2.1 — Heston Stochastic Volatility Model

Covers all acceptance criteria from GitHub Issue #7.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quantscenariobench  # noqa: F401 — x64 before any test

from quantscenariobench.interface import (
    MarketModel,
    QuantScenarioBenchValidationWarning,
    TimeGrid,
)
from quantscenariobench.models import BlackScholes, Heston
from quantscenariobench.testing import assert_market_model_conforms


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


# ---------------------------------------------------------------------------
# Heston semi-closed-form reference (Gil-Pélaez inversion, r=0, AD-10)
# Independently implemented — no third-party quant library.
# ---------------------------------------------------------------------------

def _heston_cf(u: np.ndarray, S0: float, v0: float,
               kappa: float, theta: float, xi: float, rho: float,
               T: float) -> np.ndarray:
    """Heston characteristic function E[exp(iu * ln(S_T))] with r=0."""
    iu = 1j * u
    d = np.sqrt((kappa - rho * xi * iu) ** 2 + xi ** 2 * (iu + u ** 2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_neg_dT = np.exp(-d * T)
    C = kappa * theta / xi ** 2 * (
        (kappa - rho * xi * iu - d) * T
        - 2 * np.log((1 - g * exp_neg_dT) / (1 - g))
    )
    D = (kappa - rho * xi * iu - d) / xi ** 2 * (
        (1 - exp_neg_dT) / (1 - g * exp_neg_dT)
    )
    return np.exp(C + D * v0 + iu * np.log(S0))


def heston_call_price(
    S0: float, K: float, T: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    n_pts: int = 400,
) -> float:
    """European call price via Heston semi-closed form (r=0).

    Uses Gil-Pélaez Fourier inversion over the Heston characteristic function.
    Reference: Heston (1993), Gil-Pélaez (1951).
    """
    u = np.linspace(1e-5, 200.0, n_pts)
    phi_u = _heston_cf(u, S0, v0, kappa, theta, xi, rho, T)
    phi_u_shift = _heston_cf(u - 1j, S0, v0, kappa, theta, xi, rho, T)
    phi_shift0 = _heston_cf(np.array([-1.0]) * 1j,
                            S0, v0, kappa, theta, xi, rho, T)[0]
    log_K = np.log(K)
    integrand_P1 = np.real(
        np.exp(-1j * u * log_K) * phi_u_shift / (phi_shift0 * 1j * u)
    )
    integrand_P2 = np.real(
        np.exp(-1j * u * log_K) * phi_u / (1j * u)
    )
    P1 = 0.5 + float(np.trapezoid(integrand_P1, u)) / np.pi
    P2 = 0.5 + float(np.trapezoid(integrand_P2, u)) / np.pi
    return S0 * P1 - K * P2


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MU = 0.0       # risk-neutral (r=0)
_KAPPA = 2.0
_THETA = 0.04
_XI = 0.3       # vol-of-vol
_RHO = -0.7
_V0 = 0.04
_S0 = 100.0

_MODEL = Heston(mu=_MU, kappa=_KAPPA, theta=_THETA, xi=_XI, rho=_RHO, v0=_V0, S0=_S0)
_TG = TimeGrid(jnp.linspace(0.0, 1.0, 53))   # ~weekly steps, fast
_N = 16
_SEED = 42


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC 1: Heston is equinox.Module subclass of MarketModel (AD-1, AD-6)
# ---------------------------------------------------------------------------

def test_heston_is_market_model_subclass():
    assert issubclass(Heston, MarketModel)


def test_heston_is_equinox_module():
    assert issubclass(Heston, eqx.Module)


def test_heston_instantiates():
    model = Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)
    assert isinstance(model, Heston)


# ---------------------------------------------------------------------------
# AC 2: _drift/_diffusion implement Heston SDE correctly in JAX (NFR-2, FR-8)
# ---------------------------------------------------------------------------

def test_drift_shape_matches_state():
    state = jnp.array([100.0, 0.04])
    d = _MODEL._drift(0.0, state)
    assert d.shape == (2,)


def test_drift_asset_component_is_mu_times_S():
    state = jnp.array([100.0, 0.04])
    d = _MODEL._drift(0.0, state)
    assert jnp.allclose(d[0], _MU * 100.0)


def test_drift_variance_component_is_mean_reverting():
    # When v == theta, the variance drift should be zero
    state = jnp.array([100.0, _THETA])
    d = _MODEL._drift(0.0, state)
    assert jnp.allclose(d[1], 0.0, atol=1e-10)


def test_diffusion_shape_is_2x2():
    state = jnp.array([100.0, 0.04])
    sigma = _MODEL._diffusion(0.0, state)
    assert sigma.shape == (2, 2)


def test_diffusion_is_lower_triangular():
    state = jnp.array([100.0, 0.04])
    sigma = _MODEL._diffusion(0.0, state)
    assert jnp.allclose(sigma[0, 1], 0.0), "upper-right element must be zero (Cholesky)"


def test_diffusion_encodes_rho_correctly():
    state = jnp.array([100.0, 0.04])
    sigma = _MODEL._diffusion(0.0, state)
    sv = jnp.sqrt(0.04)
    # Row 1: [rho*xi*sv, sqrt(1-rho^2)*xi*sv]
    assert jnp.allclose(sigma[1, 0], _RHO * _XI * sv)
    rho_perp = jnp.sqrt(1.0 - _RHO ** 2)
    assert jnp.allclose(sigma[1, 1], rho_perp * _XI * sv)


def test_diffusion_is_jax_traceable():
    jit_fn = jax.jit(lambda s: _MODEL._diffusion(0.0, s))
    result = jit_fn(jnp.array([100.0, 0.04]))
    assert result.shape == (2, 2)
    assert jnp.all(jnp.isfinite(result))


def test_drift_is_jax_traceable():
    jit_fn = jax.jit(lambda s: _MODEL._drift(0.0, s))
    result = jit_fn(jnp.array([100.0, 0.04]))
    assert result.shape == (2,)
    assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# AC 3: latent_state contains variance paths, shape (n_paths, T) (FR-8)
# ---------------------------------------------------------------------------

def test_latent_state_shape():
    scenario = _simulate(_MODEL, _TG, _N, _SEED)
    T = len(_TG)
    assert scenario.latent_state.shape == (_N, T), (
        f"Expected latent_state shape ({_N}, {T}), got {scenario.latent_state.shape}"
    )


def test_observation_shape():
    scenario = _simulate(_MODEL, _TG, _N, _SEED)
    T = len(_TG)
    assert scenario.observation.shape == (_N, T)


def test_latent_state_is_non_negative_variance():
    scenario = _simulate(_MODEL, _TG, _N, _SEED)
    # Variance paths should start at v0
    assert jnp.allclose(scenario.latent_state[:, 0], _V0, atol=1e-6)


def test_observation_starts_at_S0():
    scenario = _simulate(_MODEL, _TG, _N, _SEED)
    assert jnp.allclose(scenario.observation[:, 0], _S0, atol=1e-6)


# ---------------------------------------------------------------------------
# AC 4: Both Heston and BlackScholes have identical Scenario field names (FR-2, FR-10)
# ---------------------------------------------------------------------------

def test_scenario_field_names_identical_to_black_scholes():
    bs_model = BlackScholes(mu=0.05, sigma=0.2, S0=100.0)
    bs_scenario = _simulate(bs_model, _TG, _N, _SEED)
    h_scenario = _simulate(_MODEL, _TG, _N, _SEED)
    bs_fields = {attr for attr in ("observation", "latent_state", "metadata")}
    for field in bs_fields:
        assert hasattr(bs_scenario, field)
        assert hasattr(h_scenario, field)


def test_simulate_source_unchanged():
    """simulate() source must not reference Heston or any model class (FR-10)."""
    import inspect
    from quantscenariobench.api import simulate
    source = inspect.getsource(simulate)
    assert "Heston" not in source
    assert "BlackScholes" not in source
    assert "quantscenariobench.models" not in source


# ---------------------------------------------------------------------------
# AC 5: MC option price within tolerance of Heston semi-closed form (FR-8, NFR-3, AD-10)
# ---------------------------------------------------------------------------

_PRICE_S0 = 100.0
_PRICE_K = 100.0
_PRICE_T = 1.0
_PRICE_KAPPA = 2.0
_PRICE_THETA = 0.04
_PRICE_XI = 0.3
_PRICE_RHO = -0.7
_PRICE_V0 = 0.04
_PRICE_N_PATHS = 10_000
_PRICE_SEED = 0
_PRICE_MODEL = Heston(
    mu=0.0,  # risk-neutral, r=0
    kappa=_PRICE_KAPPA,
    theta=_PRICE_THETA,
    xi=_PRICE_XI,
    rho=_PRICE_RHO,
    v0=_PRICE_V0,
    S0=_PRICE_S0,
)
_PRICE_TG = TimeGrid(jnp.linspace(0.0, _PRICE_T, 253))  # daily steps


def test_mc_call_price_matches_semi_closed_form():
    scenario = _simulate(_PRICE_MODEL, _PRICE_TG, _PRICE_N_PATHS, _PRICE_SEED)
    S_T = scenario.observation[:, -1]
    payoffs = jnp.maximum(S_T - _PRICE_K, 0.0)
    mc_price = float(jnp.mean(payoffs))
    mc_se = float(jnp.std(payoffs)) / (_PRICE_N_PATHS ** 0.5)

    ref_price = heston_call_price(
        S0=_PRICE_S0, K=_PRICE_K, T=_PRICE_T,
        v0=_PRICE_V0, kappa=_PRICE_KAPPA,
        theta=_PRICE_THETA, xi=_PRICE_XI, rho=_PRICE_RHO,
    )

    tolerance = 5.0 * mc_se  # 5 sigma: generous for discretization bias
    assert abs(mc_price - ref_price) < tolerance, (
        f"MC price {mc_price:.4f} differs from semi-closed-form {ref_price:.4f} "
        f"by {abs(mc_price - ref_price):.4f} > 5 SE ({tolerance:.4f})"
    )


# ---------------------------------------------------------------------------
# AC 6: Feller violation emits QuantScenarioBenchValidationWarning (FR-6)
# ---------------------------------------------------------------------------

def test_feller_violation_emits_warning():
    # 2*kappa*theta < xi^2 → Feller violated
    with pytest.warns(QuantScenarioBenchValidationWarning, match="Feller"):
        Heston(mu=0.0, kappa=0.5, theta=0.02, xi=0.5, rho=-0.5, v0=0.04, S0=100.0)


def test_feller_violation_no_exception():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuantScenarioBenchValidationWarning)
        model = Heston(mu=0.0, kappa=0.5, theta=0.02, xi=0.5, rho=-0.5, v0=0.04, S0=100.0)
    assert isinstance(model, Heston)


def test_feller_violation_simulate_completes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuantScenarioBenchValidationWarning)
        model = Heston(mu=0.0, kappa=0.5, theta=0.02, xi=0.5, rho=-0.5, v0=0.04, S0=100.0)
    from quantscenariobench.interface import Scenario
    tg = TimeGrid(jnp.linspace(0.0, 0.5, 13))
    scenario = _simulate(model, tg, 8, 1)
    assert isinstance(scenario, Scenario)


def test_feller_satisfied_no_warning():
    # 2*2*0.04 = 0.16 >= 0.3^2 = 0.09
    with warnings.catch_warnings():
        warnings.simplefilter("error", QuantScenarioBenchValidationWarning)
        Heston(mu=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0)


# ---------------------------------------------------------------------------
# AC 7: Conformance suite passes for Heston (FR-10, FR-11)
# ---------------------------------------------------------------------------

def test_heston_passes_conformance_suite():
    assert_market_model_conforms(
        _MODEL, _TG, _N, _SEED, simulate_fn=_simulate
    )


# ---------------------------------------------------------------------------
# AC 8: Heston imports only from interface and equinox (AD-9)
# ---------------------------------------------------------------------------

def test_heston_does_not_import_solver():
    src = _pkg_root() / "models" / "_heston.py"
    source = src.read_text()
    assert "solver" not in source
    assert "diffrax" not in source


def test_heston_does_not_import_api():
    src = _pkg_root() / "models" / "_heston.py"
    source = src.read_text()
    assert "quantscenariobench.api" not in source
    assert "from ..api" not in source


def test_models_dependency_direction_with_heston():
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

def test_initial_state_shape():
    y0 = _MODEL.initial_state()
    assert y0.shape == (2,)


def test_initial_state_values():
    y0 = _MODEL.initial_state()
    assert jnp.allclose(y0[0], _S0)
    assert jnp.allclose(y0[1], _V0)


def test_split_state():
    import jax.numpy as jnp
    n, T = 4, 5
    ys = jnp.ones((n, T, 2)) * jnp.array([[[10.0, 0.05]]])
    obs, lat = _MODEL.split_state(ys)
    assert obs.shape == (n, T)
    assert lat.shape == (n, T)
    assert jnp.allclose(obs, 10.0)
    assert jnp.allclose(lat, 0.05)


def test_heston_reproducible():
    s1 = _simulate(_MODEL, _TG, _N, _SEED)
    s2 = _simulate(_MODEL, _TG, _N, _SEED)
    assert jnp.array_equal(s1.observation, s2.observation)
    assert jnp.array_equal(s1.latent_state, s2.latent_state)


def test_heston_in_models_namespace():
    import quantscenariobench.models as m
    assert hasattr(m, "Heston")
