"""
Story 13.1 — Scenario Realism Diagnostics — Stylized-Facts Validation Suite

Covers all acceptance criteria from GitHub Issue #88.

Reference values below are hand-derived independently using plain NumPy/
SciPy (statistics primitives, never quantscenariobench's own jax.numpy
implementation), computed in this test file only — mirroring the
existing metrics' "no numpy/scipy in the library, only in tests"
convention (AD-10).
"""

from __future__ import annotations

import ast
import dataclasses
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test
from quantscenariobench.api import simulate
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes, Heston


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def _tg(n=253):
    return TimeGrid(jnp.linspace(0.0, 1.0, n))


# ---------------------------------------------------------------------------
# AC4: diagnostics match independent NumPy/SciPy reference implementations
# to 1e-10 on fixed statistical fixtures (FR-49, NFR-3 extended)
# ---------------------------------------------------------------------------

def _fixed_fixture():
    rng = np.random.default_rng(0)
    returns_np = rng.normal(0, 0.02, 300)
    returns_np[50] = 0.15  # inject a fat-tail outlier
    return returns_np


def _ref_acf(x, lag):
    x = np.asarray(x)
    mean = x.mean()
    centered = x - mean
    return np.sum(centered[:-lag] * centered[lag:]) / np.sum(centered ** 2)


def test_excess_kurtosis_matches_scipy_reference():
    from quantscenariobench.diagnostics._kurtosis import excess_kurtosis_per_path

    returns_np = _fixed_fixture()
    returns = jnp.array(returns_np).reshape(1, -1)

    actual = float(excess_kurtosis_per_path(returns)[0])
    expected = scipy.stats.kurtosis(returns_np, fisher=True, bias=True)
    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("lag", [1, 5, 21])
def test_return_acf_matches_hand_derived_reference(lag):
    from quantscenariobench.diagnostics._autocorrelation import acf_at_lag

    returns_np = _fixed_fixture()
    returns = jnp.array(returns_np).reshape(1, -1)

    actual = float(acf_at_lag(returns, lag)[0])
    expected = _ref_acf(returns_np, lag)
    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("lag", [1, 5, 21])
def test_squared_return_acf_matches_hand_derived_reference(lag):
    from quantscenariobench.diagnostics._autocorrelation import acf_at_lag

    returns_np = _fixed_fixture()
    squared = returns_np ** 2
    squared_returns = jnp.array(squared).reshape(1, -1)

    actual = float(acf_at_lag(squared_returns, lag)[0])
    expected = _ref_acf(squared, lag)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_leverage_correlation_matches_numpy_reference():
    from quantscenariobench.diagnostics._autocorrelation import leverage_correlation_per_path

    returns_np = _fixed_fixture()
    returns = jnp.array(returns_np).reshape(1, -1)

    r_t = returns_np[:-1]
    r_next_squared = returns_np[1:] ** 2
    expected = np.corrcoef(r_t, r_next_squared)[0, 1]
    actual = float(leverage_correlation_per_path(returns)[0])
    assert actual == pytest.approx(expected, abs=1e-10)


def test_kurtosis_decay_matches_hand_derived_reference():
    from quantscenariobench.diagnostics._kurtosis import kurtosis_decay_per_path

    rng = np.random.default_rng(1)
    returns_np = rng.normal(0, 0.02, 210)  # 210 = 10 * 21, divides evenly
    returns = jnp.array(returns_np).reshape(1, -1)

    actual_agg, actual_decay = kurtosis_decay_per_path(returns)

    monthly = returns_np.reshape(-1, 21).sum(axis=1)
    expected_agg = scipy.stats.kurtosis(monthly, fisher=True, bias=True)
    expected_daily = scipy.stats.kurtosis(returns_np, fisher=True, bias=True)
    expected_decay = expected_daily - expected_agg

    assert float(actual_agg[0]) == pytest.approx(expected_agg, abs=1e-10)
    assert float(actual_decay[0]) == pytest.approx(expected_decay, abs=1e-10)


# ---------------------------------------------------------------------------
# AC2: Black-Scholes scenarios report near-zero return ACF, near-zero
# squared-return ACF, and excess kurtosis ~= 0 — GBM correctly reproduces
# only the trivial stylized facts (FR-49)
# ---------------------------------------------------------------------------

def test_black_scholes_reproduces_only_trivial_stylized_facts():
    from quantscenariobench.diagnostics import realism_report

    scenario = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(), n_paths=5000, seed=1
    )
    report = realism_report(scenario)

    assert abs(report.excess_kurtosis.mean) < 0.5
    assert not report.excess_kurtosis.in_band  # GBM lacks the fat tails real markets show

    for stat in (report.return_acf_lag1, report.return_acf_lag5, report.return_acf_lag21):
        assert abs(stat.mean) < 0.05
        assert stat.in_band  # the one trivial stylized fact GBM does reproduce

    for stat in (
        report.squared_return_acf_lag1,
        report.squared_return_acf_lag5,
        report.squared_return_acf_lag21,
    ):
        assert abs(stat.mean) < 0.05
        assert not stat.in_band  # GBM has no volatility clustering

    assert not report.leverage_correlation.in_band  # GBM has no leverage effect


# ---------------------------------------------------------------------------
# AC3: Heston (rho=-0.7) reports positive squared-return ACF at lag 1 and
# negative leverage correlation, both in-band; the same model with
# xi -> 0 degrades toward Black-Scholes values (monotonicity, FR-49)
# ---------------------------------------------------------------------------

def test_heston_shows_volatility_clustering_and_leverage_effect():
    from quantscenariobench.diagnostics import realism_report

    scenario = simulate(
        Heston(mu=0.05, kappa=3.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0),
        _tg(), n_paths=5000, seed=2,
    )
    report = realism_report(scenario)

    assert report.squared_return_acf_lag1.mean > 0.0
    assert report.squared_return_acf_lag1.in_band
    assert report.leverage_correlation.mean < 0.0
    assert report.leverage_correlation.in_band


def test_heston_degrades_toward_black_scholes_as_xi_shrinks():
    from quantscenariobench.diagnostics import realism_report

    scenario_high_xi = simulate(
        Heston(mu=0.05, kappa=3.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, S0=100.0),
        _tg(), n_paths=5000, seed=2,
    )
    scenario_low_xi = simulate(
        Heston(mu=0.05, kappa=2.0, theta=0.04, xi=0.001, rho=-0.7, v0=0.04, S0=100.0),
        _tg(), n_paths=5000, seed=3,
    )
    scenario_bs = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(), n_paths=5000, seed=1
    )

    report_high = realism_report(scenario_high_xi)
    report_low = realism_report(scenario_low_xi)
    report_bs = realism_report(scenario_bs)

    # Monotonicity: the high-xi Heston's effect sizes are strictly larger
    # in magnitude than the near-zero-xi Heston's, which in turn are close
    # to Black-Scholes' (near-zero) values — a real degradation toward
    # GBM, not two unrelated fixed-value assertions.
    assert abs(report_high.squared_return_acf_lag1.mean) > abs(report_low.squared_return_acf_lag1.mean)
    assert abs(report_low.squared_return_acf_lag1.mean - report_bs.squared_return_acf_lag1.mean) < \
        abs(report_high.squared_return_acf_lag1.mean - report_bs.squared_return_acf_lag1.mean)

    assert abs(report_high.leverage_correlation.mean) > abs(report_low.leverage_correlation.mean)
    assert abs(report_low.leverage_correlation.mean - report_bs.leverage_correlation.mean) < \
        abs(report_high.leverage_correlation.mean - report_bs.leverage_correlation.mean)


# ---------------------------------------------------------------------------
# AC5: RealismReport JSON round-trips and renders in a dataset card without
# manual editing (FR-49)
# ---------------------------------------------------------------------------

def test_realism_report_json_round_trips():
    from quantscenariobench.diagnostics import RealismReport, realism_report

    scenario = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(60), n_paths=200, seed=1
    )
    report = realism_report(scenario)

    payload = json.dumps(dataclasses.asdict(report))
    restored = RealismReport.from_dict(json.loads(payload))

    assert restored == report


def test_realism_report_is_plain_frozen_dataclass_not_equinox_module():
    import equinox as eqx

    from quantscenariobench.diagnostics import realism_report

    scenario = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(60), n_paths=200, seed=1
    )
    report = realism_report(scenario)

    assert dataclasses.is_dataclass(report)
    assert type(report).__dataclass_params__.frozen is True
    assert not isinstance(report, eqx.Module)

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.seed = 999


def test_realism_report_renders_in_dataset_card_without_manual_editing():
    from quantscenariobench.diagnostics import realism_report
    from quantscenariobench.export import generate_dataset_card

    scenario = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(60), n_paths=200, seed=1
    )
    report = realism_report(scenario)
    card = generate_dataset_card(scenario, realism_report=report)

    assert "Scenario Realism Diagnostics" in card
    assert "excess_kurtosis" in card
    assert "leverage_correlation" in card


# ---------------------------------------------------------------------------
# AC6: each diagnostic's reference band flags in-band/out-of-band without
# rejecting or filtering the scenario (FR-49, AD-38)
# ---------------------------------------------------------------------------

def test_out_of_band_diagnostic_is_reported_not_rejected():
    from quantscenariobench.diagnostics import realism_report

    # Black-Scholes fails volatility clustering (out-of-band) but
    # realism_report must still return a complete report, never raise.
    scenario = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(), n_paths=2000, seed=1
    )
    report = realism_report(scenario)  # must not raise

    assert report.squared_return_acf_lag1.in_band is False
    assert report.excess_kurtosis is not None  # the report is complete regardless


def test_diagnostics_module_never_rejects_or_filters_a_scenario():
    src = (_pkg_root() / "diagnostics" / "_report.py").read_text()
    assert "raise" not in src


# ---------------------------------------------------------------------------
# AC7: report on a 10k-path scenario completes in seconds — vectorized
# over paths, no per-path Python loop (FR-49)
# ---------------------------------------------------------------------------

def test_realism_report_on_10k_paths_completes_quickly():
    from quantscenariobench.diagnostics import realism_report

    scenario = simulate(
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0), _tg(), n_paths=10_000, seed=1
    )
    start = time.time()
    realism_report(scenario)
    elapsed = time.time() - start

    assert elapsed < 30.0  # generous ceiling; typical runs are ~1-2s


def test_diagnostics_modules_have_no_per_path_python_loop():
    """AC7's actual invariant, checked by construction: no `for` loop over
    a path-shaped axis anywhere in the diagnostics computation modules
    (only vectorized broadcasting/jnp reductions with an explicit axis).
    """
    for filename in ("_returns.py", "_kurtosis.py", "_autocorrelation.py"):
        src = (_pkg_root() / "diagnostics" / filename).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.For, ast.While)), (
                f"{filename} contains a Python loop ({type(node).__name__}) — "
                "diagnostics must be vectorized over paths, no per-path loop (AC7)"
            )


# ---------------------------------------------------------------------------
# Diagnostics module dependency direction (AD-9): only quantscenariobench.
# interface, never quantscenariobench.benchmark/models
# ---------------------------------------------------------------------------

def test_diagnostics_module_imports_only_interface():
    import re

    forbidden_import = re.compile(
        r"(?:import|from)\s+(?:quantscenariobench\.)?(?:\.{1,2})?(benchmark|models)\b"
    )
    for py_file in (_pkg_root() / "diagnostics").rglob("*.py"):
        src = py_file.read_text()
        assert not forbidden_import.search(src), (
            f"{py_file.name} must not import quantscenariobench.benchmark/models (AD-9)"
        )


def test_diagnostics_is_pure_jax_never_numpy_or_scipy():
    import re

    forbidden_import = re.compile(r"(?:import|from)\s+(scipy|numpy)\b")
    for py_file in (_pkg_root() / "diagnostics").rglob("*.py"):
        assert not forbidden_import.search(py_file.read_text()), (
            f"{py_file.name} must be jax.numpy-only, never scipy/numpy directly"
        )
