"""
Story 5.3 — CVaR Optimization Baseline

Covers all acceptance criteria from GitHub Issue #29.

The correctness reference below is hand-derived via a brute-force grid
search over portfolio weights, independently implemented with plain
Python/NumPy — never quantscenariobench's own scipy.optimize.linprog LP
formulation, and never a portfolio-analytics library (AD-10 amended).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


_SCIPY_IMPORT = re.compile(r"(?:import|from)\s+scipy\b")


# ---------------------------------------------------------------------------
# AC: CVaROptimization constructed without confidence_level raises (AD-15)
# ---------------------------------------------------------------------------

def test_cvar_optimization_requires_confidence_level():
    from quantscenariobench.benchmark.strategies import CVaROptimization
    with pytest.raises(TypeError):
        CVaROptimization()


# ---------------------------------------------------------------------------
# AC: CVaROptimization(confidence_level=0.95).allocate(...) calls
# solve_allocation (scipy.optimize.linprog, Rockafellar-Uryasev) and
# returns a PortfolioWeights satisfying AD-20 (FR-22, AD-14)
# ---------------------------------------------------------------------------

def _sample_returns() -> jnp.ndarray:
    return jnp.array([
        [0.05, -0.02],
        [-0.03, 0.04],
        [0.02, -0.01],
        [0.08, -0.05],
        [-0.10, 0.06],
        [0.01, 0.03],
    ])


def test_cvar_optimization_calls_solve_allocation(monkeypatch):
    import quantscenariobench.benchmark.strategies._cvar_optimization as cvar_module
    from quantscenariobench.benchmark.strategies import CVaROptimization

    called = {"count": 0}
    original = cvar_module.solve_allocation

    def spy(**kwargs):
        called["count"] += 1
        return original(**kwargs)

    monkeypatch.setattr(cvar_module, "solve_allocation", spy)

    strat = CVaROptimization(confidence_level=0.95)
    weights = strat.allocate(_sample_returns())

    assert called["count"] == 1
    assert bool(jnp.all(weights.weights >= 0))
    assert float(jnp.abs(jnp.sum(weights.weights) - 1.0)) <= 1e-6


def test_solve_allocation_uses_linprog_for_cvar(monkeypatch):
    import scipy.optimize
    from quantscenariobench.benchmark.solver import solve_allocation

    called = {"count": 0}
    original_linprog = scipy.optimize.linprog

    def spy(*args, **kwargs):
        called["count"] += 1
        return original_linprog(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", spy)

    solve_allocation(returns=_sample_returns(), confidence_level=0.95)
    assert called["count"] == 1


# ---------------------------------------------------------------------------
# AC: CVaROptimization source never imports scipy directly (AD-14)
# ---------------------------------------------------------------------------

def test_cvar_optimization_never_imports_scipy_directly():
    src = (
        _pkg_root() / "benchmark" / "strategies" / "_cvar_optimization.py"
    ).read_text()
    assert not _SCIPY_IMPORT.search(src), (
        "CVaROptimization must never import scipy directly — only "
        "quantscenariobench.benchmark.solver (AD-14)"
    )


# ---------------------------------------------------------------------------
# AC: confidence_level=0.95 is present in the strategy's recorded
# identity/parameters (FR-22, AD-15)
# ---------------------------------------------------------------------------

def test_confidence_level_is_a_recorded_field():
    from quantscenariobench.benchmark.strategies import CVaROptimization

    strat = CVaROptimization(confidence_level=0.95)
    assert strat.confidence_level == 0.95

    field_names = {f.name for f in dataclasses.fields(CVaROptimization)}
    assert "confidence_level" in field_names


# ---------------------------------------------------------------------------
# AC: correctness — weights match a hand-derived reference CVaR-minimizing
# allocation within tolerance (AD-10 amended)
# ---------------------------------------------------------------------------

def _brute_force_cvar_reference(returns: np.ndarray, alpha: float, grid_step: float = 0.0005):
    """Grid-search the long-only, 2-asset CVaR-minimizing weight w1
    (w2 = 1 - w1), evaluating the exact R-U objective at each grid point.

    For a fixed w, the R-U objective's minimizing zeta is one of the
    per-period loss values (piecewise-linear convexity), so this scan is
    exact in zeta and only approximate in the w grid.
    """
    t = returns.shape[0]
    best_w1 = None
    best_obj = None
    for w1 in np.arange(0.0, 1.0 + grid_step, grid_step):
        w = np.array([w1, 1.0 - w1])
        losses = -(returns @ w)
        best_obj_for_w = min(
            zeta + (1.0 / ((1.0 - alpha) * t)) * np.sum(np.maximum(losses - zeta, 0.0))
            for zeta in losses
        )
        if best_obj is None or best_obj_for_w < best_obj:
            best_obj = best_obj_for_w
            best_w1 = w1
    return best_w1, best_obj


def test_cvar_optimization_matches_hand_derived_reference():
    from quantscenariobench.benchmark.strategies import CVaROptimization

    returns_jax = _sample_returns()
    returns_np = np.asarray(returns_jax)
    alpha = 0.75

    strat = CVaROptimization(confidence_level=alpha)
    weights = strat.allocate(returns_jax).weights

    ref_w1, _ = _brute_force_cvar_reference(returns_np, alpha)

    assert float(weights[0]) == pytest.approx(ref_w1, abs=5e-3)
    assert float(weights[1]) == pytest.approx(1.0 - ref_w1, abs=5e-3)


# ---------------------------------------------------------------------------
# AC: the Story 4.2 conformance suite passes for CVaROptimization
# (FR-23, FR-25 cross-check)
# ---------------------------------------------------------------------------

def test_cvar_optimization_passes_conformance_suite():
    from quantscenariobench.benchmark.strategies import CVaROptimization
    from quantscenariobench.benchmark.testing import assert_baseline_strategy_conforms

    strat = CVaROptimization(confidence_level=0.95)
    assert_baseline_strategy_conforms(strat, _sample_returns())
