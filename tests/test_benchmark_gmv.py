"""
Story 5.2 — Global Minimum Variance Baseline & the Optimizer Solver Layer

Covers all acceptance criteria from GitHub Issue #28.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import scipy.optimize

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def _correlated_returns() -> jnp.ndarray:
    """A small synthetic 3-asset return series with mild positive
    correlation via a common factor, chosen so the unconstrained
    closed-form GMV solution happens to be non-negative."""
    key = jax.random.PRNGKey(0)
    t, n = 60, 3
    factor = jax.random.normal(key, (t,)) * 0.01
    idio = jax.random.normal(jax.random.fold_in(key, 1), (t, n)) * jnp.array(
        [0.02, 0.03, 0.015]
    )
    loadings = jnp.array([0.5, 0.8, 0.3])
    return factor[:, None] * loadings + idio


# ---------------------------------------------------------------------------
# AC: quantscenariobench.benchmark.solver is the only benchmark-layer
# module that imports scipy (AD-14)
# ---------------------------------------------------------------------------

_SCIPY_IMPORT = re.compile(r"(?:import|from)\s+scipy\b")


def test_solver_is_the_only_benchmark_module_importing_scipy():
    # Story 10.3 (AD-35) introduces one deliberate, bounded second
    # exception: quantscenariobench.benchmark.evaluation._compare_strategies
    # imports scipy.stats for paired significance testing (ttest_rel,
    # wilcoxon), which has no jax.numpy-native equivalent — documented
    # explicitly in that module's own docstring, not a silent violation.
    _SANCTIONED_SECOND_EXCEPTION = "_compare_strategies.py"

    benchmark_root = _pkg_root() / "benchmark"
    violations = []
    for py_file in benchmark_root.rglob("*.py"):
        if py_file.parent.name == "solver":
            continue
        if py_file.name == _SANCTIONED_SECOND_EXCEPTION:
            continue
        if _SCIPY_IMPORT.search(py_file.read_text()):
            violations.append(str(py_file.relative_to(_pkg_root().parent)))
    assert not violations, (
        f"AD-14 violation: only quantscenariobench.benchmark.solver (and "
        f"the Story 10.3/AD-35 {_SANCTIONED_SECOND_EXCEPTION} exception) "
        f"may import scipy: {violations}"
    )


def test_global_minimum_variance_never_imports_scipy_directly():
    src = (
        _pkg_root() / "benchmark" / "strategies" / "_global_minimum_variance.py"
    ).read_text()
    assert not _SCIPY_IMPORT.search(src), (
        "GlobalMinimumVariance must never import scipy directly — only "
        "quantscenariobench.benchmark.solver (AD-14)"
    )


# ---------------------------------------------------------------------------
# AC: long_only=False computes weights via jax.numpy.linalg, no solver call
# (AD-14, AD-25)
# ---------------------------------------------------------------------------

def test_gmv_unconstrained_matches_closed_form_covariance_inversion():
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    returns = _correlated_returns()
    n = returns.shape[1]

    covariance = jnp.cov(returns, rowvar=False)
    ones = jnp.ones((n,))
    expected = jnp.linalg.solve(covariance, ones)
    expected = expected / jnp.sum(expected)

    strat = GlobalMinimumVariance(long_only=False)
    weights = strat.allocate(returns)

    assert jnp.allclose(weights.weights, expected, atol=1e-8)


def test_gmv_unconstrained_does_not_call_solver(monkeypatch):
    from quantscenariobench.benchmark import solver as solver_module
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    def fail_if_called(*args, **kwargs):
        raise AssertionError("long_only=False must never call solve_allocation")

    monkeypatch.setattr(solver_module, "solve_allocation", fail_if_called)

    # GlobalMinimumVariance imports solve_allocation into its own namespace
    # at module load time, so patch it there too.
    import quantscenariobench.benchmark.strategies._global_minimum_variance as gmv_module
    monkeypatch.setattr(gmv_module, "solve_allocation", fail_if_called)

    strat = GlobalMinimumVariance(long_only=False)
    weights = strat.allocate(_correlated_returns())
    assert weights.weights.shape == (3,)


# ---------------------------------------------------------------------------
# AC: long_only=True calls solve_allocation (scipy.optimize.minimize,
# SLSQP) and returns a non-negative PortfolioWeights (AD-14)
# ---------------------------------------------------------------------------

def test_gmv_constrained_calls_solve_allocation(monkeypatch):
    import quantscenariobench.benchmark.strategies._global_minimum_variance as gmv_module
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    called = {"count": 0}
    original = gmv_module.solve_allocation

    def spy(covariance):
        called["count"] += 1
        return original(covariance)

    monkeypatch.setattr(gmv_module, "solve_allocation", spy)

    strat = GlobalMinimumVariance(long_only=True)
    weights = strat.allocate(_correlated_returns())

    assert called["count"] == 1
    assert bool(jnp.all(weights.weights >= 0))
    assert float(jnp.abs(jnp.sum(weights.weights) - 1.0)) <= 1e-6


# ---------------------------------------------------------------------------
# AC: GlobalMinimumVariance's variance is no greater than EqualWeight's on
# the same data (FR-21 sanity property)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("long_only", [False, True])
def test_gmv_variance_no_greater_than_equal_weight(long_only):
    from quantscenariobench.benchmark.strategies import EqualWeight, GlobalMinimumVariance

    returns = _correlated_returns()
    covariance = jnp.cov(returns, rowvar=False)

    gmv = GlobalMinimumVariance(long_only=long_only)
    eq = EqualWeight()

    gmv_weights = gmv.allocate(returns).weights
    eq_weights = eq.allocate(returns).weights

    gmv_variance = gmv_weights @ covariance @ gmv_weights
    eq_variance = eq_weights @ covariance @ eq_weights

    assert float(gmv_variance) <= float(eq_variance) + 1e-9


# ---------------------------------------------------------------------------
# AC: a solve_allocation(...) call that fails to converge raises
# QuantScenarioBenchSolverError (AD-14)
# ---------------------------------------------------------------------------

class _FailedResult:
    success = False
    message = "mock non-convergence"
    x = None


def test_solve_allocation_raises_on_solver_failure(monkeypatch):
    from quantscenariobench.benchmark.solver import (
        QuantScenarioBenchSolverError,
        solve_allocation,
    )

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: _FailedResult())

    with pytest.raises(QuantScenarioBenchSolverError):
        solve_allocation(jnp.eye(3))


def test_gmv_constrained_propagates_solver_failure(monkeypatch):
    from quantscenariobench.benchmark.interface import PortfolioWeights  # noqa: F401
    from quantscenariobench.benchmark.solver import QuantScenarioBenchSolverError
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: _FailedResult())

    strat = GlobalMinimumVariance(long_only=True)
    with pytest.raises(QuantScenarioBenchSolverError):
        strat.allocate(_correlated_returns())


# ---------------------------------------------------------------------------
# AC: the Story 4.2 conformance suite passes for both long_only=True and
# long_only=False (FR-23, FR-25 cross-check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("long_only", [False, True])
def test_gmv_passes_conformance_suite(long_only):
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance
    from quantscenariobench.benchmark.testing import assert_baseline_strategy_conforms

    strat = GlobalMinimumVariance(long_only=long_only)
    assert_baseline_strategy_conforms(strat, _correlated_returns())


# ---------------------------------------------------------------------------
# Story 11.1 (Issue #86) AC6: on a strongly correlated 2-block
# CorrelatedBasket, GMV exploits the block structure with lower portfolio
# variance than EqualWeight — an end-to-end proof that correlation
# reaches the benchmark layer (FR-47)
# ---------------------------------------------------------------------------

def test_gmv_exploits_correlated_basket_block_structure_vs_equal_weight():
    from quantscenariobench.api import simulate_correlated_basket
    from quantscenariobench.benchmark.returns import compose_returns
    from quantscenariobench.benchmark.strategies import EqualWeight, GlobalMinimumVariance
    from quantscenariobench.interface import TimeGrid
    from quantscenariobench.models import BlackScholes

    time_grid = TimeGrid(jnp.linspace(0.0, 1.0, 60))
    # Two highly-correlated pairs (assets 0-1, assets 2-3), pairs mutually
    # uncorrelated; the second member of each pair has meaningfully higher
    # volatility than the first, so a genuine (non-equal-weight) long-only
    # GMV optimum exists to exploit.
    models = [
        BlackScholes(mu=0.05, sigma=0.9, S0=100.0),
        BlackScholes(mu=0.05, sigma=1.1, S0=100.0),
        BlackScholes(mu=0.05, sigma=0.5, S0=100.0),
        BlackScholes(mu=0.05, sigma=0.6, S0=100.0),
    ]
    rho = jnp.array([
        [1.0, 0.9, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
    ])
    scenarios, _ = simulate_correlated_basket(
        models, time_grid, n_paths=1000, seed=3, rho=rho
    )
    returns = compose_returns(scenarios, path_index=0)
    historical_returns, evaluation_returns = returns[:30], returns[30:]

    gmv = GlobalMinimumVariance(long_only=True)
    gmv_weights = gmv.allocate(historical_returns).weights
    equal_weights = jnp.full(4, 0.25)

    assert not jnp.allclose(gmv_weights, equal_weights), (
        "GMV landed on equal weight — dataset is not a meaningful sanity check"
    )

    gmv_variance = float(jnp.var(evaluation_returns @ gmv_weights))
    equal_weight_variance = float(jnp.var(evaluation_returns @ equal_weights))
    assert gmv_variance < equal_weight_variance
