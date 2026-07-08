"""
Story 12.1 — Hierarchical Risk Parity Baseline Strategy

Covers all acceptance criteria from GitHub Issue #87.

The reference value below is hand-derived independently in this test
file from Lopez de Prado (2016)'s own published pseudocode, using plain
NumPy/SciPy primitives (scipy.cluster.hierarchy.linkage/leaves_list,
scipy.spatial.distance.squareform) directly — never adapted from an
existing HRP implementation or portfolio-analytics library (AD-10),
and never by calling quantscenariobench's own
hierarchical_risk_parity_weights.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# A fixed 5-asset covariance fixture (2 correlated blocks: 0-1-2 and 3-4),
# used for the AC1 reference-value match.
# ---------------------------------------------------------------------------

_CORR = np.array([
    [1.00, 0.90, 0.85, 0.10, 0.05],
    [0.90, 1.00, 0.88, 0.12, 0.08],
    [0.85, 0.88, 1.00, 0.09, 0.06],
    [0.10, 0.12, 0.09, 1.00, 0.80],
    [0.05, 0.08, 0.06, 0.80, 1.00],
])
_STD = np.array([0.10, 0.12, 0.09, 0.20, 0.22])
_COV = _CORR * np.outer(_STD, _STD)


def _hand_derived_reference_weights(cov: np.ndarray, linkage_method: str = "single") -> np.ndarray:
    """Independent from-scratch NumPy/SciPy implementation of Lopez de
    Prado (2016)'s three-stage HRP algorithm, written directly from the
    paper's own pseudocode (getIVP, getClusterVar, getQuasiDiag,
    getRecBipart) — not by calling this library's own solver function.
    """
    n = cov.shape[0]
    diag = np.diag(cov)
    corr = cov / np.outer(np.sqrt(diag), np.sqrt(diag))

    def get_ivp(cov_slice):
        d = np.diag(cov_slice)
        ivp = 1.0 / d
        return ivp / ivp.sum()

    def get_cluster_var(cov, items):
        cov_slice = cov[np.ix_(items, items)]
        w = get_ivp(cov_slice).reshape(-1, 1)
        return (w.T @ cov_slice @ w)[0, 0]

    distance = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(distance, 0.0)
    condensed = ssd.squareform(distance, checks=False)
    link = hierarchy.linkage(condensed, method=linkage_method)
    sort_ix = hierarchy.leaves_list(link).tolist()

    w = np.ones(len(sort_ix))
    clusters = [list(range(len(sort_ix)))]
    while clusters:
        split = []
        for c in clusters:
            if len(c) > 1:
                mid = len(c) // 2
                split.append(c[:mid])
                split.append(c[mid:])
        clusters = split
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            idx0 = [sort_ix[j] for j in c0]
            idx1 = [sort_ix[j] for j in c1]
            var0 = get_cluster_var(cov, idx0)
            var1 = get_cluster_var(cov, idx1)
            alpha = 1.0 - var0 / (var0 + var1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha

    weights = np.zeros(n)
    for position, asset in enumerate(sort_ix):
        weights[asset] = w[position]
    return weights


# ---------------------------------------------------------------------------
# AC1: QSB's HRP matches an independently computed reference to 1e-8 on a
# fixed 5-asset covariance fixture (FR-48, AD-37)
# ---------------------------------------------------------------------------

def test_hrp_matches_hand_derived_reference_to_1e_minus_8():
    from quantscenariobench.benchmark.solver import hierarchical_risk_parity_weights

    actual = np.asarray(hierarchical_risk_parity_weights(jnp.array(_COV), "single"))
    expected = _hand_derived_reference_weights(_COV, "single")

    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_hrp_strategy_matches_hand_derived_reference_via_allocate():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    rng = np.random.default_rng(0)
    returns = rng.multivariate_normal(np.zeros(5), _COV, size=200)

    strat = HierarchicalRiskParity()
    actual = np.asarray(strat.allocate(jnp.array(returns)).weights)

    sample_cov = np.cov(returns, rowvar=False)
    expected = _hand_derived_reference_weights(sample_cov, "single")

    np.testing.assert_allclose(actual, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# AC2: randomized covariance property test — weights non-negative and sum
# to 1, satisfying PortfolioWeights' existing contract (FR-48)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_assets", [2, 3, 5, 8])
@pytest.mark.parametrize("trial_seed", [0, 1, 2])
def test_hrp_weights_satisfy_portfolio_weights_contract_on_random_covariance(n_assets, trial_seed):
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    rng = np.random.default_rng(trial_seed * 100 + n_assets)
    a = rng.normal(size=(n_assets, n_assets))
    covariance = a @ a.T + 1e-6 * np.eye(n_assets)  # random PSD covariance
    returns = rng.multivariate_normal(np.zeros(n_assets), covariance, size=100)

    strat = HierarchicalRiskParity()
    weights = strat.allocate(jnp.array(returns)).weights  # raises if contract violated

    assert bool(jnp.all(weights >= 0))
    assert float(jnp.abs(jnp.sum(weights) - 1.0)) <= 1e-6


# ---------------------------------------------------------------------------
# AC3: on a 2-block correlated basket, HRP allocates across blocks rather
# than concentrating within one (FR-48)
# ---------------------------------------------------------------------------

def test_hrp_allocates_across_blocks_on_correlated_basket():
    from quantscenariobench.api import simulate_correlated_basket
    from quantscenariobench.benchmark.returns import compose_returns
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity
    from quantscenariobench.interface import TimeGrid
    from quantscenariobench.models import BlackScholes

    time_grid = TimeGrid(jnp.linspace(0.0, 1.0, 60))
    models = [
        BlackScholes(mu=0.03, sigma=0.30, S0=100.0),
        BlackScholes(mu=0.03, sigma=0.35, S0=100.0),
        BlackScholes(mu=0.02, sigma=0.20, S0=100.0),
        BlackScholes(mu=0.02, sigma=0.22, S0=100.0),
    ]
    rho = jnp.array([
        [1.0, 0.9, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
    ])
    scenarios, _ = simulate_correlated_basket(models, time_grid, n_paths=1, seed=3, rho=rho)
    returns = compose_returns(scenarios, path_index=0)

    weights = HierarchicalRiskParity().allocate(returns).weights
    block_a = float(weights[0] + weights[1])
    block_b = float(weights[2] + weights[3])

    assert block_a > 0.05
    assert block_b > 0.05


# ---------------------------------------------------------------------------
# AC4: degenerate inputs — n=1 asset, a zero-variance asset, a
# constant-correlation matrix (FR-48)
# ---------------------------------------------------------------------------

def test_hrp_n_equals_1_receives_weight_1():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    returns = jnp.array([[0.01], [0.02], [-0.01], [0.03], [0.0]])
    weights = HierarchicalRiskParity().allocate(returns).weights
    assert jnp.array_equal(weights, jnp.array([1.0]))


def test_hrp_zero_variance_asset_gets_guarded_no_nan():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    rng = np.random.default_rng(1)
    returns = rng.multivariate_normal(np.zeros(5), _COV, size=200)
    returns[:, 2] = 0.0  # asset 2 has zero variance (constant, no return)

    weights = HierarchicalRiskParity().allocate(jnp.array(returns)).weights

    assert not bool(jnp.any(jnp.isnan(weights)))
    assert not bool(jnp.any(jnp.isinf(weights)))
    assert bool(jnp.all(weights >= 0))
    assert float(jnp.abs(jnp.sum(weights) - 1.0)) <= 1e-6
    # the zero-variance asset takes the cluster's full weight in the
    # limiting inverse-variance sense (documented degenerate guard)
    assert float(weights[2]) == pytest.approx(1.0, abs=1e-8)


def test_hrp_constant_correlation_matrix_ties_break_deterministically():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    n = 4
    rng = np.random.default_rng(2)
    const_corr = np.full((n, n), 0.3)
    np.fill_diagonal(const_corr, 1.0)
    const_std = np.full(n, 0.1)
    const_cov = const_corr * np.outer(const_std, const_std)
    returns = rng.multivariate_normal(np.zeros(n), const_cov, size=150)

    strat = HierarchicalRiskParity()
    weights_a = strat.allocate(jnp.array(returns)).weights
    weights_b = strat.allocate(jnp.array(returns)).weights

    assert not bool(jnp.any(jnp.isnan(weights_a)))
    assert jnp.array_equal(weights_a, weights_b)


# ---------------------------------------------------------------------------
# AC5: run_benchmark(HierarchicalRiskParity(), ...) flows end-to-end into
# an EvaluationResult and a leaderboard row with zero Runner changes
# (FR-48)
# ---------------------------------------------------------------------------

def test_hrp_flows_end_to_end_through_run_benchmark_and_leaderboard():
    import jax

    from quantscenariobench.benchmark.evaluation import (
        aggregate_evaluation_results,
        to_evaluation_result,
    )
    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    hist = jax.random.normal(jax.random.PRNGKey(0), (30, 4)) * 0.01
    eval_ = jax.random.normal(jax.random.PRNGKey(1), (20, 4)) * 0.01

    result = run_benchmark(
        HierarchicalRiskParity(), hist, eval_,
        asset_scenario_ids=["a0", "a1", "a2", "a3"],
        time_grid_reference="tg-hrp",
    )
    assert result.strategy_name == "HierarchicalRiskParity"
    assert set(result.metrics) == {
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "final_wealth_factor",
    }

    evaluation_result = to_evaluation_result(result)
    table = aggregate_evaluation_results([evaluation_result])
    assert len(table) == 1
    assert table[0]["strategy"] == "HierarchicalRiskParity"


# ---------------------------------------------------------------------------
# AC6: linkage_method and other strategy fields are plain
# JSON-serializable dataclass fields, snapshotted unchanged into
# BenchmarkResult.strategy_parameters (FR-48, AD-37)
# ---------------------------------------------------------------------------

def test_hrp_linkage_method_is_a_recorded_strategy_parameter():
    import dataclasses

    import jax

    from quantscenariobench.benchmark.runner import run_benchmark
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    field_names = {f.name for f in dataclasses.fields(HierarchicalRiskParity)}
    assert "linkage_method" in field_names

    strat = HierarchicalRiskParity(linkage_method="average")
    assert strat.linkage_method == "average"

    hist = jax.random.normal(jax.random.PRNGKey(0), (30, 4)) * 0.01
    eval_ = jax.random.normal(jax.random.PRNGKey(1), (20, 4)) * 0.01
    result = run_benchmark(strat, hist, eval_)
    assert result.strategy_parameters == {"linkage_method": "average"}


def test_hrp_default_linkage_method_is_single():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    assert HierarchicalRiskParity().linkage_method == "single"


# ---------------------------------------------------------------------------
# AC7: same historical_returns run twice on the same backend produces
# identical weights (FR-48)
# ---------------------------------------------------------------------------

def test_hrp_is_deterministic_across_repeated_calls():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity

    rng = np.random.default_rng(5)
    returns = jnp.array(rng.multivariate_normal(np.zeros(6), np.eye(6) * 0.01, size=150))

    strat = HierarchicalRiskParity()
    weights_a = strat.allocate(returns)
    weights_b = strat.allocate(returns)

    assert jnp.array_equal(weights_a.weights, weights_b.weights)


# ---------------------------------------------------------------------------
# AC8: the Portfolio Optimizer conformance suite passes for HRP (FR-48,
# NFR-3 extended)
# ---------------------------------------------------------------------------

def test_hrp_passes_conformance_suite():
    from quantscenariobench.benchmark.strategies import HierarchicalRiskParity
    from quantscenariobench.benchmark.testing import assert_baseline_strategy_conforms

    rng = np.random.default_rng(6)
    returns = jnp.array(rng.multivariate_normal(np.zeros(5), _COV, size=100))

    assert_baseline_strategy_conforms(HierarchicalRiskParity(), returns)


# ---------------------------------------------------------------------------
# Review Focus: HRP's clustering is routed through
# quantscenariobench.benchmark.solver — never imported directly in the
# strategy class (AD-14/AD-19)
# ---------------------------------------------------------------------------

_SCIPY_IMPORT = re.compile(r"(?:import|from)\s+scipy\b")


def test_hierarchical_risk_parity_never_imports_scipy_directly():
    src = (
        _pkg_root() / "benchmark" / "strategies" / "_hierarchical_risk_parity.py"
    ).read_text()
    assert not _SCIPY_IMPORT.search(src), (
        "HierarchicalRiskParity must never import scipy directly — only "
        "quantscenariobench.benchmark.solver (AD-14)"
    )


def test_hrp_solver_function_is_exported_from_solver_package():
    from quantscenariobench.benchmark.solver import hierarchical_risk_parity_weights
    assert callable(hierarchical_risk_parity_weights)
