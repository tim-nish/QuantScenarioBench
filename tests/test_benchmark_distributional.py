"""
Story 10.3 — Distributional Evaluation Across Scenario Paths

Covers all acceptance criteria from GitHub Issue #85.
"""

from __future__ import annotations

import dataclasses
from unittest import mock

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test
from quantscenariobench.api import simulate
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes


def _scenarios(n_paths, seeds=(1, 2), t=60, mus=(0.05, 0.03), sigmas=(0.2, 0.15)):
    tg = TimeGrid(jnp.linspace(0.0, 1.0, t))
    return [
        simulate(BlackScholes(mu=mu, sigma=sigma, S0=100.0), tg, n_paths=n_paths, seed=seed)
        for mu, sigma, seed in zip(mus, sigmas, seeds)
    ]


# ---------------------------------------------------------------------------
# AC1: n_repeats=1 collapses exactly to today's single-path run_benchmark()
# result on the same path selection (FR-46)
# ---------------------------------------------------------------------------

def test_n_repeats_1_collapses_to_ordinary_run_benchmark_result():
    from quantscenariobench.benchmark.returns import compose_returns
    from quantscenariobench.benchmark.runner import run_benchmark, run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=50)
    seed = 42

    distributional_result = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=1, seed=seed
    )
    assert distributional_result.metrics_distribution is None

    path_index = int(jax.random.permutation(jax.random.PRNGKey(seed), 50)[0])
    returns = compose_returns(scenarios, path_index=path_index)
    manual_result = run_benchmark(EqualWeight(), returns[:30], returns[30:])

    actual = dataclasses.asdict(distributional_result)
    expected = dataclasses.asdict(manual_result)
    actual.pop("generated_at")
    expected.pop("generated_at")
    assert actual == expected


# ---------------------------------------------------------------------------
# AC2: a run with R repeats carries an additive metrics_distribution with
# {mean, std, ci_low, ci_high, n_repeats} per metric; metrics stays the
# scalar mean (FR-46, AD-35)
# ---------------------------------------------------------------------------

def test_metrics_distribution_shape_and_metrics_is_the_mean():
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=50)
    result = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=16, seed=1
    )

    assert set(result.metrics_distribution) == set(result.metrics)
    for name, distribution in result.metrics_distribution.items():
        assert {"mean", "std", "ci_low", "ci_high", "n_repeats"} <= set(distribution)
        assert distribution["n_repeats"] == 16
        assert distribution["ci_low"] <= distribution["mean"] <= distribution["ci_high"]
        assert distribution["std"] >= 0.0
        assert result.metrics[name] == pytest.approx(distribution["mean"])


def test_default_n_repeats_is_32():
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=50)
    result = run_benchmark_distributional(EqualWeight(), scenarios, n_historical=30, seed=1)
    assert result.metrics_distribution["sharpe_ratio"]["n_repeats"] == 32


# ---------------------------------------------------------------------------
# AC3: EqualWeight compared against itself via compare_strategies, using
# the same R draws, gives mean difference 0 and p ~= 1 (guard against
# pairing bugs)
# ---------------------------------------------------------------------------

def test_compare_strategies_self_comparison_gives_zero_difference_and_p_approx_1():
    from quantscenariobench.benchmark.evaluation import compare_strategies
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=64)
    result_a = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=32, seed=7
    )
    result_b = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=32, seed=7
    )

    comparison = compare_strategies(result_a, result_b, "sharpe_ratio")
    assert comparison["mean_difference"] == 0.0
    assert comparison["p_value_ttest"] == pytest.approx(1.0)
    assert comparison["p_value_wilcoxon"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AC4: a synthetic setup with a known dominant strategy (one asset given
# a large drift) — the paired test detects the difference at p < 0.001
# for moderate R
# ---------------------------------------------------------------------------

def test_compare_strategies_detects_known_dominant_strategy_at_p_below_0_001():
    from quantscenariobench.benchmark.evaluation import compare_strategies
    from quantscenariobench.benchmark.interface import BaselineStrategy, PortfolioWeights
    from quantscenariobench.benchmark.runner import run_benchmark_distributional

    tg = TimeGrid(jnp.linspace(0.0, 1.0, 60))
    dominant_asset = simulate(BlackScholes(mu=2.0, sigma=0.1, S0=100.0), tg, n_paths=64, seed=3)
    weak_asset = simulate(BlackScholes(mu=0.03, sigma=0.15, S0=50.0), tg, n_paths=64, seed=4)
    scenarios = [dominant_asset, weak_asset]

    class OverweightDominant(BaselineStrategy):
        def allocate(self, historical_returns):
            return PortfolioWeights(jnp.array([0.95, 0.05]))

    class OverweightWeak(BaselineStrategy):
        def allocate(self, historical_returns):
            return PortfolioWeights(jnp.array([0.05, 0.95]))

    result_good = run_benchmark_distributional(
        OverweightDominant(), scenarios, n_historical=30, n_repeats=32, seed=9
    )
    result_bad = run_benchmark_distributional(
        OverweightWeak(), scenarios, n_historical=30, n_repeats=32, seed=9
    )

    comparison = compare_strategies(result_good, result_bad, "final_wealth_factor")
    assert comparison["mean_difference"] > 0.0
    assert comparison["p_value_ttest"] < 0.001
    assert comparison["p_value_wilcoxon"] < 0.001


def test_compare_strategies_requires_same_number_of_paired_repeats():
    from quantscenariobench.benchmark.evaluation import compare_strategies
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=64)
    result_a = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=16, seed=1
    )
    result_b = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=32, seed=1
    )
    with pytest.raises(ValueError):
        compare_strategies(result_a, result_b, "sharpe_ratio")


def test_compare_strategies_requires_distributional_results():
    from quantscenariobench.benchmark.evaluation import compare_strategies
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=64)
    result_single = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=1, seed=1
    )
    result_multi = run_benchmark_distributional(
        EqualWeight(), scenarios, n_historical=30, n_repeats=32, seed=1
    )
    with pytest.raises(ValueError):
        compare_strategies(result_single, result_multi, "sharpe_ratio")


# ---------------------------------------------------------------------------
# AC5: same seed/path-selection rule run twice on the same backend gives
# bit-identical metrics_distribution (FR-46, NFR-1 extended)
# ---------------------------------------------------------------------------

def test_metrics_distribution_is_deterministic_for_the_same_seed():
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    scenarios = _scenarios(n_paths=64)
    strategy = GlobalMinimumVariance(long_only=False)

    result_a = run_benchmark_distributional(
        strategy, scenarios, n_historical=30, n_repeats=32, seed=123
    )
    result_b = run_benchmark_distributional(
        strategy, scenarios, n_historical=30, n_repeats=32, seed=123
    )
    assert result_a.metrics_distribution == result_b.metrics_distribution
    assert result_a.metrics == result_b.metrics


# ---------------------------------------------------------------------------
# AC7: the per-repeat evaluation reuses already-simulated Scenario paths
# — no re-simulation — and R defaults to 32 (already covered above)
# ---------------------------------------------------------------------------

def test_run_benchmark_distributional_never_re_simulates():
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=50)

    with mock.patch(
        "quantscenariobench.solver.solve_sde", side_effect=AssertionError("re-simulated!")
    ) as mocked:
        run_benchmark_distributional(
            EqualWeight(), scenarios, n_historical=30, n_repeats=16, seed=1
        )
    mocked.assert_not_called()


def test_run_benchmark_distributional_module_documents_memory_shape():
    from pathlib import Path

    src = (
        Path(__file__).parent.parent
        / "quantscenariobench"
        / "benchmark"
        / "runner"
        / "_run_benchmark_distributional.py"
    ).read_text()
    assert "(R, t, n)" in src


# ---------------------------------------------------------------------------
# compare_strategies is a second, deliberate scipy import point outside
# quantscenariobench.benchmark.solver, called out explicitly (Review Focus)
# ---------------------------------------------------------------------------

def test_compare_strategies_module_documents_scipy_exception():
    from pathlib import Path

    src = (
        Path(__file__).parent.parent
        / "quantscenariobench"
        / "benchmark"
        / "evaluation"
        / "_compare_strategies.py"
    ).read_text()
    assert "scipy" in src.lower()
    assert "second" in src.lower() or "deliberate" in src.lower()


def test_run_benchmark_distributional_raises_when_n_paths_below_n_repeats():
    from quantscenariobench.benchmark.runner import run_benchmark_distributional
    from quantscenariobench.benchmark.strategies import EqualWeight

    scenarios = _scenarios(n_paths=10)
    with pytest.raises(ValueError):
        run_benchmark_distributional(
            EqualWeight(), scenarios, n_historical=30, n_repeats=32, seed=1
        )
