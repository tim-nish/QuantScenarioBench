"""Paired significance testing between two distributional BenchmarkResults
(FR-46, AD-35).

This module imports scipy.stats directly — a second, deliberate exception
to the "only quantscenariobench.benchmark.solver imports scipy" framing
AD-25 establishes for the Optimizer Solver Layer. Paired significance
testing (a paired t-test, a distribution-free Wilcoxon signed-rank
alternative) has no jax.numpy-native equivalent worth reimplementing;
scipy is already a core dependency via the solver, so this module reuses
it directly rather than routing through quantscenariobench.benchmark.solver
(that module is for the LP/SLSQP Optimizer Solver Layer, a different
concern). This is a bounded, documented exception, not an undocumented
architecture violation — never route this comparison logic through
quantscenariobench.benchmark.solver.
"""
from __future__ import annotations

import numpy as np
import scipy.stats

from ..interface import BenchmarkResult


def _per_repeat_values(result: BenchmarkResult, metric_name: str) -> np.ndarray:
    if result.metrics_distribution is None:
        raise ValueError(
            "compare_strategies requires a BenchmarkResult produced by "
            "run_benchmark_distributional() with n_repeats > 1 "
            "(metrics_distribution is None)"
        )
    if metric_name not in result.metrics_distribution:
        raise ValueError(
            f"compare_strategies: metric {metric_name!r} is not present "
            "in this BenchmarkResult's metrics_distribution"
        )
    return np.asarray(result.metrics_distribution[metric_name]["values"])


def compare_strategies(
    result_a: BenchmarkResult, result_b: BenchmarkResult, metric_name: str
) -> dict:
    """Paired comparison of metric_name between two distributional runs (FR-46).

    result_a/result_b must both come from run_benchmark_distributional()
    with n_repeats > 1, and — critically — from the *same* R return
    draws: callers must pass identical scenarios/seed/n_repeats to both
    run_benchmark_distributional() calls being compared (paired by
    construction). compare_strategies does not and cannot re-derive draw
    alignment after the fact from the results alone — it pairs
    result_a.metrics_distribution[metric_name]["values"][i] with
    result_b's value at the same index i, trusting the caller's setup.

    Returns {mean_difference, p_value_ttest, p_value_wilcoxon}. When
    every paired difference is exactly zero (e.g. a strategy compared
    against itself, AC3), both p-values are defined as 1.0 directly —
    scipy.stats.ttest_rel returns NaN with a RuntimeWarning on
    zero-variance input, which this guard avoids, mirroring this
    codebase's degenerate-input-guard convention (AD-18) for a
    zero-variance case that has an unambiguous "no effect detected"
    answer.
    """
    values_a = _per_repeat_values(result_a, metric_name)
    values_b = _per_repeat_values(result_b, metric_name)
    if values_a.shape != values_b.shape:
        raise ValueError(
            "compare_strategies requires both results to have the same "
            f"number of paired repeats; got {values_a.shape[0]} vs "
            f"{values_b.shape[0]}"
        )

    differences = values_a - values_b
    mean_difference = float(differences.mean())

    if np.allclose(differences, 0.0):
        p_value_ttest = 1.0
        p_value_wilcoxon = 1.0
    else:
        p_value_ttest = float(scipy.stats.ttest_rel(values_a, values_b).pvalue)
        p_value_wilcoxon = float(scipy.stats.wilcoxon(values_a, values_b).pvalue)

    return {
        "mean_difference": mean_difference,
        "p_value_ttest": p_value_ttest,
        "p_value_wilcoxon": p_value_wilcoxon,
    }
