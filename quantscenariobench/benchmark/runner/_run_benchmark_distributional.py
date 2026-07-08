"""run_benchmark_distributional() — evaluate a strategy over R independent
draws of already-simulated Scenario paths (FR-46, AD-35).

Memory shape (AC7): evaluating R repeats conceptually touches an
(R, t, n) volume of returns data, but this implementation never
materializes it as a single array — each repeat's (t, n) returns matrix
is composed via compose_returns(..., path_index=...), scored through the
existing run_benchmark() pipeline, and discarded before the next repeat
starts. Peak additional memory beyond a single run_benchmark() call is
one (t, n) matrix plus R scalar metric values per metric name — never
R x (t, n). No Scenario is re-simulated: every repeat reuses a different
already-simulated path out of the same Scenario objects (AD-35).

This module uses plain NumPy (not jax.numpy) for the bootstrap
percentile CI and sample statistics, since that computation is a
scalar-level resample of R already-computed metric values, run entirely
outside any jit/vmap trace — the same posture the Optimizer Solver Layer
(AD-25) takes for its own non-JAX-native computation.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import jax
import numpy as np

from ...interface import Scenario
from ..interface import BaselineStrategy, BenchmarkResult, ForecastOptimizer, PolicyStrategy
from ..returns import compose_returns
from ._run_benchmark import _library_version, _utc_now, run_benchmark

_DEFAULT_N_REPEATS = 32
_DEFAULT_N_BOOTSTRAP = 2000
_DEFAULT_CONFIDENCE = 0.95


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, confidence: float
) -> tuple[float, float]:
    """A resample-of-scalars percentile bootstrap CI for the mean of
    values (AD-35) — resamples the R already-computed metric values
    themselves, never re-running run_benchmark(), so it stays cheap even
    at larger R.
    """
    n = values.shape[0]
    resampled_means = rng.choice(values, size=(_DEFAULT_N_BOOTSTRAP, n), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    ci_low = float(np.percentile(resampled_means, 100 * alpha / 2))
    ci_high = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


def run_benchmark_distributional(
    strategy: BaselineStrategy | ForecastOptimizer | PolicyStrategy,
    scenarios: Sequence[Scenario],
    n_historical: int,
    *,
    n_repeats: int = _DEFAULT_N_REPEATS,
    seed: int,
    confidence: float = _DEFAULT_CONFIDENCE,
    **run_benchmark_kwargs,
) -> BenchmarkResult:
    """Evaluate strategy over n_repeats independent path draws (FR-46).

    scenarios must share one TimeGrid (compose_returns' existing
    invariant) and carry n_paths >= n_repeats. Each repeat r selects one
    of the scenarios' available paths — a seeded random permutation of
    path indices, so a fixed seed always draws the same R paths (AC5,
    NFR-1) — composes that path's full (t, n) returns matrix, splits it
    into historical_returns = returns[:n_historical] and
    evaluation_returns = returns[n_historical:] (mirroring the single-path
    convention), and scores it through the unmodified run_benchmark()
    pipeline. **run_benchmark_kwargs (metrics, rebalance_schedule,
    cost_model, forecast, asset_scenario_ids, time_grid_reference, ...)
    are forwarded identically to every repeat.

    n_repeats=1 collapses to exactly today's single-path run_benchmark()
    result (AC1): the one repeat's own BenchmarkResult is returned
    unchanged, with metrics_distribution left None — no mean/std/CI
    machinery runs at all, by construction, not by coincidence.

    n_repeats>1 (AC2): the returned BenchmarkResult.metrics is the
    per-metric mean across repeats (unchanged scalar-field shape, for
    every existing metrics/Leaderboard consumer); metrics_distribution
    is additive: {mean, std, ci_low, ci_high, n_repeats, values} per
    metric, where "values" is the raw list of R per-repeat metric values
    — required for compare_strategies' paired significance test, and the
    channel through which draw alignment must be guaranteed by the
    caller (same scenarios/seed/n_repeats on both compared calls).
    std uses ddof=1 (sample standard deviation); ci_low/ci_high are a
    confidence-level (default 95%) bootstrap percentile CI resampled
    from the R values themselves (AD-35).
    """
    n_paths = scenarios[0].metadata.n_paths
    if n_paths < n_repeats:
        raise ValueError(
            f"run_benchmark_distributional requires n_paths >= n_repeats; "
            f"got n_paths={n_paths}, n_repeats={n_repeats}"
        )
    path_indices = jax.random.permutation(jax.random.PRNGKey(seed), n_paths)[:n_repeats]

    per_repeat_results = []
    for path_index in path_indices:
        returns = compose_returns(scenarios, path_index=int(path_index))
        historical_returns = returns[:n_historical]
        evaluation_returns = returns[n_historical:]
        per_repeat_results.append(
            run_benchmark(strategy, historical_returns, evaluation_returns, **run_benchmark_kwargs)
        )

    if n_repeats == 1:
        return per_repeat_results[0]

    metric_names = list(per_repeat_results[0].metrics.keys())
    rng = np.random.default_rng(seed)
    metrics_distribution: dict = {}
    for name in metric_names:
        values = np.asarray([result.metrics[name] for result in per_repeat_results])
        ci_low, ci_high = _bootstrap_ci(values, rng, confidence)
        metrics_distribution[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_repeats": n_repeats,
            "values": [float(v) for v in values],
        }

    base_result = per_repeat_results[0]
    mean_metrics = {name: metrics_distribution[name]["mean"] for name in metric_names}
    return dataclasses.replace(
        base_result,
        metrics=mean_metrics,
        metrics_distribution=metrics_distribution,
        library_version=_library_version(),
        generated_at=_utc_now(),
    )
