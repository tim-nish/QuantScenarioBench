"""Excess kurtosis and aggregational-Gaussianity diagnostics (FR-49, AD-38).

Pure jax.numpy, vectorized over the path axis via direct broadcasting —
no per-path Python loop (AC7). Every per-path statistic is a
degenerate-guarded jnp.where computation (a zero-variance path never
divides by zero), matching quantscenariobench.benchmark.metrics'
_sharpe.py-style degenerate-guard convention.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

_MONTHLY_AGGREGATION_WINDOW = 21  # trading days per "month" (README's canonical daily grid)


def excess_kurtosis_per_path(x: Float[Array, "n_paths t"]) -> Float[Array, " n_paths"]:
    """Per-path excess kurtosis: mean((x - mean)^4) / std(x)^4 - 3.

    Guarded against a zero-variance path (returns 0.0, never NaN/inf).
    """
    mean = jnp.mean(x, axis=1, keepdims=True)
    centered = x - mean
    variance = jnp.mean(centered ** 2, axis=1)
    fourth_moment = jnp.mean(centered ** 4, axis=1)
    is_degenerate = variance <= 0.0
    safe_variance = jnp.where(is_degenerate, 1.0, variance)
    kurtosis = fourth_moment / (safe_variance ** 2) - 3.0
    return jnp.where(is_degenerate, 0.0, kurtosis)


def aggregate_returns(
    returns: Float[Array, "n_paths t"], window: int = _MONTHLY_AGGREGATION_WINDOW
) -> Float[Array, "n_paths t_over_window"]:
    """Sum every `window` consecutive returns into one coarser-frequency
    return (e.g. daily -> monthly, matching the README's canonical daily
    grid of 21 trading days per month) — trailing returns that don't
    fill a full window are dropped.
    """
    t = returns.shape[1]
    usable_length = (t // window) * window
    trimmed = returns[:, :usable_length]
    reshaped = trimmed.reshape(returns.shape[0], -1, window)
    return jnp.sum(reshaped, axis=2)


def kurtosis_decay_per_path(
    returns: Float[Array, "n_paths t"], window: int = _MONTHLY_AGGREGATION_WINDOW
) -> tuple[Float[Array, " n_paths"], Float[Array, " n_paths"]]:
    """Aggregational Gaussianity (FR-49): excess kurtosis of the daily
    series vs. the temporally-aggregated (e.g. monthly) series. Returns
    (aggregated_kurtosis, decay), where decay = daily_kurtosis -
    aggregated_kurtosis — positive when kurtosis genuinely shrinks under
    aggregation (the stylized fact), near-zero when the daily series is
    already close to Gaussian (e.g. Black-Scholes).
    """
    daily_kurtosis = excess_kurtosis_per_path(returns)
    aggregated = aggregate_returns(returns, window)
    aggregated_kurtosis = excess_kurtosis_per_path(aggregated)
    decay = daily_kurtosis - aggregated_kurtosis
    return aggregated_kurtosis, decay
