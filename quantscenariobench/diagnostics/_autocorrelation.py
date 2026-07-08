"""Autocorrelation and leverage-effect diagnostics (FR-49, AD-38).

Pure jax.numpy, vectorized over the path axis via direct broadcasting —
no per-path Python loop (AC7). Degenerate (zero-variance) paths are
guarded via jnp.where, never dividing by zero.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

LAGS = (1, 5, 21)


def acf_at_lag(x: Float[Array, "n_paths t"], lag: int) -> Float[Array, " n_paths"]:
    """Sample autocorrelation of x at the given lag, per path:
    rho_k = sum_t (x_t - mean)(x_{t+k} - mean) / sum_t (x_t - mean)^2.
    """
    mean = jnp.mean(x, axis=1, keepdims=True)
    centered = x - mean
    numerator = jnp.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
    denominator = jnp.sum(centered ** 2, axis=1)
    is_degenerate = denominator <= 0.0
    safe_denominator = jnp.where(is_degenerate, 1.0, denominator)
    acf = numerator / safe_denominator
    return jnp.where(is_degenerate, 0.0, acf)


def leverage_correlation_per_path(returns: Float[Array, "n_paths t"]) -> Float[Array, " n_paths"]:
    """Correlation of r_t with the *subsequent* squared return r_{t+1}^2,
    per path (the leverage effect: negative for equity-like scenarios
    with rho < 0, since a negative return tends to be followed by higher
    volatility).
    """
    r_t = returns[:, :-1]
    r_next_squared = returns[:, 1:] ** 2

    mean_r = jnp.mean(r_t, axis=1, keepdims=True)
    mean_r2 = jnp.mean(r_next_squared, axis=1, keepdims=True)
    covariance = jnp.mean((r_t - mean_r) * (r_next_squared - mean_r2), axis=1)

    std_r = jnp.std(r_t, axis=1)
    std_r2 = jnp.std(r_next_squared, axis=1)
    denominator = std_r * std_r2
    is_degenerate = denominator <= 0.0
    safe_denominator = jnp.where(is_degenerate, 1.0, denominator)
    correlation = covariance / safe_denominator
    return jnp.where(is_degenerate, 0.0, correlation)
