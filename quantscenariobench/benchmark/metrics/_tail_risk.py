"""Value-at-Risk and Conditional Value-at-Risk tail-risk metrics (FR-41).

Both metrics operate on the loss series L = -portfolio_returns (losses
reported as positive numbers, direction="lower_is_better") — the sign
convention every docstring below restates explicitly, since it is the
one metric pair in this registry where "the number itself" and "the
direction" could otherwise be second-guessed.

CVaR follows the Rockafellar-Uryasev (2000) convention:
CVaR_alpha(Z) = min_nu (nu + E[(Z - nu)+] / (1 - alpha)), whose minimizing
nu* is exactly the empirical alpha-quantile of the loss series — so it is
evaluated here as a direct closed-form expression (quantile + tail mean),
never an inner optimization loop, keeping both metrics jit-compatible.

value_at_risk/conditional_value_at_risk are opt-in, parametrized Metric
factories (Story 9.1's protocol) — never added to DEFAULT_METRICS (AD-32).
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._context import MetricContext
from ._metric import Direction, Metric


class _ValueAtRisk:
    """VaR(alpha): the alpha-quantile of the loss series (FR-41).

    jnp.quantile's default method="linear" interpolation rule is used
    (matching numpy.quantile's default) — the alpha-quantile is linearly
    interpolated between the two nearest order statistics of L when
    alpha * (t - 1) is not an integer.

    Degenerate guard: when the confidence level implies fewer than one
    tail observation — (1 - alpha) * t < 1 — the quantile estimate is
    unreliable, so VaR falls back to the max loss in the series (the most
    conservative finite estimate), documented rather than left to
    silently interpolate past the available tail data.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.name = f"var_{alpha}"
        self.direction: Direction = "lower_is_better"
        self.params: dict[str, float] = {"alpha": alpha}

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        losses = -context.portfolio_returns
        t = losses.shape[0]
        var = jnp.quantile(losses, self.alpha)
        undersized_tail = (1.0 - self.alpha) * t < 1
        return jnp.where(undersized_tail, jnp.max(losses), var)


class _ConditionalValueAtRisk:
    """CVaR(alpha): mean loss in the alpha-tail (FR-41, Rockafellar-Uryasev).

    Evaluated as the closed-form nu* + mean(relu(L - nu*)) / (1 - alpha),
    where nu* = the empirical alpha-quantile of L (jnp.quantile, default
    method="linear" interpolation) — the Rockafellar-Uryasev dual optimum
    evaluated directly, not via an inner minimization over nu.

    Degenerate guard: identical to value_at_risk's — when
    (1 - alpha) * t < 1, the tail mean is unreliable and CVaR falls back
    to the max loss in the series.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.name = f"cvar_{alpha}"
        self.direction: Direction = "lower_is_better"
        self.params: dict[str, float] = {"alpha": alpha}

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        losses = -context.portfolio_returns
        t = losses.shape[0]
        nu_star = jnp.quantile(losses, self.alpha)
        tail_mean = jnp.mean(jnp.maximum(losses - nu_star, 0.0)) / (1.0 - self.alpha)
        cvar = nu_star + tail_mean
        undersized_tail = (1.0 - self.alpha) * t < 1
        return jnp.where(undersized_tail, jnp.max(losses), cvar)


def value_at_risk(alpha: float) -> Metric:
    """Construct a var_{alpha} Metric (FR-41). alpha is a required argument
    (mirrors CVaROptimization.confidence_level, AD-15) — never an internal
    hardcoded default.
    """
    return _ValueAtRisk(alpha)


def conditional_value_at_risk(alpha: float) -> Metric:
    """Construct a cvar_{alpha} Metric (FR-41). alpha is a required argument
    (mirrors CVaROptimization.confidence_level, AD-15) — never an internal
    hardcoded default.
    """
    return _ConditionalValueAtRisk(alpha)
