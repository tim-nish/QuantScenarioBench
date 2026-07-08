"""Calmar Ratio and an opt-in annualization convention (FR-42).

See README "Metric Conventions" for the risk-free-rate, annualization,
and compounding conventions shared across this metrics package.

calmar_ratio/annualized_sharpe are opt-in, parametrized Metric factories
(Story 9.1's protocol) — never added to DEFAULT_METRICS (AD-32). The
un-annualized sharpe_ratio stays every existing metric's default.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._context import MetricContext
from ._final_wealth_factor import final_wealth_factor
from ._max_drawdown import max_drawdown
from ._metric import Direction, Metric
from ._sharpe import sharpe_ratio


class _CalmarRatio:
    """Calmar Ratio: annualized geometric return / |max_drawdown| (FR-42).

    Reuses final_wealth_factor's total-return compounding and
    max_drawdown's exact wealth/drawdown definition (never re-derived) —
    both already state the compounding convention this metric shares.
    periods_per_year=252 is this package's one documented annualization
    default (see README "Metric Conventions").

    Returns 0.0 (rather than inf/NaN) when max_drawdown is 0 — a
    monotonically increasing wealth path — mirroring _sharpe.py's AD-18
    degenerate-guard posture.
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = periods_per_year
        self.name = "calmar_ratio"
        self.direction: Direction = "higher_is_better"
        self.params: dict[str, float] = {"periods_per_year": float(periods_per_year)}

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        returns = context.portfolio_returns
        t = returns.shape[0]
        total_return = final_wealth_factor(returns) - 1.0
        annualized_return = (1.0 + total_return) ** (self.periods_per_year / t) - 1.0

        drawdown = max_drawdown(returns)  # non-positive fraction
        is_degenerate = drawdown == 0.0
        safe_drawdown = jnp.where(is_degenerate, 1.0, jnp.abs(drawdown))
        return jnp.where(is_degenerate, 0.0, annualized_return / safe_drawdown)


class _AnnualizedSharpe:
    """Annualized Sharpe: sharpe_ratio(context.portfolio_returns) * sqrt(periods_per_year) (FR-42).

    Reuses the existing bare sharpe_ratio function unchanged — never
    reimplemented — under the standard iid-scaling annualization
    convention. Registered under a distinct name
    (f"sharpe_ratio_annualized_{periods_per_year}"); never redefines or
    mutates sharpe_ratio itself, which stays the un-annualized default
    for every existing registry (AD-32).
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = periods_per_year
        self.name = f"sharpe_ratio_annualized_{periods_per_year}"
        self.direction: Direction = "higher_is_better"
        self.params: dict[str, float] = {"periods_per_year": float(periods_per_year)}

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        return sharpe_ratio(context.portfolio_returns) * jnp.sqrt(self.periods_per_year)


def calmar_ratio(periods_per_year: int = 252) -> Metric:
    """Construct a calmar_ratio Metric (FR-42), annualized at periods_per_year (default 252)."""
    return _CalmarRatio(periods_per_year)


def annualized_sharpe(periods_per_year: int = 252) -> Metric:
    """Construct a sharpe_ratio_annualized_{periods_per_year} Metric (FR-42)."""
    return _AnnualizedSharpe(periods_per_year)
