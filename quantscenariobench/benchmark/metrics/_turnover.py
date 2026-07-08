"""Portfolio turnover metric (FR-45): average one-way trade size per
rebalance, and its annualized variant.

turnover reads context.weights (the target weight sequence, Story 10.1)
and context.auxiliary["drifted_weights"] (the pre-trade drifted weight
right before every rebalance after the first — Story 10.1's weight-drift
convention, populated by run_benchmark()'s rebalancing loop and reused
as ProportionalCost's cost basis). Under buy-and-hold
(rebalance_schedule=None, a single PortfolioWeights) or a single
rebalance covering the whole window (no subsequent rebalance to trade
into), there is no trade to measure and turnover is well-defined as
exactly 0.0.

One-way convention (stated explicitly, per this story's requirements):
each rebalance's turnover is sum_j |w_target,j - w_drifted,j| — one-way,
per dollar traded; buy and sell legs of the same rebalance are not
double-counted (never doubled to a "round-trip" figure).

See README "Metric Conventions" for the risk-free-rate, annualization,
and compounding conventions shared across this metrics package.

turnover/turnover_annualized are opt-in Metric instances/factories —
never added to DEFAULT_METRICS (AD-32).
"""
from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import PortfolioWeights
from ._context import MetricContext
from ._metric import Direction, Metric


def _per_rebalance_turnovers(context: MetricContext) -> Optional[Float[Array, " r"]]:
    """The one-way turnover at each rebalance after the first, or None
    when there are no such rebalances (buy-and-hold, or a single
    rebalance covering the whole evaluation window).
    """
    if isinstance(context.weights, PortfolioWeights):
        return None
    drifted_weights: Optional[Sequence[Float[Array, " n"]]] = context.auxiliary.get(
        "drifted_weights"
    )
    if not drifted_weights:
        return None
    targets = [w.weights for w in context.weights[1:]]
    return jnp.stack([
        jnp.sum(jnp.abs(target - drifted))
        for target, drifted in zip(targets, drifted_weights)
    ])


class _Turnover:
    """Average one-way turnover per rebalance: mean_t sum_j |Δw_j,t|.

    Δw_j,t is the trade at rebalance t: the freshly allocated target
    weight minus the pre-trade drifted weight (not the difference
    between consecutive targets, which would ignore drift-driven
    trades — e.g. EqualWeight always re-targets the same 1/n weights,
    so its turnover comes entirely from correcting drift). Lower is
    better; well-defined as exactly 0.0 under buy-and-hold.
    """

    name = "turnover"
    direction: Direction = "lower_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        turnovers = _per_rebalance_turnovers(context)
        if turnovers is None:
            return jnp.asarray(0.0)
        return jnp.mean(turnovers)


class _AnnualizedTurnover:
    """Annualized turnover: mean per-rebalance turnover, scaled by the
    average number of rebalances per year (Story 9.3's periods_per_year
    convention: rebalances_per_year = periods_per_year * num_rebalances
    / t2, where t2 is the evaluation window length in periods).
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = periods_per_year
        self.name = f"turnover_annualized_{periods_per_year}"
        self.direction: Direction = "lower_is_better"
        self.params: dict[str, float] = {"periods_per_year": float(periods_per_year)}

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        turnovers = _per_rebalance_turnovers(context)
        if turnovers is None:
            return jnp.asarray(0.0)
        t2 = context.evaluation_returns.shape[0]
        num_rebalances = len(context.weights)
        rebalances_per_year = self.periods_per_year * num_rebalances / t2
        return jnp.mean(turnovers) * rebalances_per_year


turnover = _Turnover()


def turnover_annualized(periods_per_year: int = 252) -> Metric:
    """Construct a turnover_annualized_{periods_per_year} Metric (FR-45)."""
    return _AnnualizedTurnover(periods_per_year)
