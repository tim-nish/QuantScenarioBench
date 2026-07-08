"""Portfolio concentration/diversification metrics (FR-43): Herfindahl-
Hirschman Index, Shannon entropy of weights, and effective number of assets.

All three read context.weights.weights (a PortfolioWeights, already
validated to sum to 1 with every entry >= 0, AD-20) and ignore
context.portfolio_returns/evaluation_returns entirely.

Written time-sequence-first: the definition is the time-average over the
weight sequence at each rebalance; today's single buy-and-hold
context.weights is the degenerate one-element case of that average, so
these formulas do not need to change once Epic 10's rebalancing lands a
weight sequence into MetricContext.

Range documentation (long-only, fully-invested weights over n assets):
HHI in [1/n, 1], entropy in [0, log(n)], ENB in [1, n].

See README "Metric Conventions" for the risk-free-rate, annualization,
and compounding conventions shared across this metrics package.

herfindahl_index/weight_entropy/effective_number_of_assets are opt-in
Metric instances — never added to DEFAULT_METRICS (AD-32).
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._context import MetricContext


def _herfindahl_index(weights: Float[Array, " n"]) -> Float[Array, ""]:
    return jnp.sum(weights ** 2)


class _HerfindahlIndex:
    """Herfindahl-Hirschman Index of the (time-averaged) weight sequence.

    HHI(w) = sum(w_j ** 2). Lower is better (more diversified); range
    [1/n, 1] for long-only fully-invested weights over n assets.
    """

    name = "herfindahl_index"
    direction = "lower_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        return _herfindahl_index(context.weights.weights)


class _WeightEntropy:
    """Shannon entropy of the (time-averaged) weight sequence.

    H(w) = -sum(w_j * log(w_j)), natural log, with the 0 * log(0) = 0
    convention applied jit-safely: the inner jnp.where(w > 0, w, 1.0)
    substitutes a safe value in log()'s argument for zero-weight
    components before the outer jnp.where selects 0.0 for that term, so
    log(0) never actually executes on the taken-or-not branch (mirroring
    _sharpe.py/_sortino.py's degenerate-guard style). Higher is better;
    range [0, log(n)] for n assets.
    """

    name = "weight_entropy"
    direction = "higher_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        w = context.weights.weights
        is_positive = w > 0
        safe_w = jnp.where(is_positive, w, 1.0)
        terms = jnp.where(is_positive, w * jnp.log(safe_w), 0.0)
        return -jnp.sum(terms)


class _EffectiveNumberOfAssets:
    """Effective number of assets: 1 / HHI(w) of the (time-averaged)
    weight sequence — the practitioner-facing transform of herfindahl_index
    (calls the same HHI computation, never re-derives sum(w ** 2)).

    Higher is better; range [1, n] for n assets.
    """

    name = "effective_number_of_assets"
    direction = "higher_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        return 1.0 / _herfindahl_index(context.weights.weights)


herfindahl_index = _HerfindahlIndex()
weight_entropy = _WeightEntropy()
effective_number_of_assets = _EffectiveNumberOfAssets()
