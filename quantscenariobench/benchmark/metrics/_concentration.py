"""Portfolio concentration/diversification metrics (FR-43): Herfindahl-
Hirschman Index, Shannon entropy of weights, and effective number of assets.

All three read context.weights (ignoring context.portfolio_returns/
evaluation_returns entirely) and are defined as the time-average over the
weight sequence at each rebalance (FR-44, AD-33): context.weights is
either a single PortfolioWeights (buy-and-hold, rebalance_schedule=None —
the average degenerates to that one value) or a Sequence[PortfolioWeights]
(one per rebalance date), and both shapes are handled identically here so
this definition did not need to change when Story 10.1 introduced the
weight sequence.

Range documentation (long-only, fully-invested weights over n assets):
HHI in [1/n, 1], entropy in [0, log(n)], ENB in [1, n].

See README "Metric Conventions" for the risk-free-rate, annualization,
and compounding conventions shared across this metrics package.

herfindahl_index/weight_entropy/effective_number_of_assets are opt-in
Metric instances — never added to DEFAULT_METRICS (AD-32).
"""
from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import PortfolioWeights
from ._context import MetricContext

WeightsOrSequence = Union[PortfolioWeights, Sequence[PortfolioWeights]]


def _weight_arrays(weights: WeightsOrSequence) -> list[Float[Array, " n"]]:
    """Normalize a single PortfolioWeights or a Sequence[PortfolioWeights]
    into a plain list of raw weight arrays, one per rebalance date (FR-44).
    """
    if isinstance(weights, PortfolioWeights):
        return [weights.weights]
    return [w.weights for w in weights]


def _herfindahl_index(weights: Float[Array, " n"]) -> Float[Array, ""]:
    return jnp.sum(weights ** 2)


def _weight_entropy(weights: Float[Array, " n"]) -> Float[Array, ""]:
    is_positive = weights > 0
    safe_weights = jnp.where(is_positive, weights, 1.0)
    terms = jnp.where(is_positive, weights * jnp.log(safe_weights), 0.0)
    return -jnp.sum(terms)


def _time_averaged_herfindahl_index(weights: WeightsOrSequence) -> Float[Array, ""]:
    arrays = _weight_arrays(weights)
    return jnp.mean(jnp.stack([_herfindahl_index(w) for w in arrays]))


class _HerfindahlIndex:
    """Time-average of the per-rebalance Herfindahl-Hirschman Index.

    HHI(w) = sum(w_j ** 2), averaged over the weight sequence (FR-44,
    AD-33) — on today's single buy-and-hold weight vector the average
    degenerates to that one value. Lower is better (more diversified);
    range [1/n, 1] for long-only fully-invested weights over n assets.
    """

    name = "herfindahl_index"
    direction = "lower_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        return _time_averaged_herfindahl_index(context.weights)


class _WeightEntropy:
    """Time-average of the per-rebalance Shannon entropy of weights.

    H(w) = -sum(w_j * log(w_j)), natural log, with the 0 * log(0) = 0
    convention applied jit-safely: the inner jnp.where(w > 0, w, 1.0)
    substitutes a safe value in log()'s argument for zero-weight
    components before the outer jnp.where selects 0.0 for that term, so
    log(0) never actually executes on the taken-or-not branch (mirroring
    _sharpe.py/_sortino.py's degenerate-guard style). Averaged over the
    weight sequence (FR-44, AD-33), degenerating to the single value on
    today's buy-and-hold weight vector. Higher is better; range
    [0, log(n)] for n assets.
    """

    name = "weight_entropy"
    direction = "higher_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        arrays = _weight_arrays(context.weights)
        return jnp.mean(jnp.stack([_weight_entropy(w) for w in arrays]))


class _EffectiveNumberOfAssets:
    """Effective number of assets: 1 / (time-averaged herfindahl_index) —
    the practitioner-facing transform of herfindahl_index (calls the same
    HHI computation, never re-derives sum(w ** 2)).

    Higher is better; range [1, n] for n assets.
    """

    name = "effective_number_of_assets"
    direction = "higher_is_better"
    params = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        return 1.0 / _time_averaged_herfindahl_index(context.weights)


herfindahl_index = _HerfindahlIndex()
weight_entropy = _WeightEntropy()
effective_number_of_assets = _EffectiveNumberOfAssets()
