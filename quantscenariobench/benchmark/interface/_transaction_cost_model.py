from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclasses.dataclass(frozen=True)
class ProportionalCost:
    """Proportional transaction-cost model (FR-45, AD-34).

    A plain immutable dataclass, not an equinox.Module — the same
    JSON-native posture Story 10.1's RebalanceSchedule established
    (AD-17-style), since this is a terminal, serializable declaration of
    protocol, not a traced pytree.

    Cost of a rebalance trading from w_drifted (Story 10.1's pre-trade
    drifted weight — the effective weight right before the new target is
    set, not a re-normalized PortfolioWeights) to w_target (the freshly
    allocated weight) is (one_way_bps / 1e4) * sum(|w_target - w_drifted|)
    — one-way, per dollar traded: buy and sell legs of the same rebalance
    are not double-counted.

    one_way_bps is a required constructor argument (mirrors
    CVaROptimization.confidence_level's AD-15 pattern) — never an
    internal hardcoded default. one_way_bps=0 is a genuine, valid
    configuration whose net return series is exactly the gross series
    (AC2) — distinct from cost_model=None, which skips cost computation
    entirely and is what guarantees run_benchmark()'s bit-for-bit legacy
    behavior (AC1).
    """

    one_way_bps: float

    def cost(
        self, w_target: Float[Array, " n"], w_drifted: Float[Array, " n"]
    ) -> Float[Array, ""]:
        return (self.one_way_bps / 1e4) * jnp.sum(jnp.abs(w_target - w_drifted))
