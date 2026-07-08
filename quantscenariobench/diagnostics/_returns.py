"""Ensemble-level return derivation for diagnostics (FR-49, AD-38).

Mirrors quantscenariobench.benchmark.returns.derive_returns' simple/
arithmetic return convention (return(t) = (price(t) - price(t-1)) /
price(t-1)), but operates on the *full* path ensemble
scenario.observation, shape (n_paths, T) — never a single selected
path. This module never calls derive_returns itself: that function is
scoped to a single 1-D path per Scenario's benchmark-layer contract
(AD-22), while diagnostics deliberately consume every simulated path
before any single-path selection, per this story's whole premise.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def ensemble_returns(observation: Float[Array, "n_paths T"]) -> Float[Array, "n_paths t"]:
    """Simple returns for every path in the ensemble, vectorized over the
    leading path axis via direct broadcasting (no per-path Python loop).
    """
    return (observation[:, 1:] - observation[:, :-1]) / observation[:, :-1]
