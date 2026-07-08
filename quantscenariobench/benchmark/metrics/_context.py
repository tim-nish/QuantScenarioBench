from __future__ import annotations

import dataclasses
from typing import Any

from jaxtyping import Array, Float

from ..interface import PortfolioWeights


@dataclasses.dataclass(frozen=True)
class MetricContext:
    """Per-call scaffolding passed to every Metric (FR-40, AD-31).

    A plain frozen dataclass, not an equinox.Module — the same posture
    AD-17 fixes for BenchmarkResult, since a MetricContext is constructed
    exactly once inside run_benchmark() and consumed immediately, never
    re-traced through jit/vmap itself (individual Metric bodies remain
    jax.numpy-native on the arrays it carries).
    """

    portfolio_returns: Float[Array, " t"]
    weights: PortfolioWeights
    evaluation_returns: Float[Array, "t n"]
    auxiliary: dict[str, Any] = dataclasses.field(default_factory=dict)
