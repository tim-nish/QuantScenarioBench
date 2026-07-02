from __future__ import annotations

from typing import Callable, Sequence

from jaxtyping import Array, Float

from ._final_wealth_factor import final_wealth_factor
from ._max_drawdown import max_drawdown
from ._sharpe import sharpe_ratio
from ._sortino import sortino_ratio

# A Portfolio Return series in, a scalar out; every MetricFn also carries
# a .name: str attribute (AD-18).
MetricFn = Callable[[Float[Array, " t"]], Float[Array, ""]]

DEFAULT_METRICS: tuple[MetricFn, ...] = (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    final_wealth_factor,
)


def validate_metric_registry(metrics: Sequence[MetricFn]) -> None:
    """Raise if two entries in a metrics registry share a .name (AD-18).

    run_benchmark() calls this before iterating a caller-supplied metrics
    registry, so that one MetricFn can never silently shadow another.
    """
    seen: dict[str, MetricFn] = {}
    for metric in metrics:
        name = metric.name
        if name in seen:
            raise ValueError(
                f"Duplicate MetricFn name {name!r} in metrics registry "
                f"(AD-18 requires every MetricFn.name to be unique)"
            )
        seen[name] = metric
