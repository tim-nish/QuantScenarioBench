from __future__ import annotations

from typing import Sequence

from ._final_wealth_factor import final_wealth_factor
from ._max_drawdown import max_drawdown
from ._metric import Metric, MetricFn, wrap_legacy_metric
from ._sharpe import sharpe_ratio
from ._sortino import sortino_ratio

DEFAULT_METRICS: tuple[Metric, ...] = (
    wrap_legacy_metric(sharpe_ratio, direction="higher_is_better"),
    wrap_legacy_metric(sortino_ratio, direction="higher_is_better"),
    # max_drawdown reports a non-positive fraction (e.g. -0.2); higher
    # (less negative) is better.
    wrap_legacy_metric(max_drawdown, direction="higher_is_better"),
    wrap_legacy_metric(final_wealth_factor, direction="higher_is_better"),
)


def validate_metric_registry(metrics: Sequence[Metric | MetricFn]) -> None:
    """Raise if two entries in a metrics registry share a .name (AD-18).

    run_benchmark() calls this before iterating a caller-supplied metrics
    registry, so that one Metric/MetricFn can never silently shadow
    another. Works unchanged across a mixed legacy/native registry since
    both expose .name (AD-31).
    """
    seen: dict[str, Metric | MetricFn] = {}
    for metric in metrics:
        name = metric.name
        if name in seen:
            raise ValueError(
                f"Duplicate metric name {name!r} in metrics registry "
                f"(AD-18 requires every metric's .name to be unique)"
            )
        seen[name] = metric
