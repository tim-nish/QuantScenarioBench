from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

from jaxtyping import Array, Float

from ._context import MetricContext

Direction = Literal["higher_is_better", "lower_is_better"]

# The pre-Story-9.1 metric contract: a Portfolio Return series in, a
# scalar out, plus a .name: str attribute (AD-18). Still accepted
# anywhere a Metric is expected, via wrap_legacy_metric.
MetricFn = Callable[[Float[Array, " t"]], Float[Array, ""]]


@runtime_checkable
class Metric(Protocol):
    """Context-aware metric contract (FR-40, AD-31).

    Coexists with the legacy MetricFn: any object exposing name,
    direction, params, and __call__(context) satisfies this protocol
    structurally, with no required inheritance.
    """

    name: str
    direction: Direction
    params: dict[str, float] | None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        ...


class _LegacyMetricAdapter:
    """Adapts a legacy MetricFn callable into the Metric protocol (FR-40, AD-31).

    __call__ forwards context.portfolio_returns to the wrapped fn
    unchanged, so wrapped values stay bit-identical to calling fn
    directly (AC4); direction is supplied by the caller at wrap time
    since legacy MetricFn carries no such metadata (AC5).
    """

    def __init__(self, fn: MetricFn, *, direction: Direction) -> None:
        self._fn = fn
        self.name = fn.name
        self.direction: Direction = direction
        self.params: dict[str, float] | None = None

    def __call__(self, context: MetricContext) -> Float[Array, ""]:
        return self._fn(context.portfolio_returns)


def wrap_legacy_metric(fn: MetricFn, *, direction: Direction) -> Metric:
    """Adapt a legacy bare MetricFn callable into the Metric protocol (AC5)."""
    return _LegacyMetricAdapter(fn, direction=direction)
