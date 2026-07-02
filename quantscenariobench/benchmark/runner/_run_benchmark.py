"""run_benchmark() — the single entry point for the benchmark pipeline (FR-27).

Mirrors simulate()'s architectural posture: strategy behaviour is
dispatched exclusively through the Portfolio Optimizer Interface
(BaselineStrategy/ForecastOptimizer.allocate), with no strategy-specific
branching beyond the isinstance(ForecastOptimizer) check AD-23 requires
to pick allocate()'s one- vs two-argument shape.
"""
from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
from typing import Any, Optional, Sequence

from jaxtyping import Array, Float

from ..interface import BaselineStrategy, BenchmarkResult, ForecastOptimizer
from ..metrics import DEFAULT_METRICS, MetricFn, validate_metric_registry


def _library_version() -> str:
    try:
        return importlib.metadata.version("quantscenariobench")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _strategy_parameters(strategy: Any) -> dict:
    """JSON-native snapshot of a strategy's own recorded fields (AD-15, AD-24)."""
    params: dict = {}
    for field in dataclasses.fields(strategy):
        value = getattr(strategy, field.name)
        if isinstance(value, (bool, int, float, str)):
            params[field.name] = value
        else:
            params[field.name] = float(value)
    return params


def run_benchmark(
    strategy: BaselineStrategy | ForecastOptimizer,
    historical_returns: Float[Array, "t n"],
    evaluation_returns: Float[Array, "t2 n"],
    *,
    forecast: Optional[Float[Array, " n"]] = None,
    metrics: Sequence[MetricFn] = DEFAULT_METRICS,
    asset_scenario_ids: Sequence[str] = (),
    time_grid_reference: str = "",
) -> BenchmarkResult:
    """Run the full returns -> strategy -> weights -> portfolio returns ->
    metrics -> BenchmarkResult pipeline for a single strategy (FR-27).

    strategy is fit exactly once, on historical_returns; the resulting
    PortfolioWeights are applied unchanged (buy-and-hold) across the full
    evaluation_returns window — no intra-run refitting or rebalancing
    (AD-23).

    Dispatch (AD-23): a ForecastOptimizer requires forecast and is called
    as allocate(historical_returns, forecast); a BaselineStrategy rejects
    a caller-supplied forecast and is called as allocate(historical_returns).
    """
    if isinstance(strategy, ForecastOptimizer):
        if forecast is None:
            raise ValueError(
                "run_benchmark requires a forecast argument when strategy "
                "is a ForecastOptimizer"
            )
        weights = strategy.allocate(historical_returns, forecast)
    else:
        if forecast is not None:
            raise ValueError(
                "run_benchmark does not accept a forecast argument when "
                "strategy is a BaselineStrategy"
            )
        weights = strategy.allocate(historical_returns)

    validate_metric_registry(metrics)

    portfolio_returns = evaluation_returns @ weights.weights
    metrics_result = {metric.name: float(metric(portfolio_returns)) for metric in metrics}

    return BenchmarkResult(
        strategy_name=type(strategy).__name__,
        strategy_parameters=_strategy_parameters(strategy),
        metrics=metrics_result,
        asset_scenario_ids=list(asset_scenario_ids),
        time_grid_reference=time_grid_reference,
        library_version=_library_version(),
        generated_at=_utc_now(),
    )
