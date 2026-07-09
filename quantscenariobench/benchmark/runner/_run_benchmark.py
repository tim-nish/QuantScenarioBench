"""run_benchmark() — the single entry point for the benchmark pipeline (FR-27, FR-44).

Mirrors simulate()'s architectural posture: strategy behaviour is
dispatched exclusively through the Portfolio Optimizer Interface
(BaselineStrategy/ForecastOptimizer/PolicyStrategy.allocate*), with no
strategy-specific branching beyond the isinstance(ForecastOptimizer)
check AD-23 requires to pick allocate()'s one- vs two-argument shape.
The rebalance-schedule-aware rolling loop (AD-33) is a separate function,
_run_rebalancing_loop, so the rebalance_schedule=None path above executes
byte-for-byte unchanged (AC1) with zero new branching in run_benchmark's
own body.
"""
from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
from typing import Any, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import (
    BaselineStrategy,
    BenchmarkResult,
    ForecastOptimizer,
    PolicyStrategy,
    PortfolioWeights,
    ProportionalCost,
    RebalanceSchedule,
)
from ..metrics import DEFAULT_METRICS, Metric, MetricContext, MetricFn, validate_metric_registry


def _library_version() -> str:
    try:
        return importlib.metadata.version("quantscenariobench")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _serialize_parameter(value: Any) -> Any:
    """One strategy field → a JSON-native value (issue #99).

    Primitives and None pass through unchanged; lists/tuples/dicts are
    serialized recursively. Anything else first keeps the historical
    float() conversion (scalar JAX/NumPy arrays), then falls back to
    repr() — so a strategy holding, say, a neural-network module as a
    field serializes to an identifying string instead of raising. The
    broad except is deliberate: __float__ on arbitrary objects can raise
    anything, and serialization must never fail a completed run.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_parameter(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_parameter(item) for key, item in value.items()}
    try:
        return float(value)
    except Exception:
        return repr(value)


def _strategy_parameters(strategy: Any) -> dict:
    """JSON-native snapshot of a strategy's own recorded fields (AD-15, AD-24)."""
    return {
        field.name: _serialize_parameter(getattr(strategy, field.name))
        for field in dataclasses.fields(strategy)
    }


def _run_rebalancing_loop(
    strategy: BaselineStrategy | ForecastOptimizer | PolicyStrategy,
    historical_returns: Float[Array, "t n"],
    evaluation_returns: Float[Array, "t2 n"],
    forecast: Optional[Float[Array, " n"]],
    k: int,
    cost_model: Optional[ProportionalCost],
) -> Tuple[Float[Array, " t2"], List[PortfolioWeights], List[Float[Array, " n"]]]:
    """The periodic rebalancing loop (AD-33, FR-44) — a Python for loop,
    not lax.scan, since some BaselineStrategy refits call into the
    scipy-backed Optimizer Solver Layer, already outside jit today
    (AD-25, AD-14); this is the documented fallback AC8 accepts.

    Interface-level dispatch — PolicyStrategy.allocate_sequence vs
    ForecastOptimizer/BaselineStrategy.allocate — lives entirely in this
    function, never inside run_benchmark's own body, so run_benchmark's
    rebalance_schedule=None path keeps exactly one isinstance check
    (AD-19/FR-27's no-strategy-specific-branching invariant).

    No-lookahead invariant (AC3): at rebalance date t_i, the strategy
    sees only historical_returns concatenated with evaluation_returns[:t_i]
    — returns strictly before t_i, never evaluation_returns[t_i] itself.

    Weight-drift convention (AC4, AD-33 — chosen explicitly over the
    simpler reset-every-step alternative): entering weights at t_i are
    held as per-asset dollar allocations, not reset to their target
    fraction every step, so the effective weight drifts with each
    asset's relative performance within the holding period
    [t_i, t_{i+1}) until the next rebalance resets it.

    Transaction costs (FR-45, AD-34): at every rebalance after the first,
    the trade is w_target (the freshly allocated weight) minus w_drifted
    (the pre-trade drifted weight — the effective weight at the end of
    the previous holding period). When cost_model is set, that
    rebalance's cost is deducted directly from the gross portfolio return
    realized on the rebalance day, producing the net series every Metric
    then scores unchanged — the cost deduction happens once, here, never
    inside any Metric/MetricFn. The first rebalance (t_0) never carries a
    cost: there is no prior position to trade out of. drifted_weights
    (returned alongside weight_sequence) is the same list the turnover
    Metric reads via MetricContext.auxiliary — one entry per rebalance
    after the first, aligned with weight_sequence[1:].
    """
    if isinstance(strategy, PolicyStrategy):
        if forecast is not None:
            raise ValueError(
                "run_benchmark does not accept a forecast argument when "
                "strategy is a PolicyStrategy"
            )

        def allocate(returns_so_far: Float[Array, "t n"]) -> PortfolioWeights:
            return strategy.allocate_sequence(returns_so_far)

    elif isinstance(strategy, ForecastOptimizer):
        if forecast is None:
            raise ValueError(
                "run_benchmark requires a forecast argument when strategy "
                "is a ForecastOptimizer"
            )

        def allocate(returns_so_far: Float[Array, "t n"]) -> PortfolioWeights:
            return strategy.allocate(returns_so_far, forecast)

    else:
        if forecast is not None:
            raise ValueError(
                "run_benchmark does not accept a forecast argument when "
                "strategy is a BaselineStrategy"
            )

        def allocate(returns_so_far: Float[Array, "t n"]) -> PortfolioWeights:
            return strategy.allocate(returns_so_far)

    t2 = evaluation_returns.shape[0]
    rebalance_starts = list(range(0, t2, k))

    portfolio_return_segments = []
    weight_sequence: List[PortfolioWeights] = []
    drifted_weights: List[Float[Array, " n"]] = []
    rebalance_costs: List[Tuple[int, Float[Array, ""]]] = []
    previous_drifted: Optional[Float[Array, " n"]] = None

    for idx, t_i in enumerate(rebalance_starts):
        t_next = rebalance_starts[idx + 1] if idx + 1 < len(rebalance_starts) else t2

        returns_so_far = jnp.concatenate(
            [historical_returns, evaluation_returns[:t_i]], axis=0
        )
        weights = allocate(returns_so_far)

        if previous_drifted is not None:
            drifted_weights.append(previous_drifted)
            if cost_model is not None:
                rebalance_costs.append(
                    (t_i, cost_model.cost(weights.weights, previous_drifted))
                )

        weight_sequence.append(weights)

        period_returns = evaluation_returns[t_i:t_next]
        cumulative_asset_growth = jnp.cumprod(1.0 + period_returns, axis=0)
        asset_wealth = weights.weights[None, :] * cumulative_asset_growth
        portfolio_wealth = jnp.sum(asset_wealth, axis=1)
        portfolio_wealth_prev = jnp.concatenate(
            [jnp.ones((1,)), portfolio_wealth[:-1]]
        )
        portfolio_return_segments.append(portfolio_wealth / portfolio_wealth_prev - 1.0)

        previous_drifted = asset_wealth[-1] / portfolio_wealth[-1]

    portfolio_returns = jnp.concatenate(portfolio_return_segments)

    if rebalance_costs:
        cost_indices = jnp.array([index for index, _ in rebalance_costs])
        cost_values = jnp.stack([cost for _, cost in rebalance_costs])
        portfolio_returns = portfolio_returns.at[cost_indices].add(-cost_values)

    return portfolio_returns, weight_sequence, drifted_weights


def run_benchmark(
    strategy: BaselineStrategy | ForecastOptimizer | PolicyStrategy,
    historical_returns: Float[Array, "t n"],
    evaluation_returns: Float[Array, "t2 n"],
    *,
    forecast: Optional[Float[Array, " n"]] = None,
    metrics: Sequence[Metric | MetricFn] = DEFAULT_METRICS,
    asset_scenario_ids: Sequence[str] = (),
    time_grid_reference: str = "",
    rebalance_schedule: Optional[RebalanceSchedule] = None,
    cost_model: Optional[ProportionalCost] = None,
) -> BenchmarkResult:
    """Run the full returns -> strategy -> weights -> portfolio returns ->
    metrics -> BenchmarkResult pipeline for a single strategy (FR-27, FR-44).

    rebalance_schedule=None (the default) — buy-and-hold, unchanged from
    before this argument existed: strategy is fit exactly once, on
    historical_returns; the resulting PortfolioWeights are applied
    unchanged across the full evaluation_returns window — no intra-run
    refitting or rebalancing (AD-23). This branch is byte-for-byte
    identical to the pre-Story-10.1 implementation whenever
    rebalance_schedule is None or rebalance_schedule.k is None, so every
    previously published BenchmarkResult stays reproducible (AC1, AD-33).
    cost_model has no effect on this path: buy-and-hold has no rebalance
    trades to cost.

    rebalance_schedule.k=<int> (AD-33, FR-44) — periodic rebalancing via
    _run_rebalancing_loop: the strategy is refit every k evaluation steps
    at rebalance dates t_0=0, t_1=k, t_2=2k, .... Each refit sees only
    returns strictly before its rebalance date (a tested no-lookahead
    invariant, AC3). Between rebalances, the realized portfolio-return
    series follows the weight-drift convention (AD-33): entering weights
    evolve with each asset's relative performance until the next
    rebalance resets them — the literature-default convention, chosen
    explicitly over the simpler reset-every-step alternative (AC4).

    cost_model=None (the default, FR-45, AD-34) skips cost computation
    entirely — required for bit-for-bit legacy behavior (AC1); a
    ProportionalCost(one_way_bps), including one_way_bps=0, nets each
    rebalance's transaction cost out of that rebalance day's portfolio
    return, scored by every Metric with zero metric-side changes.
    Sensitivity sweeps across bps are a plain loop over this argument:
    `for bps in (0, 5, 10): run_benchmark(strategy, ..., rebalance_schedule=RebalanceSchedule(k=21), cost_model=ProportionalCost(bps))`.

    Dispatch (AD-23, AD-33): a ForecastOptimizer requires forecast and is
    called as allocate(historical_returns, forecast); a BaselineStrategy
    rejects a caller-supplied forecast and is called as
    allocate(historical_returns); a PolicyStrategy is called once per
    rebalance date as allocate_sequence(observed_returns) and requires
    rebalance_schedule.k to be set — it has no single-shot allocate().
    """
    if rebalance_schedule is None or rebalance_schedule.k is None:
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
        context = MetricContext(
            portfolio_returns=portfolio_returns,
            weights=weights,
            evaluation_returns=evaluation_returns,
        )
    else:
        validate_metric_registry(metrics)

        portfolio_returns, weight_sequence, drifted_weights = _run_rebalancing_loop(
            strategy, historical_returns, evaluation_returns, forecast,
            rebalance_schedule.k, cost_model,
        )
        context = MetricContext(
            portfolio_returns=portfolio_returns,
            weights=weight_sequence,
            evaluation_returns=evaluation_returns,
            auxiliary={"drifted_weights": drifted_weights},
        )

    metrics_result = {metric.name: float(metric(context)) for metric in metrics}

    return BenchmarkResult(
        strategy_name=type(strategy).__name__,
        strategy_parameters=_strategy_parameters(strategy),
        metrics=metrics_result,
        asset_scenario_ids=list(asset_scenario_ids),
        time_grid_reference=time_grid_reference,
        library_version=_library_version(),
        generated_at=_utc_now(),
        rebalance_schedule=(
            dataclasses.asdict(rebalance_schedule) if rebalance_schedule is not None else None
        ),
        cost_model=(
            {"model": type(cost_model).__name__, **dataclasses.asdict(cost_model)}
            if cost_model is not None else None
        ),
    )
