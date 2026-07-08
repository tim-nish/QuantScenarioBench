from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RebalanceSchedule:
    """Periodic rebalancing schedule for run_benchmark() (FR-44, AD-33).

    A plain immutable dataclass, not an equinox.Module — the same
    JSON-native posture AD-17 fixes for BenchmarkResult, since this is a
    terminal, serializable declaration of protocol, not a traced pytree.

    k=None (the default) is buy-and-hold: strategy.allocate() is called
    exactly once on historical_returns, and the resulting weights are
    applied unchanged across the full evaluation window — today's only
    behavior, and the exact code path run_benchmark() still executes,
    byte-for-byte, when rebalance_schedule is None or k is None (AC1).

    k=<int> refits the strategy every k evaluation steps; see
    run_benchmark()'s docstring for the weight-drift convention applied
    to the realized portfolio-return series between rebalances (AD-33).
    """

    k: int | None = None
