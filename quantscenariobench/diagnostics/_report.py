"""RealismReport aggregator (FR-49, AD-38): the public entry point that
turns a Scenario's full path ensemble into a stylized-facts realism
report, scored against literature reference bands.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..interface import Scenario
from ._autocorrelation import LAGS, acf_at_lag, leverage_correlation_per_path
from ._bands import (
    EXCESS_KURTOSIS_BAND,
    KURTOSIS_DECAY_BAND,
    LEVERAGE_CORRELATION_BAND,
    MONTHLY_EXCESS_KURTOSIS_BAND,
    RETURN_ACF_BAND,
    SQUARED_RETURN_ACF_BANDS,
    is_in_band,
)
from ._kurtosis import excess_kurtosis_per_path, kurtosis_decay_per_path
from ._returns import ensemble_returns

_STAT_FIELD_NAMES = (
    "excess_kurtosis",
    "return_acf_lag1",
    "return_acf_lag5",
    "return_acf_lag21",
    "squared_return_acf_lag1",
    "squared_return_acf_lag5",
    "squared_return_acf_lag21",
    "leverage_correlation",
    "monthly_excess_kurtosis",
    "kurtosis_decay",
)


@dataclasses.dataclass(frozen=True)
class DiagnosticStat:
    """One diagnostic's cross-path summary (FR-49): the per-path
    statistic's mean and std across the Scenario's full path ensemble,
    plus its literature reference band and in-band/out-of-band flag.

    A plain immutable dataclass — never an equinox.Module — matching
    BenchmarkResult's AD-17 house style, since this is a terminal,
    JSON-serializable artifact, not a traced pytree.
    """

    mean: float
    std: float
    in_band: bool
    reference_low: float
    reference_high: float


@dataclasses.dataclass(frozen=True)
class RealismReport:
    """A Scenario's stylized-facts realism report (FR-49, AD-38).

    One field per diagnostic (each a DiagnosticStat: cross-path mean and
    std, plus its reference band and in-band flag), plus generation
    provenance reused directly from scenario.metadata — never a parallel
    provenance scheme. In-band/out-of-band flags are informational only:
    realism_report() always returns a complete report regardless of how
    many diagnostics are out-of-band (AD-38) — a Black-Scholes scenario
    failing the volatility-clustering band is a reported finding, not an
    error.
    """

    model_name: str
    seed: int
    generated_at: str
    n_paths: int
    excess_kurtosis: DiagnosticStat
    return_acf_lag1: DiagnosticStat
    return_acf_lag5: DiagnosticStat
    return_acf_lag21: DiagnosticStat
    squared_return_acf_lag1: DiagnosticStat
    squared_return_acf_lag5: DiagnosticStat
    squared_return_acf_lag21: DiagnosticStat
    leverage_correlation: DiagnosticStat
    monthly_excess_kurtosis: DiagnosticStat
    kurtosis_decay: DiagnosticStat

    @classmethod
    def from_dict(cls, data: dict) -> "RealismReport":
        """Reconstruct a RealismReport from its dataclasses.asdict() shape
        (the exact inverse of json.loads(json.dumps(dataclasses.asdict(report)))),
        mirroring EvaluationResult.from_dict()'s nested-record reconstruction.
        """
        kwargs = dict(data)
        for name in _STAT_FIELD_NAMES:
            kwargs[name] = DiagnosticStat(**kwargs[name])
        return cls(**kwargs)


def _stat(values: Float[Array, " n_paths"], band: tuple) -> DiagnosticStat:
    mean = float(jnp.mean(values))
    std = float(jnp.std(values))
    return DiagnosticStat(
        mean=mean, std=std, in_band=is_in_band(mean, band),
        reference_low=band[0], reference_high=band[1],
    )


def realism_report(scenario: Scenario) -> RealismReport:
    """Compute a stylized-facts realism report for scenario (FR-49).

    Every diagnostic is computed once per path (vectorized over the
    n_paths axis via broadcasting, no per-path Python loop, AC7) then
    aggregated to a cross-path mean/std — a pure function of scenario,
    consistent with the rest of the library's determinism/jit posture
    (FR-E).
    """
    returns = ensemble_returns(scenario.observation)
    squared_returns = returns ** 2

    kurtosis = excess_kurtosis_per_path(returns)
    return_acfs = {lag: acf_at_lag(returns, lag) for lag in LAGS}
    squared_acfs = {lag: acf_at_lag(squared_returns, lag) for lag in LAGS}
    leverage = leverage_correlation_per_path(returns)
    monthly_kurtosis, decay = kurtosis_decay_per_path(returns)

    metadata = scenario.metadata
    return RealismReport(
        model_name=metadata.model_name,
        seed=metadata.seed,
        generated_at=metadata.generated_at,
        n_paths=metadata.n_paths,
        excess_kurtosis=_stat(kurtosis, EXCESS_KURTOSIS_BAND),
        return_acf_lag1=_stat(return_acfs[1], RETURN_ACF_BAND),
        return_acf_lag5=_stat(return_acfs[5], RETURN_ACF_BAND),
        return_acf_lag21=_stat(return_acfs[21], RETURN_ACF_BAND),
        squared_return_acf_lag1=_stat(squared_acfs[1], SQUARED_RETURN_ACF_BANDS[1]),
        squared_return_acf_lag5=_stat(squared_acfs[5], SQUARED_RETURN_ACF_BANDS[5]),
        squared_return_acf_lag21=_stat(squared_acfs[21], SQUARED_RETURN_ACF_BANDS[21]),
        leverage_correlation=_stat(leverage, LEVERAGE_CORRELATION_BAND),
        monthly_excess_kurtosis=_stat(monthly_kurtosis, MONTHLY_EXCESS_KURTOSIS_BAND),
        kurtosis_decay=_stat(decay, KURTOSIS_DECAY_BAND),
    )
