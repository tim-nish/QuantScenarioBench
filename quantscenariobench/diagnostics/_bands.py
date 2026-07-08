"""Literature reference bands for scenario-realism diagnostics (FR-49, AD-38).

Every band below is a documented constant sourced from the qualitative
stylized facts described in:

    Cont, R. (2001). "Empirical properties of asset returns: stylized
    facts and statistical issues." Quantitative Finance, 1(2), 223-236.

cited by arXiv:2510.03129 as the motivating description of real market
behavior. Cont (2001) documents these properties qualitatively across
many markets/instruments rather than a single universal numeric range;
the bands below are illustrative representative ranges consistent with
that literature (not a single precise quoted statistic), used only to
flag a computed diagnostic in-band/out-of-band — never to reject or
filter a scenario (AD-38): a Black-Scholes scenario failing the
volatility-clustering band is a correct, reported finding, not an error.

No band constant here is fetched or computed from external/live data at
runtime.
"""
from __future__ import annotations

# Heavy tails: daily equity log-returns are strongly leptokurtic (Cont
# 2001, stylized fact 1); GBM's Gaussian log-returns (excess kurtosis 0)
# fall below this band by construction.
EXCESS_KURTOSIS_BAND = (0.5, 20.0)

# Absence of linear autocorrelation (Cont 2001, stylized fact 2): raw
# returns show no significant linear autocorrelation at any lag,
# including short ones. Applied identically at lags 1, 5, 21.
RETURN_ACF_BAND = (-0.05, 0.05)

# Volatility clustering (Cont 2001, stylized fact 3): squared/absolute
# returns show slowly-decaying positive autocorrelation over many lags.
SQUARED_RETURN_ACF_BANDS = {1: (0.03, 0.5), 5: (0.02, 0.4), 21: (0.01, 0.3)}

# Leverage effect (Cont 2001, stylized fact 4): most equity-index
# returns show negative correlation between returns and subsequent
# volatility.
LEVERAGE_CORRELATION_BAND = (-0.5, -0.01)

# Aggregational Gaussianity (Cont 2001, stylized fact 5): temporally
# aggregated (e.g. monthly) returns are markedly less leptokurtic than
# daily returns, though not necessarily fully Gaussian.
MONTHLY_EXCESS_KURTOSIS_BAND = (-1.0, 5.0)
KURTOSIS_DECAY_BAND = (0.1, 15.0)


def is_in_band(value: float, band: tuple) -> bool:
    low, high = band
    return low <= value <= high
