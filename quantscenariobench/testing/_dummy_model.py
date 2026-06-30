"""Test-only MarketModel for the conformance suite (FR-11).

DummyModel is a trivial GBM-like model whose only purpose is to serve as
a second conforming implementation in conformance tests.  It must not be
exported from any non-testing module (FR-11, AD-9).
"""
import warnings
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..interface import MarketModel, QuantScenarioBenchValidationWarning


class DummyModel(MarketModel):
    """Minimal GBM-like model for conformance testing only.

    Declared research constraint: sigma >= 0.
    Constructing with sigma < 0 emits QuantScenarioBenchValidationWarning (FR-6).
    """

    alpha: float  # drift coefficient (dS = alpha*S dt + sigma*S dW)
    sigma: float  # diffusion coefficient
    S0: float     # initial state

    def __check_init__(self) -> None:
        super().__check_init__()
        if self.sigma < 0:
            warnings.warn(
                f"DummyModel: sigma={self.sigma!r} is negative; "
                "the research constraint requires sigma >= 0",
                QuantScenarioBenchValidationWarning,
                stacklevel=2,
            )

    def _drift(self, t: Any, state: Any) -> Any:
        return self.alpha * state

    def _diffusion(self, t: Any, state: Any) -> Any:
        return self.sigma * state

    def initial_state(self) -> Any:
        return jnp.array(self.S0, dtype=float)
