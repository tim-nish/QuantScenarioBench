import warnings
from typing import Any

import jax.numpy as jnp

from ..interface import MarketModel, QuantScenarioBenchValidationWarning


class BlackScholes(MarketModel):
    """Geometric Brownian Motion market model.

    Implements the SDE: dS = mu*S dt + sigma*S dW (FR-7, AD-1, AD-6).
    No latent state — the full observation is the price path.
    """

    mu: float
    sigma: float
    S0: float

    def __check_init__(self) -> None:
        super().__check_init__()
        if self.sigma < 0:
            warnings.warn(
                f"BlackScholes: sigma={self.sigma!r} is negative; "
                "a valid Black-Scholes model requires sigma >= 0",
                QuantScenarioBenchValidationWarning,
                stacklevel=2,
            )

    def _drift(self, t: Any, state: Any) -> Any:
        return self.mu * state

    def _diffusion(self, t: Any, state: Any) -> Any:
        return self.sigma * state

    def initial_state(self) -> Any:
        return jnp.array(self.S0, dtype=float)
