"""Heston stochastic volatility model (FR-8, AD-1, AD-6, AD-9)."""
import warnings
from typing import Any

import jax.numpy as jnp

from ..interface import MarketModel, QuantScenarioBenchValidationWarning


class Heston(MarketModel):
    """Heston (1993) stochastic volatility model.

    SDE system under risk-neutral or physical measure:
      dS  = mu * S dt + sqrt(v) * S dW_S
      dv  = kappa * (theta - v) dt + xi * sqrt(v) dW_v
      Corr(dW_S, dW_v) = rho

    State vector: [S, v]  (observation = S paths; latent_state = v paths)

    Feller condition for variance to stay strictly positive: 2*kappa*theta >= xi^2.
    Violation emits QuantScenarioBenchValidationWarning (FR-6).
    """

    mu: float     # drift of asset price
    kappa: float  # mean-reversion speed of variance
    theta: float  # long-run variance
    xi: float     # vol-of-vol
    rho: float    # correlation between asset and variance Brownian motions
    v0: float     # initial variance
    S0: float     # initial asset price

    def __check_init__(self) -> None:
        super().__check_init__()
        if 2.0 * self.kappa * self.theta < self.xi ** 2:
            warnings.warn(
                f"Heston: Feller condition violated "
                f"(2*kappa*theta={2*self.kappa*self.theta:.4g} < xi^2={self.xi**2:.4g}). "
                "The variance process may reach zero.",
                QuantScenarioBenchValidationWarning,
                stacklevel=2,
            )

    def _drift(self, t: Any, state: Any) -> Any:
        S, v = state[0], state[1]
        return jnp.array([
            self.mu * S,
            self.kappa * (self.theta - v),
        ])

    def _diffusion(self, t: Any, state: Any) -> Any:
        S, v = state[0], state[1]
        sv = jnp.sqrt(jnp.maximum(v, 0.0))  # floor at zero for numerical stability
        rho_perp = jnp.sqrt(jnp.maximum(1.0 - self.rho ** 2, 0.0))
        # Cholesky factor (lower-triangular) of the correlated diffusion:
        #   dS = sqrt(v)*S * dZ_1
        #   dv = xi*sqrt(v) * (rho*dZ_1 + sqrt(1-rho^2)*dZ_2)
        return jnp.array([
            [sv * S,               0.0],
            [self.rho * self.xi * sv,  rho_perp * self.xi * sv],
        ])

    def initial_state(self) -> Any:
        return jnp.array([self.S0, self.v0], dtype=float)

    def split_state(self, ys: Any) -> tuple[Any, Any]:
        """Split [S, v] paths: observation=S paths, latent_state=v paths."""
        return ys[:, :, 0], ys[:, :, 1]
