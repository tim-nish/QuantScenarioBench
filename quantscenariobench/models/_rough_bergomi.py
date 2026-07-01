"""Rough Bergomi (rBergomi) stochastic volatility model (FR-9, AD-1, AD-6, AD-9).

Reference: Bayer, Friz, Gatheral (2016) "Pricing Under Rough Volatility".
"""
from typing import Any

import jax
import jax.numpy as jnp

from ..interface import MarketModel


class RoughBergomi(MarketModel):
    """Rough Bergomi model with fractional Brownian motion volatility.

    The variance process is driven by fractional BM with Hurst exponent H:
      V_t  = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})
      dS_t = mu * S_t dt + sqrt(V_t) * S_t dW^S_t
      Corr(dW^S_t, dW_t) = rho   (W_t drives the Volterra integral for W^H)

    The Riemann-Liouville fBM is discretised via the Volterra representation:
      W^H_{t_i} = sum_{j=0}^{i-1} (t_i - t_j)^{H - 1/2} * delta_W_j

    where delta_W_j = W_{t_{j+1}} - W_{t_j}.

    Because the variance is non-Markovian, this model overrides _generate_paths
    to pre-compute the full Volterra kernel and generate paths via a single
    matrix-vector product per path (no iterative Euler-Maruyama on variance).

    Parameters
    ----------
    H : float
        Hurst exponent, H in (0, 1). Rough volatility corresponds to H in (0, 0.5).
    eta : float
        Vol-of-vol (controls the amplitude of variance fluctuations).
    rho : float
        Correlation between the asset Brownian motion and the fBM driver.
    xi0 : float
        Initial variance level (V_0 = xi0).
    S0 : float
        Initial asset price.
    mu : float
        Drift of the asset price (use 0 for risk-neutral, r=0 setting).
    """

    H: float
    eta: float
    rho: float
    xi0: float
    S0: float
    mu: float

    def _drift(self, t: Any, state: Any) -> Any:
        """Local drift for state [S, V].

        Under rBergomi, V_t is driven by fBM (non-Markovian); its drift in the
        standard Itô sense is zero. The asset drift is mu * S.
        """
        S, _V = state[0], state[1]
        return jnp.array([self.mu * S, 0.0])

    def _diffusion(self, t: Any, state: Any) -> Any:
        """Local diffusion for state [S, V] using the discretised Volterra kernel.

        The instantaneous diffusion of V is approximated by the Volterra kernel
        K(t, 0) = t^{H - 1/2} evaluated at lag t (the Molchan-Golosov form).
        Row 0: asset price driven by correlated and perpendicular BM.
        Row 1: variance driven by fBM; local approximation shows the H exponent.
        """
        S, V = state[0], state[1]
        sigma = jnp.sqrt(jnp.maximum(V, 0.0))
        rho_perp = jnp.sqrt(jnp.maximum(1.0 - self.rho ** 2, 0.0))
        # Volterra kernel at lag t: K(t, 0) = t^{H - 1/2}
        k_t = jnp.where(t > 0.0, t ** (self.H - 0.5), 0.0)
        return jnp.array([
            [sigma * S * self.rho,        sigma * S * rho_perp],
            [self.eta * V * k_t,          0.0],
        ])

    def initial_state(self) -> Any:
        return jnp.array([self.S0, self.xi0], dtype=jnp.float64)

    def split_state(self, ys: Any) -> tuple[Any, Any]:
        """Split [S, V] paths: observation=S paths, latent_state=V paths."""
        return ys[:, :, 0], ys[:, :, 1]

    def _generate_paths(self, ts: Any, n_paths: int, key: Any) -> Any:
        """Generate rBergomi paths via discretised Volterra integral (FR-9, NFR-2).

        Algorithm:
        1. Precompute the (T-1) x (T-1) Volterra kernel matrix K where
           K[i, j] = (ts[i+1] - ts[j])^{H - 1/2} for j <= i, else 0.
        2. For each path:
           a. Sample z_fBM ~ N(0, 1) and z_perp ~ N(0, 1), both shape (T-1,).
           b. Compute BM increments dW = sqrt(dt) * z_fBM.
           c. Compute fBM path W^H = K @ dW  (shape T-1), prepend 0.
           d. Compute variance path V_t = xi0 * exp(eta*W^H_t - 0.5*eta^2*t^{2H}).
           e. Euler-Maruyama on asset: dS = mu*S*dt + sqrt(V)*S*dW_S.
           f. Return stacked [S, V], shape (T, 2).
        """
        T = ts.shape[0]
        dt = jnp.diff(ts)           # (T-1,)
        sqrt_dt = jnp.sqrt(dt)      # (T-1,)

        # Volterra kernel: K[i, j] = (ts[i+1] - ts[j])^{H-0.5} for j <= i
        time_diff = ts[1:][:, None] - ts[:-1][None, :]   # (T-1, T-1)
        # Mask upper triangle (j > i → time_diff <= 0) to zero.
        # Use safe_diff = 1.0 for masked entries to avoid NaN from negative bases.
        safe_diff = jnp.where(time_diff > 0.0, time_diff, 1.0)
        K = jnp.where(time_diff > 0.0, safe_diff ** (self.H - 0.5), 0.0)

        rho_perp = jnp.sqrt(jnp.maximum(1.0 - self.rho ** 2, 0.0))
        t_2H = ts ** (2.0 * self.H)   # (T,) — precompute t^{2H}

        def _one_path(path_key: Any) -> Any:
            key_fBM, key_perp = jax.random.split(path_key)
            z_fBM  = jax.random.normal(key_fBM,  shape=(T - 1,))
            z_perp = jax.random.normal(key_perp, shape=(T - 1,))

            dW = sqrt_dt * z_fBM    # BM increments driving fBM

            # Fractional BM via Volterra convolution (NFR-2)
            W_H_steps = K @ dW      # (T-1,): W^H at t_1,...,t_{T-1}
            W_H = jnp.concatenate([jnp.zeros((1,)), W_H_steps])  # (T,)

            # Variance path: V_t = xi0 * exp(eta*W^H_t - 0.5*eta^2*t^{2H})
            V = self.xi0 * jnp.exp(self.eta * W_H - 0.5 * self.eta ** 2 * t_2H)

            # Correlated BM for asset price
            dW_S = self.rho * dW + rho_perp * (sqrt_dt * z_perp)

            # Euler scheme on log(S): d log(S) = (mu - 0.5*V_t) dt + sqrt(V_t) dW_S
            # Simulating in log-space guarantees S > 0 for all paths.
            def _step(log_S: Any, inp: Any) -> tuple[Any, Any]:
                V_t, dti, dW_S_t = inp
                sigma = jnp.sqrt(jnp.maximum(V_t, 0.0))
                log_S_next = log_S + (self.mu - 0.5 * V_t) * dti + sigma * dW_S_t
                return log_S_next, log_S_next

            log_S0 = jnp.log(jnp.array(self.S0, dtype=jnp.float64))
            _, log_S_steps = jax.lax.scan(_step, log_S0, (V[:-1], dt, dW_S))
            log_S = jnp.concatenate([log_S0[None], log_S_steps])   # (T,)
            S = jnp.exp(log_S)

            return jnp.stack([S, V], axis=-1)   # (T, 2)

        keys = jax.random.split(key, n_paths)
        return jax.vmap(_one_path)(keys)   # (n_paths, T, 2)
