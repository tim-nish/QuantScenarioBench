"""The Optimizer Solver Layer (AD-14).

quantscenariobench.benchmark.solver is the only benchmark-layer module
that imports scipy. It is responsible for converting a JAX array to a
plain NumPy array on the way in and back to a JAX array on the way out
— no other benchmark module performs this conversion.

solve_allocation(...) is the single public entry point both
GlobalMinimumVariance and CVaROptimization call; neither strategy imports
scipy directly, and solve_allocation's internal implementation is free to
change (e.g. to a future JAX-native QP/LP solver, AD-25) without either
strategy class changing.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jaxtyping import Array, Float

from ._errors import QuantScenarioBenchSolverError


def solve_allocation(
    covariance: Optional[Float[Array, "n n"]] = None,
    *,
    returns: Optional[Float[Array, "t n"]] = None,
    confidence_level: Optional[float] = None,
) -> Float[Array, " n"]:
    """Solve a long-only, fully-invested portfolio allocation.

    Two mutually exclusive problem formulations, selected by which
    arguments are supplied:

    - covariance: GlobalMinimumVariance's long-only-constrained quadratic
      program (minimize w^T Sigma w s.t. sum(w) == 1, w >= 0), via
      scipy.optimize.minimize (SLSQP).
    - returns + confidence_level: CVaROptimization's Rockafellar-Uryasev
      linear program (minimize Conditional Value-at-Risk at
      confidence_level s.t. sum(w) == 1, w >= 0), via
      scipy.optimize.linprog.

    Converts input from JAX to NumPy on the way in, and the result back
    to JAX on the way out. Raises QuantScenarioBenchSolverError if the
    solver fails to converge, rather than returning a degenerate or
    unconverged weight vector.
    """
    if returns is not None:
        if confidence_level is None:
            raise ValueError(
                "solve_allocation requires confidence_level when returns is given"
            )
        return _solve_cvar(returns, confidence_level)
    if covariance is not None:
        return _solve_min_variance(covariance)
    raise ValueError(
        "solve_allocation requires either covariance, or returns and "
        "confidence_level"
    )


def _solve_min_variance(covariance: Float[Array, "n n"]) -> Float[Array, " n"]:
    cov_np = np.asarray(covariance, dtype=np.float64)
    n = cov_np.shape[0]

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_np @ w)

    def objective_grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov_np @ w)

    constraints = (
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones(n),
        },
    )
    bounds = [(0.0, None)] * n
    w0 = np.full(n, 1.0 / n)

    result = scipy.optimize.minimize(
        objective,
        w0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise QuantScenarioBenchSolverError(
            f"solve_allocation failed to converge: {result.message}"
        )

    return jnp.asarray(result.x)


def _solve_cvar(
    returns: Float[Array, "t n"], confidence_level: float
) -> Float[Array, " n"]:
    """Rockafellar-Uryasev CVaR minimization LP.

    Variables: x = [w (n); zeta (1); z (t)], z_i >= max(loss_i(w) - zeta, 0).
    Minimize zeta + (1 / ((1 - confidence_level) * t)) * sum(z)
    subject to sum(w) == 1, w >= 0, z >= 0,
    z_i >= -returns_i . w - zeta for every period i.
    """
    returns_np = np.asarray(returns, dtype=np.float64)
    t, n = returns_np.shape

    c = np.concatenate(
        [np.zeros(n), [1.0], np.full(t, 1.0 / ((1.0 - confidence_level) * t))]
    )

    # -returns @ w - zeta - z <= 0  <=>  z_i >= -returns_i . w - zeta
    A_ub = np.hstack([-returns_np, -np.ones((t, 1)), -np.eye(t)])
    b_ub = np.zeros(t)

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(t)]).reshape(1, -1)
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n + [(None, None)] + [(0.0, None)] * t

    result = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise QuantScenarioBenchSolverError(
            f"solve_allocation failed to converge: {result.message}"
        )

    weights = result.x[:n]
    return jnp.asarray(weights)
