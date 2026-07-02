"""The Optimizer Solver Layer (AD-14).

quantscenariobench.benchmark.solver is the only benchmark-layer module
that imports scipy. It is responsible for converting a JAX array to a
plain NumPy array on the way in and back to a JAX array on the way out
— no other benchmark module performs this conversion.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.optimize
from jaxtyping import Array, Float

from ._errors import QuantScenarioBenchSolverError


def solve_allocation(covariance: Float[Array, "n n"]) -> Float[Array, " n"]:
    """Solve a long-only, fully-invested minimum-variance allocation.

    Minimizes w^T Sigma w subject to sum(w) == 1 and w >= 0, via
    scipy.optimize.minimize (SLSQP). Converts covariance from JAX to
    NumPy on the way in, and the result back to JAX on the way out.

    Raises QuantScenarioBenchSolverError if the solver fails to converge,
    rather than returning a degenerate or unconverged weight vector.
    """
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
