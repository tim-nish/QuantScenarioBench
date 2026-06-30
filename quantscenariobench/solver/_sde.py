"""
Solver Layer — sole importer of diffrax in QuantScenarioBench (AD-4, AD-9).

Two independent construction paths (AD-3):
  _default_path     : uses diffrax.VirtualBrownianTree so that no noise array
                      is ever materialised in memory.
  _randomness_path  : pre-generates explicit Brownian increments and runs a
                      manual Euler-Maruyama scan, returning both the path and
                      the raw increments (FR-5).

These are fully separate functions — _default_path contains no return_randomness
conditional whatsoever (AD-3).

Public entry point: solve_sde(model, time_grid, n_paths, key, y0,
                               *, return_randomness=False)
"""
from __future__ import annotations

from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax

from ..interface import MarketModel, TimeGrid


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class SDEResult(NamedTuple):
    """Return type from the default (no randomness) construction path."""
    ys: jax.Array  # (n_paths, T, *state_shape)


class SDEResultWithRandomness(NamedTuple):
    """Return type from the explicit-randomness construction path (FR-5)."""
    ys: jax.Array                  # (n_paths, T, *state_shape)
    brownian_increments: jax.Array  # (n_paths, T-1, *state_shape)


# ---------------------------------------------------------------------------
# Construction path A — VirtualBrownianTree (default, AD-3)
# ---------------------------------------------------------------------------

def _default_path(
    model: MarketModel,
    ts: jax.Array,
    y0: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """One SDE path via diffrax.VirtualBrownianTree.

    No Brownian noise array is ever allocated; the tree evaluates increments
    on demand during the solve (AD-3).  Saves at every explicit TimeGrid point
    via diffrax.StepTo (AD-12).
    """
    t0, t1 = ts[0], ts[-1]

    bm = diffrax.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        tol=1e-5,
        shape=y0.shape,
        key=key,
    )
    def _diffusion_op(t, y, args):
        diff = model._diffusion(t, y)
        if diff.ndim >= 2:
            return lineax.MatrixLinearOperator(diff)
        return lineax.DiagonalLinearOperator(diff)

    drift_term = diffrax.ODETerm(lambda t, y, args: model._drift(t, y))
    noise_term = diffrax.ControlTerm(_diffusion_op, bm)
    solution = diffrax.diffeqsolve(
        diffrax.MultiTerm(drift_term, noise_term),
        solver=diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.StepTo(ts=ts),
        max_steps=4096,
    )
    return solution.ys  # (T, *state_shape)


# ---------------------------------------------------------------------------
# Shared Euler-Maruyama scan (used by both path B and replay)
# ---------------------------------------------------------------------------

def _euler_maruyama_scan(
    model: MarketModel,
    ts: jax.Array,
    y0: jax.Array,
    dW: jax.Array,  # (T-1, *state_shape) pre-computed increments
) -> jax.Array:
    """Euler-Maruyama scan over pre-computed Brownian increments.

    Returns the full path of shape (T, *state_shape), prepending y0.
    """
    dt = jnp.diff(ts)

    def _step(
        carry_y: jax.Array,
        inp: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        t, dw, dti = inp
        diff = model._diffusion(t, carry_y)
        # matrix-valued diffusion: use matmul; scalar/diagonal: use elementwise
        noise = diff @ dw if diff.ndim >= 2 else diff * dw
        y_next = carry_y + model._drift(t, carry_y) * dti + noise
        return y_next, y_next

    _, ys_steps = jax.lax.scan(_step, y0, (ts[:-1], dW, dt))
    return jnp.concatenate([y0[None], ys_steps], axis=0)  # (T, *state_shape)


# ---------------------------------------------------------------------------
# Construction path B — explicit Brownian increments (FR-5, AD-3)
# ---------------------------------------------------------------------------

def _randomness_path(
    model: MarketModel,
    ts: jax.Array,
    y0: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One path with explicitly materialised Brownian increments (FR-5).

    Separate construction from _default_path: no diffrax Brownian tree, no
    conditional flag.  Pre-generates dW_i ~ N(0, dt_i) for each time step,
    then delegates to _euler_maruyama_scan (AD-3).
    """
    dt = jnp.diff(ts)
    noise_keys = jax.random.split(key, dt.shape[0])

    def _sample_increment(k: jax.Array, dti: jax.Array) -> jax.Array:
        return jax.random.normal(k, shape=y0.shape) * jnp.sqrt(dti)

    dW = jax.vmap(_sample_increment)(noise_keys, dt)  # (T-1, *state_shape)
    ys = _euler_maruyama_scan(model, ts, y0, dW)
    return ys, dW


# ---------------------------------------------------------------------------
# Replay entry point — deterministic replay from stored increments (FR-5)
# ---------------------------------------------------------------------------

def replay_sde(
    model: MarketModel,
    time_grid: TimeGrid,
    y0: jax.Array,
    brownian_increments: jax.Array,  # (n_paths, T-1, *state_shape)
) -> SDEResult:
    """Reproduce paths deterministically from pre-computed Brownian increments.

    Runs _euler_maruyama_scan for each path, producing bit-identical results
    to the _randomness_path that generated the same ``brownian_increments``.
    """
    ts = time_grid.t
    ys = jax.vmap(
        lambda dW: _euler_maruyama_scan(model, ts, y0, dW)
    )(brownian_increments)
    return SDEResult(ys=ys)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve_sde(
    model: MarketModel,
    time_grid: TimeGrid,
    n_paths: int,
    key: jax.Array,
    y0: jax.Array,
    *,
    return_randomness: bool = False,
) -> SDEResult | SDEResultWithRandomness:
    """Simulate ``model`` for ``n_paths`` independent paths.

    Parameters
    ----------
    model:
        Concrete :class:`~quantscenariobench.interface.MarketModel` whose
        ``_drift(t, y)`` and ``_diffusion(t, y)`` define the SDE.
    time_grid:
        Explicit, ordered time points (AD-12).  The solver steps to exactly
        these points — non-uniform spacing is fully supported (FR-3).
    n_paths:
        Number of independent paths.  Leading axis of all returned arrays.
    key:
        Master JAX PRNG key.  Internally split into ``n_paths`` sub-keys.
    y0:
        Initial state shared across all paths.  Its shape determines the
        state dimension.
    return_randomness:
        ``False`` (default) — dispatches to :func:`_default_path` which uses
        ``diffrax.VirtualBrownianTree``; no noise array is ever stored (AD-3).

        ``True`` — dispatches to :func:`_randomness_path`, a completely
        separate implementation that pre-generates Brownian increments and
        returns them alongside the paths (AD-3, FR-5).

    Returns
    -------
    SDEResult | SDEResultWithRandomness
        ``.ys`` always has shape ``(n_paths, T, *state_shape)`` where
        ``T = len(time_grid)``.  :class:`SDEResultWithRandomness` also
        exposes ``.brownian_increments`` of shape
        ``(n_paths, T-1, *state_shape)``.
    """
    path_keys = jax.random.split(key, n_paths)
    ts = time_grid.t

    if return_randomness:
        ys, dW = jax.vmap(lambda k: _randomness_path(model, ts, y0, k))(path_keys)
        return SDEResultWithRandomness(ys=ys, brownian_increments=dW)

    ys = jax.vmap(lambda k: _default_path(model, ts, y0, k))(path_keys)
    return SDEResult(ys=ys)
