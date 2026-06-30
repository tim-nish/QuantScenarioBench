"""
Public orchestrator — the single entry point for scenario generation (FR-1).

simulate() is model-agnostic: all model behaviour is dispatched exclusively
through MarketModel._drift and MarketModel._diffusion (AD-4).  This module
imports only from quantscenariobench.interface and quantscenariobench.solver
(AD-9).
"""
from __future__ import annotations

import datetime
import importlib.metadata
from typing import Any

import jax
import jax.numpy as jnp

from ..interface import MarketModel, Metadata, Scenario, TimeGrid
from ..solver import replay_sde, solve_sde


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _library_version() -> str:
    try:
        return importlib.metadata.version("quantscenariobench")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _build_metadata(
    model: MarketModel,
    time_grid: TimeGrid,
    n_paths: int,
    seed: int,
) -> Metadata:
    return Metadata(
        seed=seed,
        prng_key_info=f"jax.random.PRNGKey({seed})",
        model_name=type(model).__name__,
        model_version=getattr(model, "version", "0.1.0"),
        parameters=model,
        time_grid=time_grid,
        n_paths=n_paths,
        library_version=_library_version(),
        dataset_version="1.0.0",
        generated_at=_utc_now(),
    )


def _assemble_scenario(
    model: MarketModel,
    time_grid: TimeGrid,
    n_paths: int,
    seed: int,
    ys: jax.Array,
) -> Scenario:
    """Build a Scenario from raw SDE output.

    Delegates state splitting to model.split_state(ys): models without a
    latent process return an empty latent_state by default; models like
    Heston override split_state to separate observation from latent (FR-2).
    """
    observation, latent_state = model.split_state(ys)
    metadata = _build_metadata(model, time_grid, n_paths, seed)
    return Scenario(
        observation=observation,
        latent_state=latent_state,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate(
    model: MarketModel,
    time_grid: TimeGrid,
    n_paths: int,
    seed: int,
    y0: jax.Array | None = None,
    *,
    return_randomness: bool = False,
    randomness: jax.Array | None = None,
) -> Scenario | tuple[Scenario, jax.Array]:
    """Generate a reproducible market scenario.

    Parameters
    ----------
    model:
        Any concrete :class:`~quantscenariobench.interface.MarketModel`.
        simulate() dispatches all model behaviour through ``_drift`` and
        ``_diffusion`` — there is no model-specific branching here (AD-4).
    time_grid:
        Explicit, ordered simulation time points (FR-3, AD-12).
    n_paths:
        Number of independent sample paths.
    seed:
        Integer seed.  Deterministically converted to a JAX PRNGKey so that
        identical (model, time_grid, n_paths, seed) inputs on the same backend
        always produce bit-identical results (FR-4, NFR-1).
    y0:
        Initial state, shared across all paths.  When ``None``, the model's
        ``initial_state()`` method is called to obtain the initial state.
    return_randomness:
        When ``True``, also returns the Brownian increments used to generate
        the paths, enabling deterministic replay (FR-5).
    randomness:
        Pre-computed Brownian increments (from a prior ``return_randomness=True``
        call).  When provided, the simulation is replayed from these increments
        rather than generating new ones; ``seed`` is used only for metadata.

    Returns
    -------
    Scenario
        When ``return_randomness=False`` and ``randomness`` is ``None``.
    tuple[Scenario, jax.Array]
        When ``return_randomness=True``; the second element is Brownian
        increments of shape ``(n_paths, T-1, *state_shape)``.
    """
    if y0 is None:
        y0 = model.initial_state()

    if randomness is not None:
        # Replay mode: run the deterministic scan over pre-computed increments.
        sde_result = replay_sde(model, time_grid, y0, randomness)
        return _assemble_scenario(model, time_grid, n_paths, seed, sde_result.ys)

    key = jax.random.PRNGKey(seed)

    if return_randomness:
        sde_result = solve_sde(model, time_grid, n_paths, key, y0, return_randomness=True)
        scenario = _assemble_scenario(model, time_grid, n_paths, seed, sde_result.ys)
        return scenario, sde_result.brownian_increments

    sde_result = solve_sde(model, time_grid, n_paths, key, y0)
    return _assemble_scenario(model, time_grid, n_paths, seed, sde_result.ys)
