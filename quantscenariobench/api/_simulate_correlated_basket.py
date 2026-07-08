"""
simulate_correlated_basket() — N-asset scenario generation with
Brownian increments correlated by a validated correlation matrix
(FR-47, AD-36).

This module imports only from quantscenariobench.interface and
quantscenariobench.solver (AD-9), mirroring simulate()'s own dependency
direction.

Key architectural trade-off (AD-36, Task 2.1): solve_sde's default
construction path (_default_path) uses diffrax.VirtualBrownianTree and
never materializes Brownian increments (AD-3) — increments for
different assets are therefore never simultaneously available to
correlate. Correlated basket generation instead reuses the
explicit-randomness path (solve_sde(..., return_randomness=True) /
replay_sde), already built for FR-5 replay, which does materialize a
(n_paths, T-1, *state_shape) increment array per asset. This trades
away the default path's no-materialization memory guarantee, but only
for basket generation — simulate()'s own default behavior is completely
unchanged.

Cross-asset correlation is applied to each asset's price-driving noise
sub-component only (Task 2.3, matching SPEC-QSB-008 FR-A): for a
scalar-state model (e.g. BlackScholes) the entire increment array is
the price-driving noise; for a multi-dimensional-state model (e.g.
Heston) it is index 0 of the state axis specifically — matching the
Scenario/split_state() convention that index 0 is always the
observation ("price") component. Any other per-asset state component
(e.g. Heston's own vol-of-vol driver, already correlated with its own
price driver via that model's own _diffusion Cholesky structure, a
separate and unchanged concern) is drawn independently across assets.

RoughBergomi (and any future model overriding MarketModel._generate_paths)
is not supported: such models are non-Markovian and bypass the
Euler-Maruyama/explicit-randomness machinery this basket construction
relies on entirely (fractional-Brownian-motion cross-correlation is a
different, harder problem than iid Gaussian increment mixing) — a
constituent of this kind raises NotImplementedError before any
simulation runs, rather than silently producing uncorrelated or
incorrect output.

Out of scope for v1 (documented here per SPEC-QSB-008): time-varying or
regime-switching correlation, and true multivariate models (multi-asset
Heston with cross vol spillovers, Wishart processes).
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from ..interface import BasketMetadata, MarketModel, Scenario, TimeGrid
from ..solver import replay_sde, solve_sde
from ._simulate import _assemble_scenario

_PSD_TOLERANCE = 1e-8


def _validate_correlation_matrix(rho: jax.Array, n_assets: int) -> None:
    """Raise before any simulation runs if rho is not a valid correlation
    matrix (AC2, AD-36): square N x N, symmetric, unit diagonal, PSD.
    """
    if rho.ndim != 2 or rho.shape != (n_assets, n_assets):
        raise ValueError(
            f"simulate_correlated_basket requires rho to be a "
            f"{n_assets}x{n_assets} matrix (one row/column per model); "
            f"got shape {rho.shape}"
        )
    if not bool(jnp.allclose(rho, rho.T)):
        raise ValueError(
            "simulate_correlated_basket requires rho to be symmetric (AD-36)"
        )
    if not bool(jnp.allclose(jnp.diag(rho), 1.0)):
        raise ValueError(
            "simulate_correlated_basket requires rho to have a unit diagonal (AD-36)"
        )
    eigenvalues = jnp.linalg.eigvalsh(rho)
    if bool(jnp.any(eigenvalues < -_PSD_TOLERANCE)):
        raise ValueError(
            "simulate_correlated_basket requires rho to be positive "
            f"semi-definite; got eigenvalues {eigenvalues} (AD-36)"
        )


def _require_markovian(model: MarketModel) -> None:
    """Raise NotImplementedError for any model overriding _generate_paths()
    (non-Markovian, e.g. RoughBergomi) — a generic check, not a hardcoded
    model-name branch, mirroring MarketModel.__check_init__'s own style of
    detecting an unoverridden base-class method.
    """
    if type(model)._generate_paths is not MarketModel._generate_paths:
        raise NotImplementedError(
            f"simulate_correlated_basket does not support {type(model).__name__} "
            "constituents: models overriding _generate_paths() are "
            "non-Markovian and bypass the Euler-Maruyama/explicit-randomness "
            "machinery this basket construction relies on. BlackScholes and "
            "Heston constituents are supported in v1 (SPEC-QSB-008)."
        )


def simulate_correlated_basket(
    models: Sequence[MarketModel],
    time_grid: TimeGrid,
    n_paths: int,
    seed: int,
    rho: jax.Array,
) -> tuple[list[Scenario], BasketMetadata]:
    """Simulate N assets with Brownian increments correlated by rho (FR-47).

    Seed-derivation rule (AC4, documented and tested): the basket key
    jax.random.PRNGKey(seed) is split via jax.random.split into one
    sub-key per asset (the same jax.random.split convention solve_sde
    itself uses to derive per-path keys) — asset i's raw, uncorrelated
    increments are exactly what
    solve_sde(models[i], time_grid, n_paths, asset_keys[i], y0_i,
    return_randomness=True) would produce on its own. When rho is the
    identity matrix, the Cholesky factor is the identity, so the
    correlation step leaves every asset's increments completely
    unchanged — the basket collapses to that same computation
    bit-identically (AC4).

    Returns (scenarios, basket_metadata): scenarios is exactly
    list[Scenario] with the unchanged Scenario schema (AC5) —
    compose_returns and every existing Scenario consumer accept it with
    zero signature changes; basket_metadata is the additive basket-level
    record (rho, basket_seed, constituent identifiers).
    """
    n_assets = len(models)
    if n_assets == 0:
        raise ValueError("simulate_correlated_basket requires at least one model")

    rho = jnp.asarray(rho, dtype=float)
    _validate_correlation_matrix(rho, n_assets)
    for model in models:
        _require_markovian(model)

    cholesky = jnp.linalg.cholesky(rho)
    asset_keys = jax.random.split(jax.random.PRNGKey(seed), n_assets)
    y0s = [model.initial_state() for model in models]

    raw = [
        solve_sde(model, time_grid, n_paths, asset_keys[i], y0s[i], return_randomness=True)
        for i, model in enumerate(models)
    ]
    raw_increments = [result.brownian_increments for result in raw]

    # Extract the price-driving noise sub-component: the whole array for a
    # scalar state (BlackScholes), index 0 of the state axis otherwise
    # (Heston) — then Cholesky-mix it across the asset axis.
    price_noise = jnp.stack(
        [
            increments if increments.ndim == 2 else increments[..., 0]
            for increments in raw_increments
        ],
        axis=0,
    )  # (N, n_paths, T-1)
    correlated_price_noise = jnp.einsum("ij,j...->i...", cholesky, price_noise)

    correlated_increments = [
        correlated_price_noise[i] if increments.ndim == 2
        else increments.at[..., 0].set(correlated_price_noise[i])
        for i, increments in enumerate(raw_increments)
    ]

    scenarios = [
        _assemble_scenario(
            model, time_grid, n_paths, seed,
            replay_sde(model, time_grid, y0s[i], correlated_increments[i]).ys,
        )
        for i, model in enumerate(models)
    ]

    constituents = [
        {
            "model_name": type(model).__name__,
            "model_version": getattr(model, "version", "0.1.0"),
        }
        for model in models
    ]
    basket_metadata = BasketMetadata(rho=rho, basket_seed=seed, constituents=constituents)
    return scenarios, basket_metadata
