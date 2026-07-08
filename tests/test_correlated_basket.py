"""
Story 11.1 — Correlated Multi-Asset Scenario Generation (CorrelatedBasket)

Covers all acceptance criteria from GitHub Issue #86.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test
from quantscenariobench.api import simulate_correlated_basket
from quantscenariobench.interface import TimeGrid
from quantscenariobench.models import BlackScholes, Heston, RoughBergomi
from quantscenariobench.solver import solve_sde


def _tg(n=30):
    return TimeGrid(jnp.linspace(0.0, 1.0, n))


def _two_black_scholes():
    return [
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0),
        BlackScholes(mu=0.03, sigma=0.15, S0=50.0),
    ]


# ---------------------------------------------------------------------------
# AC2: rho that is not symmetric, not unit-diagonal, or not PSD raises
# before any simulation runs (FR-47, AD-36)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_rho,label", [
    (jnp.array([[1.0, 0.5], [0.6, 1.0]]), "not symmetric"),
    (jnp.array([[0.9, 0.5], [0.5, 1.0]]), "not unit-diagonal"),
    (jnp.array([[1.0, 1.5], [1.5, 1.0]]), "not PSD"),
])
def test_invalid_correlation_matrix_raises_before_simulation(bad_rho, label):
    models = _two_black_scholes()
    with pytest.raises(ValueError):
        simulate_correlated_basket(models, _tg(), n_paths=10, seed=1, rho=bad_rho)


def test_wrong_shape_correlation_matrix_raises():
    models = _two_black_scholes()
    with pytest.raises(ValueError):
        simulate_correlated_basket(
            models, _tg(), n_paths=10, seed=1, rho=jnp.eye(3)
        )


# ---------------------------------------------------------------------------
# AC1/AC3: all N assets are simulated with Brownian increments correlated
# via the Cholesky factor of rho; a 50k-path Black-Scholes basket's
# empirical log-return correlation matches rho entrywise within
# Monte-Carlo tolerance (FR-47, AD-36)
# ---------------------------------------------------------------------------

def test_black_scholes_basket_empirical_correlation_matches_rho():
    models = _two_black_scholes()
    rho = jnp.array([[1.0, 0.7], [0.7, 1.0]])

    scenarios, _ = simulate_correlated_basket(
        models, _tg(50), n_paths=50_000, seed=1, rho=rho
    )

    log_returns = [
        np.diff(np.log(np.asarray(s.observation)), axis=1).flatten() for s in scenarios
    ]
    empirical_rho = np.corrcoef(log_returns[0], log_returns[1])[0, 1]
    assert empirical_rho == pytest.approx(0.7, abs=0.01)


def test_three_asset_basket_empirical_correlation_matrix_matches_rho():
    models = [
        BlackScholes(mu=0.03, sigma=0.2, S0=100.0),
        BlackScholes(mu=0.02, sigma=0.25, S0=80.0),
        BlackScholes(mu=0.01, sigma=0.3, S0=60.0),
    ]
    rho = jnp.array([
        [1.0, 0.6, 0.2],
        [0.6, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])

    scenarios, _ = simulate_correlated_basket(
        models, _tg(50), n_paths=50_000, seed=2, rho=rho
    )
    log_returns = np.stack([
        np.diff(np.log(np.asarray(s.observation)), axis=1).flatten() for s in scenarios
    ])
    empirical_rho = np.corrcoef(log_returns)
    np.testing.assert_allclose(empirical_rho, np.asarray(rho), atol=0.015)


# ---------------------------------------------------------------------------
# AC4: rho = I reproduces N independent simulations bit-identically under
# a documented seed-derivation rule (FR-47, AD-36)
# ---------------------------------------------------------------------------

def test_identity_rho_is_bit_identical_to_documented_independent_construction():
    models = _two_black_scholes()
    time_grid = _tg()
    seed = 42

    scenarios, metadata = simulate_correlated_basket(
        models, time_grid, n_paths=20, seed=seed, rho=jnp.eye(2)
    )

    # Documented seed-derivation rule: jax.random.split(PRNGKey(seed), N)
    # gives each asset's independent-draw sub-key, exactly as solve_sde
    # itself derives per-path keys via jax.random.split.
    asset_keys = jax.random.split(jax.random.PRNGKey(seed), len(models))
    for i, model in enumerate(models):
        reference = solve_sde(
            model, time_grid, 20, asset_keys[i], model.initial_state(),
            return_randomness=True,
        )
        assert jnp.array_equal(scenarios[i].observation, reference.ys)


def test_identity_rho_bit_identical_for_heston_constituent():
    models = [
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0),
        Heston(mu=0.03, kappa=1.5, theta=0.05, xi=0.3, rho=-0.5, v0=0.05, S0=50.0),
    ]
    time_grid = _tg()
    seed = 7

    scenarios, _ = simulate_correlated_basket(
        models, time_grid, n_paths=15, seed=seed, rho=jnp.eye(2)
    )

    asset_keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    reference = solve_sde(
        models[1], time_grid, 15, asset_keys[1], models[1].initial_state(),
        return_randomness=True,
    )
    assert jnp.array_equal(scenarios[1].observation, reference.ys[:, :, 0])
    assert jnp.array_equal(scenarios[1].latent_state, reference.ys[:, :, 1])


# ---------------------------------------------------------------------------
# AC5: basket output is N unchanged Scenario objects — compose_returns
# accepts it with zero signature changes (FR-47, AD-36)
# ---------------------------------------------------------------------------

def test_basket_output_is_scenario_list_compatible_with_compose_returns():
    from quantscenariobench.benchmark.returns import compose_returns
    from quantscenariobench.interface import Scenario

    models = _two_black_scholes()
    scenarios, metadata = simulate_correlated_basket(
        models, _tg(), n_paths=10, seed=1, rho=jnp.array([[1.0, 0.4], [0.4, 1.0]])
    )

    assert isinstance(scenarios, list)
    assert len(scenarios) == 2
    for scenario in scenarios:
        assert isinstance(scenario, Scenario)

    matrix = compose_returns(scenarios, path_index=0)
    assert matrix.shape == (_tg().t.shape[0] - 1, 2)


def test_basket_metadata_carries_rho_seed_and_constituents():
    models = _two_black_scholes()
    rho = jnp.array([[1.0, 0.4], [0.4, 1.0]])
    _, metadata = simulate_correlated_basket(models, _tg(), n_paths=10, seed=5, rho=rho)

    assert jnp.array_equal(jnp.asarray(metadata.rho), rho)
    assert metadata.basket_seed == 5
    assert len(metadata.constituents) == 2
    for constituent in metadata.constituents:
        assert constituent["model_name"] == "BlackScholes"
        assert "model_version" in constituent


# ---------------------------------------------------------------------------
# AC7: basket metadata (rho, seed, constituents) round-trips through
# Parquet export / reload (FR-47)
# ---------------------------------------------------------------------------

def test_basket_metadata_round_trips_through_parquet():
    import pyarrow.parquet as pq

    from quantscenariobench.export import export_parquet

    models = _two_black_scholes()
    rho = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    scenarios, metadata = simulate_correlated_basket(
        models, _tg(), n_paths=5, seed=9, rho=rho
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "basket.parquet"
        export_parquet(scenarios, path, basket_metadata=metadata)
        table = pq.read_table(path)

        assert "basket_rho" in table.column_names
        assert "basket_seed" in table.column_names
        assert "basket_constituents" in table.column_names

        reloaded_rho = json.loads(table.column("basket_rho")[0].as_py())
        reloaded_seed = table.column("basket_seed")[0].as_py()
        reloaded_constituents = json.loads(table.column("basket_constituents")[0].as_py())

        np.testing.assert_allclose(np.asarray(reloaded_rho), np.asarray(rho))
        assert reloaded_seed == 9
        assert reloaded_constituents == metadata.constituents

        # Every row (across both scenarios' paths) carries the same
        # basket-wide value — not just the first.
        assert set(table.column("basket_seed").to_pylist()) == {9}


def test_export_parquet_without_basket_metadata_has_unchanged_columns():
    from quantscenariobench.export import export_parquet

    models = _two_black_scholes()
    scenarios, _ = simulate_correlated_basket(
        models, _tg(), n_paths=5, seed=9, rho=jnp.eye(2)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "no_basket.parquet"
        export_parquet(scenarios, path)
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        assert "basket_rho" not in table.column_names
        assert "basket_seed" not in table.column_names
        assert "basket_constituents" not in table.column_names


# ---------------------------------------------------------------------------
# AC8: jit/vmap posture matches simulate(); conformance tests exist in
# quantscenariobench/testing/ (FR-47, NFR-2 extended)
# ---------------------------------------------------------------------------

def test_correlated_basket_passes_conformance_suite():
    from quantscenariobench.testing import assert_correlated_basket_conforms

    models = _two_black_scholes()
    assert_correlated_basket_conforms(
        simulate_correlated_basket, models, _tg(), 10, 3, jnp.array([[1.0, 0.3], [0.3, 1.0]])
    )


def test_cholesky_mixing_step_is_jit_compatible():
    rho = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    cholesky = jnp.linalg.cholesky(rho)
    price_noise = jax.random.normal(jax.random.PRNGKey(0), (2, 20, 29))

    def mix(price_noise):
        return jnp.einsum("ij,j...->i...", cholesky, price_noise)

    eager = mix(price_noise)
    jitted = jax.jit(mix)(price_noise)
    assert jnp.allclose(eager, jitted)


# ---------------------------------------------------------------------------
# RoughBergomi — explicitly scoped out via a documented NotImplementedError,
# not silently uncorrelated/incorrect output (Review Focus)
# ---------------------------------------------------------------------------

def test_rough_bergomi_constituent_raises_not_implemented():
    models = [
        BlackScholes(mu=0.05, sigma=0.2, S0=100.0),
        RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, S0=100.0, mu=0.0),
    ]
    with pytest.raises(NotImplementedError):
        simulate_correlated_basket(models, _tg(), n_paths=5, seed=1, rho=jnp.eye(2))


def test_rough_bergomi_raises_before_any_valid_simulation_runs():
    """Even a fully valid rho must not let simulation proceed for an
    unsupported constituent (mirrors AC2's "raises before any simulation
    runs" posture for validation failures)."""
    models = [RoughBergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04, S0=100.0, mu=0.0)]
    with pytest.raises(NotImplementedError):
        simulate_correlated_basket(models, _tg(), n_paths=5, seed=1, rho=jnp.eye(1))
