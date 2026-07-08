"""Reusable conformance assertions for MarketModel implementors (FR-10, FR-11).

This module imports only from quantscenariobench.interface and the standard
library (AD-9).  The simulate() callable is injected by the caller so this
module never depends on the api or solver sub-packages.

Pass the simulate function from the api sub-package as simulate_fn; see
tests/test_conformance.py for reference usage.
"""
from __future__ import annotations

import warnings
from typing import Callable, Any

import jax.numpy as jnp

from ..interface import MarketModel, Scenario, TimeGrid, QuantScenarioBenchValidationWarning


def assert_scenario_schema(scenario: Any) -> None:
    """Assert the Scenario always carries observation, latent_state, and metadata (FR-2)."""
    assert hasattr(scenario, "observation"), \
        "Scenario missing 'observation' field (FR-2)"
    assert hasattr(scenario, "latent_state"), \
        "Scenario missing 'latent_state' field (FR-2)"
    assert hasattr(scenario, "metadata"), \
        "Scenario missing 'metadata' field (FR-2)"
    assert scenario.latent_state is not None, \
        "latent_state must be present (possibly empty array), never None (FR-2)"
    assert isinstance(scenario, Scenario), \
        f"simulate() must return a Scenario, got {type(scenario)!r}"


def assert_reproducible(scenario_a: Any, scenario_b: Any) -> None:
    """Assert two Scenarios produced with identical inputs are bit-identical (FR-4, NFR-1)."""
    assert jnp.array_equal(scenario_a.observation, scenario_b.observation), \
        "observation arrays are not bit-identical for identical inputs (FR-4, NFR-1)"
    assert jnp.array_equal(scenario_a.latent_state, scenario_b.latent_state), \
        "latent_state arrays are not bit-identical for identical inputs (FR-4, NFR-1)"


def assert_validation_behaviour(invalid_model_factory: Callable[[], MarketModel]) -> None:
    """Assert that constructing a model with a violated constraint emits the
    canonical warning and does NOT raise an exception (FR-6).

    Parameters
    ----------
    invalid_model_factory:
        A zero-argument callable that constructs a MarketModel instance
        with at least one parameter that violates a declared research constraint.
    """
    caught: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            invalid_model_factory()
        except Exception as exc:
            raise AssertionError(
                f"Expected only a QuantScenarioBenchValidationWarning but "
                f"construction raised {exc!r} (FR-6 requires warning, not exception)"
            ) from exc

    matching = [
        w for w in caught
        if issubclass(w.category, QuantScenarioBenchValidationWarning)
    ]
    assert matching, (
        "Expected QuantScenarioBenchValidationWarning to be emitted for a "
        f"constraint-violating model, but got: {[str(w.category) for w in caught]} (FR-6)"
    )


def assert_market_model_conforms(
    model: MarketModel,
    time_grid: TimeGrid,
    n_paths: int = 8,
    seed: int = 0,
    *,
    simulate_fn: Callable,
) -> None:
    """Run the full conformance suite against a MarketModel.

    Checks Scenario schema (FR-2) and reproducibility (FR-4, NFR-1).
    Validation behaviour is tested separately via assert_validation_behaviour.

    Parameters
    ----------
    model:
        A conforming MarketModel instance (with initial_state() implemented).
    time_grid:
        TimeGrid to pass to simulate_fn.
    n_paths:
        Number of paths for the conformance run.
    seed:
        PRNG seed for the conformance run.
    simulate_fn:
        The simulate() callable (or any compatible implementation).
        Injected by the caller so this module stays free of api imports
        (AD-9).
    """
    # --- Schema check (FR-2) ---
    scenario = simulate_fn(model, time_grid, n_paths, seed)
    assert_scenario_schema(scenario)

    # --- Reproducibility check (FR-4, NFR-1) ---
    scenario_again = simulate_fn(model, time_grid, n_paths, seed)
    assert_reproducible(scenario, scenario_again)


def assert_correlated_basket_conforms(
    simulate_correlated_basket_fn: Callable,
    models: Any,
    time_grid: TimeGrid,
    n_paths: int,
    seed: int,
    rho: Any,
) -> None:
    """Run the conformance suite against simulate_correlated_basket() (FR-47,
    AD-36, NFR-2 extended) — mirrors assert_market_model_conforms's schema
    and reproducibility checks, applied to every constituent Scenario, so
    a correlated basket's jit/vmap posture (inherited from the same
    solve_sde/replay_sde machinery simulate() itself uses) and
    reproducibility guarantee match simulate()'s own (AC8).

    Parameters
    ----------
    simulate_correlated_basket_fn:
        The simulate_correlated_basket() callable (or any compatible
        implementation). Injected by the caller so this module stays free
        of api imports (AD-9), mirroring simulate_fn above.
    """
    scenarios, _basket_metadata = simulate_correlated_basket_fn(
        models, time_grid, n_paths, seed, rho
    )
    for scenario in scenarios:
        assert_scenario_schema(scenario)

    scenarios_again, _ = simulate_correlated_basket_fn(models, time_grid, n_paths, seed, rho)
    for scenario, scenario_again in zip(scenarios, scenarios_again):
        assert_reproducible(scenario, scenario_again)
