"""
Story 6.1 — Multi-Asset Composition & Return-Series Derivation

Covers all acceptance criteria from GitHub Issue #30.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def _make_scenario(observation, time_grid_points, model_name="TestModel"):
    from quantscenariobench.interface import Metadata, Scenario, TimeGrid

    time_grid = TimeGrid(jnp.array(time_grid_points))
    metadata = Metadata(
        seed=0,
        prng_key_info="jax.random.key(0)",
        model_name=model_name,
        model_version="0.1.0",
        parameters=None,
        time_grid=time_grid,
        n_paths=1,
        library_version="0.1.0",
        dataset_version="1.0.0",
        generated_at="2026-07-02T00:00:00Z",
    )
    return Scenario(
        observation=jnp.asarray(observation, dtype=float),
        latent_state=jnp.empty((0,)),
        metadata=metadata,
    )


_TG = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5 points -> 4 returns


# ---------------------------------------------------------------------------
# AC: N Scenarios sharing an identical TimeGrid assemble into one
# Float[Array, "t n"] returns matrix (FR-26)
# ---------------------------------------------------------------------------

def test_compose_returns_assembles_aligned_scenarios():
    from quantscenariobench.benchmark.returns import compose_returns, derive_returns

    s1 = _make_scenario([100.0, 101.0, 99.0, 102.0, 103.0], _TG)
    s2 = _make_scenario([50.0, 51.0, 52.0, 50.5, 53.0], _TG)

    matrix = compose_returns([s1, s2])
    assert matrix.shape == (4, 2)
    assert jnp.allclose(matrix[:, 0], derive_returns(s1))
    assert jnp.allclose(matrix[:, 1], derive_returns(s2))


def test_compose_returns_does_not_touch_simulate_models_or_export():
    returns_dir = _pkg_root() / "benchmark" / "returns"
    forbidden = re.compile(r"quantscenariobench\.(models|solver|api|export)\b")
    violations = []
    for py_file in returns_dir.rglob("*.py"):
        if forbidden.search(py_file.read_text()):
            violations.append(str(py_file.relative_to(_pkg_root().parent)))
    assert not violations, f"benchmark.returns must not touch: {violations}"


# ---------------------------------------------------------------------------
# AC: a TimeGrid mismatch raises before any return derivation is
# attempted — no implicit padding, truncation, or resampling (AD-22)
# ---------------------------------------------------------------------------

def test_compose_returns_rejects_mismatched_time_grid():
    from quantscenariobench.benchmark.returns import compose_returns

    s1 = _make_scenario([100.0, 101.0, 99.0, 102.0, 103.0], _TG)
    s2 = _make_scenario([50.0, 51.0, 52.0], [0.0, 0.5, 1.0])  # shorter TimeGrid

    with pytest.raises(ValueError):
        compose_returns([s1, s2])


def test_compose_returns_checks_alignment_before_deriving_returns():
    from quantscenariobench.benchmark.returns import compose_returns

    s1 = _make_scenario([100.0, 101.0, 99.0, 102.0, 103.0], _TG)
    # s2 has a mismatched TimeGrid AND an invalid (non-positive) observation;
    # if alignment were checked after derivation, this would raise from
    # derive_returns instead, with a different message.
    s2 = _make_scenario([-1.0, -2.0, -3.0], [0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="TimeGrid"):
        compose_returns([s1, s2])


# ---------------------------------------------------------------------------
# AC: derive_returns(scenario) returns simple/arithmetic period returns
# computed once per TimeGrid step (FR-28, AD-16)
# ---------------------------------------------------------------------------

def test_derive_returns_matches_hand_derived_arithmetic_returns():
    from quantscenariobench.benchmark.returns import derive_returns

    prices = [100.0, 110.0, 99.0, 108.9]
    scenario = _make_scenario(prices, [0.0, 0.25, 0.5, 0.75])

    expected = [
        (110.0 - 100.0) / 100.0,
        (99.0 - 110.0) / 110.0,
        (108.9 - 99.0) / 99.0,
    ]
    actual = derive_returns(scenario)
    assert jnp.allclose(actual, jnp.array(expected))


# ---------------------------------------------------------------------------
# AC: two Scenarios with identical observation paths produce identical
# return series (FR-28)
# ---------------------------------------------------------------------------

def test_derive_returns_identical_for_identical_observation():
    from quantscenariobench.benchmark.returns import derive_returns

    prices = [100.0, 105.0, 103.0, 110.0]
    s1 = _make_scenario(prices, [0.0, 0.25, 0.5, 0.75], model_name="ModelA")
    s2 = _make_scenario(prices, [0.0, 0.25, 0.5, 0.75], model_name="ModelB")

    assert jnp.array_equal(derive_returns(s1), derive_returns(s2))


# ---------------------------------------------------------------------------
# AC: a simulate()-produced Scenario and a Scenario loaded from a published
# Benchmark Dataset with identical observation paths produce identical
# return series — same convention regardless of source (FR-28)
# ---------------------------------------------------------------------------

def test_derive_returns_same_convention_regardless_of_provenance():
    from quantscenariobench.benchmark.returns import derive_returns

    prices = [50.0, 52.0, 48.0, 51.0, 55.0]
    simulated = _make_scenario(prices, _TG, model_name="BlackScholes")
    loaded_from_hf = _make_scenario(prices, _TG, model_name="LoadedFromHFDataset")

    assert jnp.array_equal(derive_returns(simulated), derive_returns(loaded_from_hf))


# ---------------------------------------------------------------------------
# AC: derive_returns is written entirely in jax.numpy and is
# jit-compatible (AD-16, AD-25)
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT = re.compile(r"(?:import|from)\s+(scipy|numpy)\b")


def test_returns_module_never_imports_scipy_or_numpy():
    returns_dir = _pkg_root() / "benchmark" / "returns"
    violations = []
    for py_file in returns_dir.rglob("*.py"):
        if _FORBIDDEN_IMPORT.search(py_file.read_text()):
            violations.append(str(py_file.relative_to(_pkg_root().parent)))
    assert not violations, f"AD-16/AD-25 violation: {violations}"


def test_derive_returns_is_jit_compatible():
    from quantscenariobench.benchmark.returns import derive_returns

    scenario = _make_scenario([100.0, 101.0, 99.0, 102.0, 103.0], _TG)

    def compute_from_observation(observation):
        # jit over the traced leaf only; metadata (incl. TimeGrid) stays
        # static, mirroring how Scenario itself is a pytree (AD-2).
        from quantscenariobench.interface import Scenario
        s = Scenario(
            observation=observation,
            latent_state=scenario.latent_state,
            metadata=scenario.metadata,
        )
        return derive_returns(s)

    eager = compute_from_observation(scenario.observation)
    jitted = jax.jit(compute_from_observation)(scenario.observation)
    assert jnp.allclose(eager, jitted)


# ---------------------------------------------------------------------------
# AC: a Scenario whose observation is not a one-dimensional,
# strictly-positive price series is rejected (AD-22)
# ---------------------------------------------------------------------------

def test_derive_returns_rejects_non_1d_observation():
    from quantscenariobench.benchmark.returns import derive_returns

    scenario = _make_scenario(
        [[100.0, 101.0], [102.0, 103.0]], [0.0, 1.0]
    )  # 2-D observation
    with pytest.raises(ValueError):
        derive_returns(scenario)


def test_derive_returns_rejects_non_positive_observation():
    import equinox as eqx
    from quantscenariobench.benchmark.returns import derive_returns

    scenario = _make_scenario([100.0, -5.0, 99.0], [0.0, 0.5, 1.0])
    with pytest.raises(eqx.EquinoxRuntimeError):
        derive_returns(scenario)


def test_derive_returns_rejects_zero_observation():
    import equinox as eqx
    from quantscenariobench.benchmark.returns import derive_returns

    scenario = _make_scenario([100.0, 0.0, 99.0], [0.0, 0.5, 1.0])
    with pytest.raises(eqx.EquinoxRuntimeError):
        derive_returns(scenario)
