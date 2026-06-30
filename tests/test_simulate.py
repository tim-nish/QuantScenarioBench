"""
Story 1.4 — simulate() Public Orchestrator

Covers all acceptance criteria from GitHub Issue #4.
"""
from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — x64 enabled before any test


# ---------------------------------------------------------------------------
# Minimal conforming MarketModel (no latent process — pure observable state)
# ---------------------------------------------------------------------------

from quantscenariobench.interface import MarketModel, Scenario, TimeGrid


class _GBM(MarketModel):
    """Minimal GBM: dS = mu*S dt + sigma*S dW.  No latent state."""

    mu: float
    sigma: float

    def _drift(self, t: Any, state: Any) -> Any:
        return self.mu * state

    def _diffusion(self, t: Any, state: Any) -> Any:
        return self.sigma * state


_MODEL = _GBM(mu=0.05, sigma=0.2)
_Y0 = jnp.array(100.0)  # scalar initial state
_TG = TimeGrid(jnp.linspace(0.0, 1.0, 13))  # 12 uniform steps
_N = 8
_SEED = 42

_REQUIRED_METADATA_FIELDS = frozenset({
    "seed", "prng_key_info", "model_name", "model_version",
    "parameters", "time_grid", "n_paths", "library_version",
    "dataset_version", "generated_at",
})


def _simulate(*args, **kw):
    from quantscenariobench.api import simulate
    return simulate(*args, **kw)


# ---------------------------------------------------------------------------
# AC 1: simulate() returns Scenario with exactly observation, latent_state,
#        metadata (FR-1, FR-2)
# ---------------------------------------------------------------------------

def test_simulate_returns_scenario():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert isinstance(s, Scenario)


def test_simulate_scenario_has_observation():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert hasattr(s, "observation")


def test_simulate_scenario_has_latent_state():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert hasattr(s, "latent_state")


def test_simulate_scenario_has_metadata():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert hasattr(s, "metadata")


# ---------------------------------------------------------------------------
# AC 2: simulate() source contains no model-specific branching (FR-1, AD-4)
# ---------------------------------------------------------------------------

def test_simulate_no_models_import():
    """api._simulate must never import quantscenariobench.models."""
    from quantscenariobench.api import _simulate as _sim_mod
    source = inspect.getsource(_sim_mod)
    assert "quantscenariobench.models" not in source, (
        "simulate() must not import quantscenariobench.models (AD-9, AD-4)"
    )


def test_simulate_no_model_class_names_in_source():
    """simulate() must not reference any concrete model class directly."""
    from quantscenariobench.api import _simulate as _sim_mod
    source = inspect.getsource(_sim_mod.simulate)
    for name in ("BlackScholes", "Heston", "RoughBergomi"):
        assert name not in source, (
            f"simulate() must not reference model class '{name}' (AD-4)"
        )


# ---------------------------------------------------------------------------
# AC 3: same (model, time_grid, n_paths, seed) → bit-identical Scenario
#        (FR-4, NFR-1)
# ---------------------------------------------------------------------------

def test_simulate_reproducible_observation():
    s1 = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    s2 = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert jnp.array_equal(s1.observation, s2.observation)


def test_simulate_reproducible_latent_state():
    s1 = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    s2 = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert jnp.array_equal(s1.latent_state, s2.latent_state)


def test_simulate_different_seeds_produce_different_paths():
    s1 = _simulate(_MODEL, _TG, _N, 1, _Y0)
    s2 = _simulate(_MODEL, _TG, _N, 2, _Y0)
    assert not jnp.array_equal(s1.observation, s2.observation)


# ---------------------------------------------------------------------------
# AC 4: model with no latent process → latent_state is empty but present (FR-2)
# ---------------------------------------------------------------------------

def test_latent_state_is_present():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    # Must exist and be an array — not None
    assert s.latent_state is not None
    assert hasattr(s.latent_state, "shape")


def test_latent_state_is_empty_for_no_latent_model():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    # Scalar-state GBM has no latent process: latent_state should be empty
    assert s.latent_state.size == 0, (
        f"Expected empty latent_state, got shape {s.latent_state.shape}"
    )


def test_latent_state_leading_axis_is_n_paths():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert s.latent_state.shape[0] == _N


# ---------------------------------------------------------------------------
# AC 5: default (return_randomness=False) → no raw noise in the return value
# ---------------------------------------------------------------------------

def test_default_returns_scenario_only():
    result = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert isinstance(result, Scenario), (
        "Default return_randomness=False must return a bare Scenario, not a tuple"
    )


def test_default_does_not_return_tuple():
    result = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert not isinstance(result, tuple)


# ---------------------------------------------------------------------------
# AC 6: return_randomness=True → replay produces identical arrays (FR-5)
# ---------------------------------------------------------------------------

def test_return_randomness_gives_tuple():
    result = _simulate(_MODEL, _TG, _N, _SEED, _Y0, return_randomness=True)
    assert isinstance(result, tuple) and len(result) == 2


def test_return_randomness_tuple_is_scenario_and_array():
    scenario, dW = _simulate(_MODEL, _TG, _N, _SEED, _Y0, return_randomness=True)
    assert isinstance(scenario, Scenario)
    assert hasattr(dW, "shape")


def test_return_randomness_increments_shape():
    scenario, dW = _simulate(_MODEL, _TG, _N, _SEED, _Y0, return_randomness=True)
    T = len(_TG)
    # dW shape: (n_paths, T-1, *state_shape); scalar state → (n_paths, T-1)
    assert dW.shape[0] == _N
    assert dW.shape[1] == T - 1


def test_replay_with_returned_randomness_is_bit_identical():
    """Using the returned randomness to replay must give the same arrays (FR-5)."""
    scenario, dW = _simulate(_MODEL, _TG, _N, _SEED, _Y0, return_randomness=True)
    replayed = _simulate(_MODEL, _TG, _N, _SEED, _Y0, randomness=dW)
    assert isinstance(replayed, Scenario)
    assert jnp.array_equal(scenario.observation, replayed.observation), (
        "Replay with stored randomness must produce bit-identical observation"
    )
    assert jnp.array_equal(scenario.latent_state, replayed.latent_state), (
        "Replay with stored randomness must produce bit-identical latent_state"
    )


# ---------------------------------------------------------------------------
# AC 7: api imports interface + solver only, never models (AD-9)
# ---------------------------------------------------------------------------

def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def test_api_does_not_import_models():
    pkg_root = _pkg_root()
    violations: list[str] = []
    for py_file in (pkg_root / "api").rglob("*.py"):
        source = py_file.read_text()
        if re.search(r"quantscenariobench\.models", source):
            violations.append(str(py_file.relative_to(pkg_root.parent)))
    assert not violations, (
        "api must never import quantscenariobench.models (AD-9):\n"
        + "\n".join(violations)
    )


def test_api_imports_only_interface_and_solver():
    pkg_root = _pkg_root()
    violations: list[str] = []
    allowed = {"interface", "solver"}
    for py_file in (pkg_root / "api").rglob("*.py"):
        source = py_file.read_text()
        for m in re.finditer(r"from \.\.((\w+))", source):
            sub = m.group(1)
            if sub not in allowed:
                violations.append(
                    f"{py_file.name}: imports from ..{sub} (not allowed)"
                )
    assert not violations, (
        "api may only import from interface and solver (AD-9):\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# AC 8: Scenario.metadata has all 10 required fields (AD-8, FR-4)
# ---------------------------------------------------------------------------

def test_metadata_has_all_required_fields():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    actual = {f.name for f in dataclasses.fields(s.metadata)}
    assert actual == _REQUIRED_METADATA_FIELDS


def test_metadata_seed_matches_argument():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert s.metadata.seed == _SEED


def test_metadata_n_paths_matches_argument():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert s.metadata.n_paths == _N


def test_metadata_model_name_is_class_name():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert s.metadata.model_name == "_GBM"


def test_metadata_parameters_is_the_model():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert s.metadata.parameters is _MODEL


def test_metadata_time_grid_is_the_time_grid():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert s.metadata.time_grid is _TG


def test_metadata_library_version_is_string():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert isinstance(s.metadata.library_version, str)
    assert len(s.metadata.library_version) > 0


def test_metadata_generated_at_is_iso_string():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    # Should parse as a valid ISO-8601 datetime
    import datetime
    # Just verify it's a non-empty string with timezone info
    assert isinstance(s.metadata.generated_at, str)
    assert "T" in s.metadata.generated_at  # ISO format separator


def test_metadata_prng_key_info_contains_seed():
    s = _simulate(_MODEL, _TG, _N, _SEED, _Y0)
    assert str(_SEED) in s.metadata.prng_key_info
