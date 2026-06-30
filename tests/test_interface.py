"""
Story 1.2 — State-Space Interface Core Types (MarketModel, Scenario, TimeGrid)

Covers all acceptance criteria from GitHub Issue #2.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def _make_metadata():
    """Return a minimal valid Metadata for use in Scenario tests."""
    from quantscenariobench.interface import Metadata, TimeGrid
    return Metadata(
        seed=0,
        prng_key_info="jax.random.key(0)",
        model_name="TestModel",
        model_version="0.1.0",
        parameters=None,
        time_grid=TimeGrid(jnp.array([0.0, 0.5, 1.0])),
        n_paths=1,
        library_version="0.1.0",
        dataset_version="1.0.0",
        generated_at="2026-06-30T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# AC: quantscenariobench.interface exports the four public types
# ---------------------------------------------------------------------------

def test_interface_exports_all_four_types():
    from quantscenariobench import interface
    for name in ("MarketModel", "Scenario", "Metadata", "TimeGrid"):
        assert hasattr(interface, name), f"interface must export {name}"


# ---------------------------------------------------------------------------
# AC: MarketModel cannot be instantiated directly → TypeError
# ---------------------------------------------------------------------------

def test_market_model_direct_instantiation_raises_type_error():
    from quantscenariobench.interface import MarketModel
    with pytest.raises(TypeError, match="abstract"):
        MarketModel()


# ---------------------------------------------------------------------------
# AC: MarketModel is an equinox.Module subclass with _drift and _diffusion
# ---------------------------------------------------------------------------

def test_market_model_is_equinox_module_subclass():
    import equinox as eqx
    from quantscenariobench.interface import MarketModel
    assert issubclass(MarketModel, eqx.Module)


def test_market_model_exposes_drift_and_diffusion():
    from quantscenariobench.interface import MarketModel
    assert callable(getattr(MarketModel, "_drift", None)), \
        "MarketModel must define _drift"
    assert callable(getattr(MarketModel, "_diffusion", None)), \
        "MarketModel must define _diffusion"


# ---------------------------------------------------------------------------
# AC: Concrete subclass missing _drift or _diffusion → TypeError on init
# ---------------------------------------------------------------------------

def test_subclass_missing_drift_raises_type_error():
    from quantscenariobench.interface import MarketModel

    class NoDrift(MarketModel):
        def _diffusion(self, t: Any, state: Any) -> Any:
            return state

    with pytest.raises(TypeError, match="_drift"):
        NoDrift()


def test_subclass_missing_diffusion_raises_type_error():
    from quantscenariobench.interface import MarketModel

    class NoDiffusion(MarketModel):
        def _drift(self, t: Any, state: Any) -> Any:
            return state

    with pytest.raises(TypeError, match="_diffusion"):
        NoDiffusion()


def test_fully_implemented_subclass_instantiates_without_error():
    import equinox as eqx
    from quantscenariobench.interface import MarketModel

    class FullModel(MarketModel):
        def _drift(self, t: Any, state: Any) -> Any:
            return state

        def _diffusion(self, t: Any, state: Any) -> Any:
            return state

    model = FullModel()
    assert isinstance(model, MarketModel)
    assert isinstance(model, eqx.Module)


# ---------------------------------------------------------------------------
# AC: Scenario.metadata is pytree aux_data, never a pytree leaf (AD-2)
# ---------------------------------------------------------------------------

def test_scenario_observation_and_latent_state_are_leaves():
    from quantscenariobench.interface import Scenario
    obs = jnp.array([1.0, 2.0, 3.0])
    lat = jnp.array([0.04, 0.04, 0.04])
    meta = _make_metadata()

    s = Scenario(observation=obs, latent_state=lat, metadata=meta)
    leaves = jax.tree_util.tree_leaves(s)

    assert any(jnp.array_equal(leaf, obs) for leaf in leaves), \
        "observation must appear as a pytree leaf"
    assert any(jnp.array_equal(leaf, lat) for leaf in leaves), \
        "latent_state must appear as a pytree leaf"


def test_scenario_metadata_is_not_a_pytree_leaf():
    from quantscenariobench.interface import Scenario
    obs = jnp.array([1.0, 2.0])
    lat = jnp.array([0.0, 0.0])
    meta = _make_metadata()

    s = Scenario(observation=obs, latent_state=lat, metadata=meta)
    leaves = jax.tree_util.tree_leaves(s)

    # None of the leaves should be the Metadata object itself
    assert not any(leaf is meta for leaf in leaves), \
        "metadata must not appear as a pytree leaf (must be static aux_data)"


# ---------------------------------------------------------------------------
# AC: Metadata carries exactly the 10 required fields (AD-8)
# ---------------------------------------------------------------------------

REQUIRED_METADATA_FIELDS = frozenset({
    "seed", "prng_key_info", "model_name", "model_version",
    "parameters", "time_grid", "n_paths", "library_version",
    "dataset_version", "generated_at",
})


def test_metadata_has_exactly_the_required_fields():
    from quantscenariobench.interface import Metadata
    actual = {f.name for f in dataclasses.fields(Metadata)}
    assert actual == REQUIRED_METADATA_FIELDS, (
        f"Metadata field mismatch.\n"
        f"  Missing : {sorted(REQUIRED_METADATA_FIELDS - actual)}\n"
        f"  Extra   : {sorted(actual - REQUIRED_METADATA_FIELDS)}"
    )


# ---------------------------------------------------------------------------
# AC: TimeGrid rejects non-monotonic arrays → ValueError
# ---------------------------------------------------------------------------

def test_time_grid_rejects_decreasing_sequence():
    from quantscenariobench.interface import TimeGrid
    with pytest.raises(ValueError):
        TimeGrid(jnp.array([1.0, 0.5, 0.0]))


def test_time_grid_rejects_partially_non_monotonic():
    from quantscenariobench.interface import TimeGrid
    with pytest.raises(ValueError):
        TimeGrid(jnp.array([0.0, 0.5, 0.3, 1.0]))


def test_time_grid_rejects_repeated_values():
    from quantscenariobench.interface import TimeGrid
    with pytest.raises(ValueError):
        TimeGrid(jnp.array([0.0, 0.5, 0.5, 1.0]))


# ---------------------------------------------------------------------------
# AC: TimeGrid accepts non-uniform (but strictly increasing) arrays (FR-3)
# ---------------------------------------------------------------------------

def test_time_grid_accepts_non_uniform_spacing():
    from quantscenariobench.interface import TimeGrid
    tg = TimeGrid(jnp.array([0.0, 0.01, 0.1, 0.5, 1.0]))
    assert len(tg) == 5


def test_time_grid_accepts_uniform_spacing():
    from quantscenariobench.interface import TimeGrid
    tg = TimeGrid(jnp.linspace(0.0, 1.0, 252))
    assert len(tg) == 252


def test_time_grid_preserves_array_values():
    from quantscenariobench.interface import TimeGrid
    pts = jnp.array([0.0, 0.25, 0.75, 1.0])
    tg = TimeGrid(pts)
    assert jnp.array_equal(tg.t, pts)


# ---------------------------------------------------------------------------
# AC: One-way dependency direction (AD-9)
# Each sub-package may import from quantscenariobench.interface and
# approved third-party deps, but never from sibling sub-packages.
# ---------------------------------------------------------------------------

# Sub-package → set of allowed intra-project imports
_ALLOWED = {
    "models":   {"interface"},
    "solver":   {"interface"},
    "api":      {"interface", "solver"},
    "export":   {"interface"},
    "testing":  {"interface"},
}


def _qsb_imports(source: str) -> set[str]:
    """Return the set of quantscenariobench sub-packages imported."""
    pkg = "quantscenariobench"
    return {
        m.group(1)
        for m in re.finditer(rf"(?:import|from)\s+{pkg}\.(\w+)", source)
    }


def test_dependency_direction_no_cross_subpackage_imports():
    pkg_root = _pkg_root()
    violations: list[str] = []

    for sub, allowed in _ALLOWED.items():
        for py_file in (pkg_root / sub).rglob("*.py"):
            illegal = _qsb_imports(py_file.read_text()) - allowed
            if illegal:
                violations.append(
                    f"{py_file.relative_to(pkg_root.parent)}: "
                    f"imports {illegal} (not allowed)"
                )

    assert not violations, (
        "AD-9 dependency direction violation:\n" + "\n".join(violations)
    )
