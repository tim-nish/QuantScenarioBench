"""
Story 1.3 — Solver Layer (diffrax Integration)

Covers all acceptance criteria from GitHub Issue #3.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — x64 enabled before any test


# ---------------------------------------------------------------------------
# Minimal conforming MarketModel for solver tests
# ---------------------------------------------------------------------------

from quantscenariobench.interface import MarketModel, TimeGrid


class _GBM(MarketModel):
    """Minimal GBM model: dS = mu*S dt + sigma*S dW."""

    mu: float
    sigma: float

    def _drift(self, t: Any, state: Any) -> Any:
        return self.mu * state

    def _diffusion(self, t: Any, state: Any) -> Any:
        return self.sigma * state


_MODEL = _GBM(mu=0.05, sigma=0.2)
_Y0 = jnp.array(100.0)
_TS_UNIFORM = jnp.linspace(0.0, 1.0, 13)   # 12 steps, uniform
_TS_NON_UNIFORM = jnp.array([0.0, 0.01, 0.1, 0.5, 0.9, 1.0])  # irregular
_N_PATHS = 8
_KEY = jax.random.PRNGKey(0)


def _make_tg(ts: jax.Array) -> TimeGrid:
    return TimeGrid(ts)


# ---------------------------------------------------------------------------
# AC: diffrax appears only inside quantscenariobench.solver (AD-4, AD-9)
# ---------------------------------------------------------------------------

def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


def test_diffrax_import_only_in_solver():
    pkg_root = _pkg_root()
    violations: list[str] = []

    for py_file in pkg_root.rglob("*.py"):
        # Determine the sub-package relative to quantscenariobench/
        parts = py_file.relative_to(pkg_root).parts
        sub = parts[0] if len(parts) > 1 else "__init__"
        if sub == "solver":
            continue  # solver is allowed to import diffrax

        source = py_file.read_text()
        if re.search(r"\bimport diffrax\b", source) or re.search(r"\bfrom diffrax\b", source):
            violations.append(str(py_file.relative_to(pkg_root.parent)))

    assert not violations, (
        "diffrax imported outside quantscenariobench.solver (AD-4, AD-9):\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# AC: solve_sde returns arrays with shape (n_paths, T, *state_shape)
# ---------------------------------------------------------------------------

def test_solve_sde_output_shape_uniform_grid():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    # scalar state → shape (n_paths, T)
    assert result.ys.shape == (_N_PATHS, len(tg)), (
        f"Expected ({_N_PATHS}, {len(tg)}), got {result.ys.shape}"
    )


def test_solve_sde_output_shape_non_uniform_grid():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_NON_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    assert result.ys.shape == (_N_PATHS, len(tg))


def test_solve_sde_leading_axis_is_n_paths():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    for n in (1, 4, 16):
        result = solve_sde(_MODEL, tg, n, _KEY, _Y0)
        assert result.ys.shape[0] == n


# ---------------------------------------------------------------------------
# AC: reproducibility — same args → bit-identical output (NFR-1, FR-4)
# ---------------------------------------------------------------------------

def test_solve_sde_default_path_is_reproducible():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    r1 = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    r2 = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    assert jnp.array_equal(r1.ys, r2.ys), "Default path must be bit-identical on same inputs"


def test_solve_sde_randomness_path_is_reproducible():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    r1 = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0, return_randomness=True)
    r2 = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0, return_randomness=True)
    assert jnp.array_equal(r1.ys, r2.ys)
    assert jnp.array_equal(r1.brownian_increments, r2.brownian_increments)


def test_different_keys_produce_different_paths():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    key_a = jax.random.PRNGKey(1)
    key_b = jax.random.PRNGKey(2)
    r1 = solve_sde(_MODEL, tg, _N_PATHS, key_a, _Y0)
    r2 = solve_sde(_MODEL, tg, _N_PATHS, key_b, _Y0)
    assert not jnp.array_equal(r1.ys, r2.ys), "Different keys must yield different paths"


# ---------------------------------------------------------------------------
# AC: default path uses VirtualBrownianTree — no full noise array stored (AD-3)
# ---------------------------------------------------------------------------

def test_default_path_uses_virtual_brownian_tree():
    """Structural check: _default_path source must reference VirtualBrownianTree."""
    from quantscenariobench.solver import _sde

    source = inspect.getsource(_sde._default_path)
    assert "VirtualBrownianTree" in source, (
        "_default_path must use diffrax.VirtualBrownianTree (AD-3)"
    )


def test_default_path_has_no_return_randomness_conditional():
    """_default_path must be free of any return_randomness branch (AD-3)."""
    from quantscenariobench.solver import _sde

    source = inspect.getsource(_sde._default_path)
    assert "return_randomness" not in source, (
        "_default_path must not contain a return_randomness conditional (AD-3)"
    )


# ---------------------------------------------------------------------------
# AC: non-uniform TimeGrid — time axis of output corresponds to exact grid points
# ---------------------------------------------------------------------------

def test_non_uniform_grid_time_axis_exact():
    """Verify the number of time points in output matches the TimeGrid exactly."""
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_NON_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    # Output must have one value per time point in the TimeGrid
    assert result.ys.shape[1] == len(tg), (
        f"Expected {len(tg)} time values, got {result.ys.shape[1]}"
    )


def test_initial_state_is_preserved_at_t0():
    """First time point of every path must equal y0 (Euler initialised at y0)."""
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_NON_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    # All paths should start at y0
    assert jnp.allclose(result.ys[:, 0], _Y0), (
        "All paths must start at y0"
    )


# ---------------------------------------------------------------------------
# AC: return_randomness=True uses a separate construction path (AD-3, FR-5)
# ---------------------------------------------------------------------------

def test_randomness_path_returns_increments():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0, return_randomness=True)
    assert hasattr(result, "brownian_increments"), (
        "return_randomness=True must include brownian_increments"
    )
    # increments shape: (n_paths, T-1, *state_shape)
    T = len(tg)
    assert result.brownian_increments.shape == (_N_PATHS, T - 1), (
        f"Expected ({_N_PATHS}, {T - 1}), got {result.brownian_increments.shape}"
    )


def test_randomness_path_ys_shape_matches_default():
    from quantscenariobench.solver import solve_sde

    tg = _make_tg(_TS_UNIFORM)
    default_result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    rand_result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0, return_randomness=True)
    assert default_result.ys.shape == rand_result.ys.shape


def test_randomness_path_is_separate_function():
    """Structural: _randomness_path must exist and not contain VirtualBrownianTree."""
    from quantscenariobench.solver import _sde

    source = inspect.getsource(_sde._randomness_path)
    assert "VirtualBrownianTree" not in source, (
        "_randomness_path must not use VirtualBrownianTree — it is a separate path (AD-3)"
    )
    assert "return_randomness" not in source, (
        "_randomness_path must not contain a return_randomness conditional (AD-3)"
    )


def test_default_path_result_type():
    from quantscenariobench.solver import SDEResult, solve_sde

    tg = _make_tg(_TS_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0)
    assert isinstance(result, SDEResult)


def test_randomness_path_result_type():
    from quantscenariobench.solver import SDEResultWithRandomness, solve_sde

    tg = _make_tg(_TS_UNIFORM)
    result = solve_sde(_MODEL, tg, _N_PATHS, _KEY, _Y0, return_randomness=True)
    assert isinstance(result, SDEResultWithRandomness)
