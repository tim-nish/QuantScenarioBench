"""
Story 5.1 — Equal Weight Baseline

Covers all acceptance criteria from GitHub Issue #27.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test


def _pkg_root() -> Path:
    return Path(__file__).parent.parent / "quantscenariobench"


# ---------------------------------------------------------------------------
# AC: EqualWeight is an equinox.Module subclass of BaselineStrategy (AD-13, AD-6)
# ---------------------------------------------------------------------------

def test_equal_weight_is_equinox_module_and_baseline_strategy():
    from quantscenariobench.benchmark.interface import BaselineStrategy
    from quantscenariobench.benchmark.strategies import EqualWeight

    strat = EqualWeight()
    assert isinstance(strat, eqx.Module)
    assert isinstance(strat, BaselineStrategy)


# ---------------------------------------------------------------------------
# AC: allocate() for an N-asset portfolio returns 1/N for every asset,
# regardless of historical_returns' content (FR-20)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_assets", [1, 2, 3, 7])
def test_equal_weight_allocates_uniformly(n_assets):
    from quantscenariobench.benchmark.strategies import EqualWeight

    strat = EqualWeight()
    historical_returns = jnp.linspace(-0.5, 0.5, 20 * n_assets).reshape(20, n_assets)

    weights = strat.allocate(historical_returns)
    assert weights.weights.shape == (n_assets,)
    assert jnp.allclose(weights.weights, 1.0 / n_assets)


def test_equal_weight_ignores_historical_returns_content():
    from quantscenariobench.benchmark.strategies import EqualWeight

    strat = EqualWeight()
    n = 4
    returns_a = jnp.zeros((10, n))
    returns_b = jnp.array([[100.0, -50.0, 0.3, 7.0]] * 10)

    weights_a = strat.allocate(returns_a)
    weights_b = strat.allocate(returns_b)
    assert jnp.array_equal(weights_a.weights, weights_b.weights)
    assert jnp.allclose(weights_a.weights, 1.0 / n)


# ---------------------------------------------------------------------------
# AC: EqualWeight is written entirely in jax.numpy and never calls
# quantscenariobench.benchmark.solver (AD-25)
# ---------------------------------------------------------------------------

_FORBIDDEN_SOLVER_OR_NUMPY = re.compile(r"(?:import|from)\s+(scipy|numpy)\b")


def test_equal_weight_never_imports_scipy_numpy_or_solver():
    src = (_pkg_root() / "benchmark" / "strategies" / "_equal_weight.py").read_text()
    assert not _FORBIDDEN_SOLVER_OR_NUMPY.search(src), \
        "EqualWeight must be jax.numpy-only, never scipy/numpy (AD-25)"
    assert not re.search(r"(?:import\s+.*\bsolver\b|solve_allocation\s*\()", src), \
        "EqualWeight must never call quantscenariobench.benchmark.solver (AD-25)"


# ---------------------------------------------------------------------------
# AC: quantscenariobench.benchmark.strategies.EqualWeight source imports
# only from quantscenariobench.benchmark.interface and equinox (AD-19)
# ---------------------------------------------------------------------------

_ALLOWED_MODULE_ROOTS = {"__future__", "jax", "jaxtyping", "equinox"}


def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # relative import, e.g. `from ..interface import X`
                roots.add("." * node.level + (node.module or ""))
            else:
                roots.add((node.module or "").split(".")[0])
    return roots


def test_equal_weight_imports_only_interface_and_equinox():
    path = _pkg_root() / "benchmark" / "strategies" / "_equal_weight.py"
    tree = ast.parse(path.read_text())
    roots = _imported_roots(tree)

    disallowed = {
        r for r in roots
        if r not in _ALLOWED_MODULE_ROOTS and r != "..interface"
    }
    assert not disallowed, (
        f"EqualWeight must import only quantscenariobench.benchmark.interface "
        f"and equinox (plus jax/jaxtyping); found disallowed imports: {disallowed}"
    )


def test_strategies_package_never_imports_runner_or_testing():
    strategies_dir = _pkg_root() / "benchmark" / "strategies"
    forbidden = re.compile(r"quantscenariobench\.benchmark\.(runner|testing)\b")
    violations = []
    for py_file in strategies_dir.rglob("*.py"):
        if forbidden.search(py_file.read_text()):
            violations.append(str(py_file.relative_to(_pkg_root().parent)))
    assert not violations, (
        f"AD-19 violation: benchmark.strategies must never import "
        f"benchmark.runner/testing: {violations}"
    )


# ---------------------------------------------------------------------------
# AC: the Story 4.2 conformance suite run against EqualWeight passes
# (FR-23, FR-25 cross-check)
# ---------------------------------------------------------------------------

def test_equal_weight_passes_conformance_suite():
    from quantscenariobench.benchmark.strategies import EqualWeight
    from quantscenariobench.benchmark.testing import assert_baseline_strategy_conforms

    strat = EqualWeight()
    historical_returns = jnp.linspace(-0.1, 0.1, 30).reshape(10, 3)

    assert_baseline_strategy_conforms(strat, historical_returns)
