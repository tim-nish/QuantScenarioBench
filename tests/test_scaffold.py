"""
Story 1.1 — Package Scaffold & Development Environment

Verifies all acceptance criteria from GitHub Issue #1:
- x64 precision is enabled on import (AD-7)
- All six sub-packages are importable
- pyproject.toml declares the pinned stack
- jax_enable_x64 appears exactly once, only in quantscenariobench/__init__.py
"""

import importlib
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# AC: import quantscenariobench → jax.config.x64_enabled is True (AD-7)
# ---------------------------------------------------------------------------

def test_x64_enabled_on_import():
    import jax
    import quantscenariobench  # noqa: F401 — import triggers __init__.py

    assert jax.config.x64_enabled, (
        "jax x64 must be enabled after 'import quantscenariobench' (AD-7)"
    )


# ---------------------------------------------------------------------------
# AC: all six sub-packages are importable without errors
# ---------------------------------------------------------------------------

SUBPACKAGES = [
    "quantscenariobench.interface",
    "quantscenariobench.models",
    "quantscenariobench.solver",
    "quantscenariobench.api",
    "quantscenariobench.export",
    "quantscenariobench.testing",
]


def test_all_subpackages_importable():
    for name in SUBPACKAGES:
        mod = importlib.import_module(name)
        assert mod is not None, f"{name} failed to import"


# ---------------------------------------------------------------------------
# AC: pyproject.toml declares the pinned stack (NFR-5, Architecture Spine)
# ---------------------------------------------------------------------------

def _load_pyproject_text() -> str:
    root = Path(__file__).parent.parent
    path = root / "pyproject.toml"
    assert path.exists(), "pyproject.toml not found at repo root"
    return path.read_text()


def test_pyproject_python_requires():
    text = _load_pyproject_text()
    assert 'requires-python' in text and '3.11' in text, (
        "pyproject.toml must specify requires-python = '>=3.11'"
    )


def test_pyproject_jax_dependency():
    text = _load_pyproject_text()
    assert re.search(r'jax\s*>=\s*0\.4\.38', text), (
        "pyproject.toml must declare jax>=0.4.38"
    )


def test_pyproject_diffrax_dependency():
    text = _load_pyproject_text()
    assert re.search(r'diffrax\s*==\s*0\.7\.2', text), (
        "pyproject.toml must pin diffrax==0.7.2"
    )


def test_pyproject_equinox_dependency():
    text = _load_pyproject_text()
    assert re.search(r'equinox\s*>=\s*0\.11\.10', text), (
        "pyproject.toml must declare equinox>=0.11.10"
    )


# ---------------------------------------------------------------------------
# AC: jax_enable_x64 appears exactly once, only in quantscenariobench/__init__.py
# ---------------------------------------------------------------------------

def test_jax_enable_x64_single_location():
    root = Path(__file__).parent.parent / "quantscenariobench"
    py_files = list(root.rglob("*.py"))
    assert py_files, "No .py files found under quantscenariobench/"

    hits = [
        f for f in py_files
        if "jax_enable_x64" in f.read_text()
    ]

    assert len(hits) == 1, (
        f"jax_enable_x64 must appear in exactly one file; found in: {hits}"
    )
    assert hits[0].name == "__init__.py" and hits[0].parent.name == "quantscenariobench", (
        f"jax_enable_x64 must be in quantscenariobench/__init__.py; found in: {hits[0]}"
    )
