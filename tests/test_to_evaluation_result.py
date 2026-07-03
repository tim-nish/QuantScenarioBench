"""
Story 7.2 — BenchmarkResult -> EvaluationResult Transformation

Covers all acceptance criteria from GitHub Issue #47: a single, pure
function that converts a BenchmarkResult into an EvaluationResult
(FR-31, AD-26).
"""

from __future__ import annotations

import ast
import copy
import pathlib

import jax
import pytest

import quantscenariobench  # noqa: F401 — ensures x64 is enabled before any test

_EVALUATION_PACKAGE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "quantscenariobench"
    / "benchmark"
    / "evaluation"
)

# Sibling benchmark submodules the evaluation package must never import
# directly (mirrors AD-19's dependency posture).
_FORBIDDEN_BENCHMARK_SIBLINGS = frozenset({
    "strategies",
    "solver",
    "metrics",
    "returns",
    "runner",
})


def _make_benchmark_result(**overrides):
    from quantscenariobench.benchmark.interface import BenchmarkResult

    defaults = dict(
        strategy_name="EqualWeight",
        strategy_parameters={},
        metrics={"sharpe_ratio": 1.23, "max_drawdown": -0.12},
        asset_scenario_ids=["scenario-asset-0", "scenario-asset-1"],
        time_grid_reference="tg-daily-2026-07-02",
        library_version="1.0.0",
        generated_at="2026-07-03T00:00:00+00:00",
    )
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def _returns(key, t, n):
    return jax.random.normal(key, (t, n)) * 0.01


_HIST = _returns(jax.random.PRNGKey(0), 20, 3)
_EVAL = _returns(jax.random.PRNGKey(1), 10, 3)


def _run(strategy, **kwargs):
    from quantscenariobench.benchmark.runner import run_benchmark
    return run_benchmark(strategy, _HIST, _EVAL, **kwargs)


# ---------------------------------------------------------------------------
# AC: to_evaluation_result(result) returns an EvaluationResult populating
# every required field from the corresponding BenchmarkResult field
# (FR-31, AD-26)
# ---------------------------------------------------------------------------

def test_to_evaluation_result_populates_every_field_from_benchmark_result():
    from quantscenariobench.benchmark.evaluation import to_evaluation_result

    benchmark_result = _make_benchmark_result()
    evaluation_result = to_evaluation_result(benchmark_result)

    assert evaluation_result.strategy.name == benchmark_result.strategy_name
    assert evaluation_result.strategy.parameters == benchmark_result.strategy_parameters
    assert (
        evaluation_result.benchmark_dataset.asset_scenario_ids
        == benchmark_result.asset_scenario_ids
    )
    assert (
        evaluation_result.benchmark_dataset.time_grid_reference
        == benchmark_result.time_grid_reference
    )
    assert [(m.name, m.value) for m in evaluation_result.metrics] == list(
        benchmark_result.metrics.items()
    )
    assert evaluation_result.library_version == benchmark_result.library_version
    assert evaluation_result.generated_at == benchmark_result.generated_at
    assert evaluation_result.schema_version
    assert evaluation_result.result_id


def test_to_evaluation_result_on_a_completed_run_benchmark_call():
    from quantscenariobench.benchmark.evaluation import to_evaluation_result
    from quantscenariobench.benchmark.strategies import GlobalMinimumVariance

    benchmark_result = _run(GlobalMinimumVariance(long_only=True))
    evaluation_result = to_evaluation_result(benchmark_result)

    assert evaluation_result.strategy.name == "GlobalMinimumVariance"
    assert evaluation_result.strategy.parameters == benchmark_result.strategy_parameters
    assert [(m.name, m.value) for m in evaluation_result.metrics] == list(
        benchmark_result.metrics.items()
    )


# ---------------------------------------------------------------------------
# AC: to_evaluation_result(result) called twice on the same BenchmarkResult
# produces identical EvaluationResults (determinism, FR-31)
# ---------------------------------------------------------------------------

def test_to_evaluation_result_is_deterministic_for_the_same_input():
    from quantscenariobench.benchmark.evaluation import to_evaluation_result

    benchmark_result = _make_benchmark_result()

    first = to_evaluation_result(benchmark_result)
    second = to_evaluation_result(benchmark_result)

    assert first == second
    assert first.result_id == second.result_id


def test_to_evaluation_result_id_differs_for_different_benchmark_results():
    from quantscenariobench.benchmark.evaluation import to_evaluation_result

    first = to_evaluation_result(_make_benchmark_result())
    second = to_evaluation_result(_make_benchmark_result(strategy_name="GlobalMinimumVariance"))

    assert first.result_id != second.result_id


# ---------------------------------------------------------------------------
# AC: to_evaluation_result's source code does not mutate or subclass
# BenchmarkResult, and requires zero changes to run_benchmark() or any
# Epic 6 module (FR-31, AD-26)
# ---------------------------------------------------------------------------

def test_to_evaluation_result_does_not_mutate_the_input_benchmark_result():
    from quantscenariobench.benchmark.evaluation import to_evaluation_result
    from quantscenariobench.benchmark.interface import BenchmarkResult

    benchmark_result = _make_benchmark_result()
    before = copy.deepcopy(benchmark_result)

    to_evaluation_result(benchmark_result)

    assert benchmark_result == before
    assert type(benchmark_result) is BenchmarkResult


# ---------------------------------------------------------------------------
# AC: quantscenariobench.benchmark.evaluation's import statements import
# only quantscenariobench.benchmark.interface — never strategies, solver,
# metrics, returns, or runner directly (mirrors AD-19's dependency posture)
# ---------------------------------------------------------------------------

def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            # relative imports (e.g. `from ..strategies import X`) carry no
            # module string component-wise beyond what `module` already
            # gives when level > 0 and module is set (e.g. "strategies").
    return names


@pytest.mark.parametrize(
    "path", sorted(_EVALUATION_PACKAGE_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_evaluation_package_never_imports_forbidden_benchmark_siblings(path):
    source = path.read_text()
    imported = _imported_module_names(source)

    for forbidden in _FORBIDDEN_BENCHMARK_SIBLINGS:
        assert forbidden not in imported, (
            f"{path.name} imports forbidden sibling module {forbidden!r}: {imported}"
        )
