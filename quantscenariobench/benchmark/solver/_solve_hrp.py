"""Hierarchical Risk Parity weights via López de Prado's (2016) three-stage
algorithm (FR-48, AD-37).

This module is part of the Optimizer Solver Layer (AD-14): together with
_solve_allocation.py it is one of the two files in
quantscenariobench.benchmark.solver — the only benchmark-layer package
that imports scipy (AD-14/AD-19). HierarchicalRiskParity (the strategy
class) never imports scipy itself; it calls hierarchical_risk_parity_weights
below, exactly as GlobalMinimumVariance/CVaROptimization call
solve_allocation.

Converts a JAX covariance matrix to NumPy on the way in and the resulting
weight vector back to JAX on the way out, following solve_allocation's
exact conversion pattern.

Implemented directly from López de Prado's published pseudocode (2016),
not adapted from any existing open-source HRP implementation (AD-10's
"never borrowed from a bundled quant library" posture, extended here to
the implementation itself, not just its test reference).

Unlike solve_allocation's SLSQP/linprog calls, this algorithm is
deterministic and non-iterative (tree clustering, a single dendrogram
ordering, then a fixed recursive bisection) — there is no convergence
failure mode, so QuantScenarioBenchSolverError is never raised here.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial.distance
from jaxtyping import Array, Float


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """rho_ij = cov_ij / sqrt(cov_ii * cov_jj), with a zero-variance asset's
    row/column defined as uncorrelated (0) rather than dividing by zero —
    the degenerate-input guard AC4 requires. Clipped to [-1, 1] to absorb
    any floating-point excursion before the sqrt in the distance formula.
    """
    diag = np.diag(covariance)
    zero_variance = diag <= 0.0
    safe_std = np.sqrt(np.where(zero_variance, 1.0, diag))
    correlation = covariance / np.outer(safe_std, safe_std)
    correlation[zero_variance, :] = 0.0
    correlation[:, zero_variance] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def _inverse_variance_portfolio(variances: np.ndarray) -> np.ndarray:
    """The inverse-variance portfolio (IVP) over a cluster's own
    variances, guarded against a zero-variance asset (AC4): in the
    limit var -> 0+, inverse-variance weight -> infinity, so a
    zero-variance ("risk-free") asset takes all the weight within its
    cluster, split equally among any other zero-variance assets present,
    rather than dividing by zero.
    """
    zero_variance = variances <= 0.0
    if np.any(zero_variance):
        weights = np.where(zero_variance, 1.0, 0.0)
        return weights / weights.sum()
    inverse_variance = 1.0 / variances
    return inverse_variance / inverse_variance.sum()


def _cluster_variance(covariance: np.ndarray, items: list) -> float:
    cluster_covariance = covariance[np.ix_(items, items)]
    weights = _inverse_variance_portfolio(np.diag(cluster_covariance)).reshape(-1, 1)
    return float((weights.T @ cluster_covariance @ weights)[0, 0])


def _recursive_bisection(covariance: np.ndarray, sort_order: list) -> np.ndarray:
    """Stage 3 (López de Prado 2016): top-down split of the
    quasi-diagonalized asset order into contiguous halves, each
    recursion level weighting the two halves inversely to their own
    inverse-variance-portfolio variance.
    """
    weight_by_position = np.ones(len(sort_order))
    clusters = [list(range(len(sort_order)))]

    while clusters:
        split_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                midpoint = len(cluster) // 2
                split_clusters.append(cluster[:midpoint])
                split_clusters.append(cluster[midpoint:])
        clusters = split_clusters

        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            left_assets = [sort_order[position] for position in left]
            right_assets = [sort_order[position] for position in right]
            left_variance = _cluster_variance(covariance, left_assets)
            right_variance = _cluster_variance(covariance, right_assets)
            left_share = 1.0 - left_variance / (left_variance + right_variance)
            weight_by_position[left] *= left_share
            weight_by_position[right] *= 1.0 - left_share

    weights = np.zeros(len(sort_order))
    for position, asset in enumerate(sort_order):
        weights[asset] = weight_by_position[position]
    return weights


def hierarchical_risk_parity_weights(
    covariance: Float[Array, "n n"], linkage_method: str = "single"
) -> Float[Array, " n"]:
    """Solve for Hierarchical Risk Parity weights (FR-48, AD-37).

    Three stages, in order:
    1. Tree clustering: correlation-distance d_ij = sqrt(0.5 * (1 - rho_ij))
       (rho from covariance); condensed distance matrix via
       scipy.spatial.distance.squareform; linkage via
       scipy.cluster.hierarchy.linkage(..., method=linkage_method).
    2. Quasi-diagonalization: reorder assets by the dendrogram leaf order
       (scipy.cluster.hierarchy.leaves_list).
    3. Recursive bisection: top-down split of the quasi-diagonalized
       order into contiguous halves, each level weighting the two halves
       inversely to their own inverse-variance-portfolio variance.

    Degenerate inputs (AC4): n=1 short-circuits to weight 1.0 before any
    clustering call. A zero-variance asset is handled by
    _correlation_from_covariance/_inverse_variance_portfolio's guards
    (never divides by zero, never produces NaN/inf). A constant-
    correlation matrix (every rho_ij equal) gives every off-diagonal
    entry of the condensed distance matrix the same value; scipy's
    linkage tie-breaking for equal distances is deterministic given a
    fixed input asset ordering (lower-indexed pairs merge first) — not
    specially handled here, since that determinism is exactly what AC7's
    reproducibility requirement needs, and scipy's behavior is already
    stable for identical inputs.
    """
    # jnp.cov(returns, rowvar=False) returns a 0-D scalar for a single
    # asset (n=1), not a (1, 1) matrix — atleast_2d normalizes that case
    # before the n=1 short-circuit below.
    covariance_np = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
    n = covariance_np.shape[0]

    if n == 1:
        return jnp.asarray([1.0])

    correlation = _correlation_from_covariance(covariance_np)
    distance = np.sqrt(0.5 * (1.0 - correlation))
    np.fill_diagonal(distance, 0.0)
    condensed_distance = scipy.spatial.distance.squareform(distance, checks=False)

    linkage_matrix = scipy.cluster.hierarchy.linkage(condensed_distance, method=linkage_method)
    sort_order = scipy.cluster.hierarchy.leaves_list(linkage_matrix).tolist()

    weights = _recursive_bisection(covariance_np, sort_order)
    return jnp.asarray(weights)
