"""Correlation structure for election-day polling error.

The single most important thing a seat model must get right is that polling
misses are *correlated*. If all 35 races erred independently, a chamber
forecast would be absurdly confident — the errors would average out and the
seat distribution would collapse to a spike. In reality, when the polls miss in
Ohio they tend to miss the same way in Iowa.

We model that with two pieces:

    national   a common shift applied to every race alike
    state      a shift whose correlation between two races decays with the
               political distance between their states

Distance is built from what we actually know about each state: its presidential
lean in 2024 and 2020 (standardised), plus a region indicator. States that vote
alike and sit in the same region are treated as likely to miss alike.
"""

from __future__ import annotations

import numpy as np

from ..config import CorrelationConfig
from ..fundamentals import Fundamentals


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-SD. Constant inputs map to zeros rather than NaN."""
    sd = float(np.std(values))
    if sd < 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / sd


def covariate_matrix(fundamentals: Fundamentals, cfg: CorrelationConfig) -> np.ndarray:
    """Standardised covariates used to measure political distance between races.

    Columns are the 2024 lean, the 2020 lean, and one scaled indicator column
    per region. Two races in different regions are pushed
    ``sqrt(2) * region_weight`` further apart than they would otherwise be.
    """
    lean_2024 = _standardise(fundamentals.lean_2024)
    lean_2020 = _standardise(fundamentals.lean_2020)

    unique_regions = sorted(set(fundamentals.regions))
    region_onehot = np.zeros((len(fundamentals.regions), len(unique_regions)))
    for i, region in enumerate(fundamentals.regions):
        region_onehot[i, unique_regions.index(region)] = 1.0
    region_onehot *= cfg.region_weight

    return np.column_stack([lean_2024, lean_2020, region_onehot])


def distance_matrix(covariates: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between rows."""
    diff = covariates[:, None, :] - covariates[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def correlation_matrix(
    fundamentals: Fundamentals, cfg: CorrelationConfig
) -> np.ndarray:
    """Correlation matrix for the state component of election-day error.

        corr(i, j) = (1 - nugget) * exp(-d(i, j) / length_scale) + nugget * [i == j]

    The exponential kernel is positive definite for Euclidean distance (it is
    the Matern-1/2 kernel), and adding a positive multiple of the identity keeps
    it comfortably so, which matters because we Cholesky-factor it.
    """
    covariates = covariate_matrix(fundamentals, cfg)
    distances = distance_matrix(covariates)

    corr = (1.0 - cfg.nugget) * np.exp(-distances / cfg.length_scale)
    corr[np.diag_indices_from(corr)] += cfg.nugget

    # Enforce exact symmetry; floating-point asymmetry upsets Cholesky.
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor, with a jitter fallback."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        for _ in range(8):
            try:
                return np.linalg.cholesky(corr + jitter * np.eye(len(corr)))
            except np.linalg.LinAlgError:
                jitter *= 10
        raise
