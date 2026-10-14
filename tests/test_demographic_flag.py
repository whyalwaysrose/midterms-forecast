"""The demographic covariates as a config flag, default off.

Measured, they do not improve the kernel (see test_demographic_covariates.py and
METHODOLOGY 5), so the default has to be genuinely off -- not "off-ish". The
first test is the one that matters: with the committed config the covariate
matrix must be exactly what it was before this code existed, because a silent
change to the kernel changes every race's correlated error and the width of the
chamber forecast with it.

The rest pin the enabled path, since a flag nobody exercises is a flag that
breaks quietly: fail-closed on a state with no data, residualisation actually
applied, and the matrix still positive-definite once a column is added, which it
must be because it gets Cholesky-factored on every run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from midterms import fundamentals as F
from midterms.config import load_all
from midterms.data import demographics as D
from midterms.model.correlation import (
    cholesky_factor,
    correlation_matrix,
    covariate_matrix,
)


@pytest.fixture(scope="module")
def senate():
    races, cfg = load_all()
    return F.compute(races, cfg), cfg


@pytest.fixture(scope="module")
def house():
    races, cfg = load_all(chamber="house")
    return F.compute(races, cfg), cfg


# --- the default ------------------------------------------------------------

def test_the_committed_config_has_them_off(senate, house):
    for _fund, cfg in (senate, house):
        corr = cfg.election_day_error.correlation
        assert corr.demographic_weight == 0.0
        assert corr.demographic_covariates == ()


def test_the_default_matrix_is_what_it_was_before_this_code(senate, house):
    """The test that keeps "default off" honest.

    Compares the committed config against one with the fields explicitly
    cleared. Any drift -- a weight that crept above zero, a column appended
    unconditionally -- shows up as an array that is not equal.
    """
    for fund, cfg in (senate, house):
        corr = cfg.election_day_error.correlation
        cleared = replace(corr, demographic_weight=0.0, demographic_covariates=())
        assert np.array_equal(
            covariate_matrix(fund, corr), covariate_matrix(fund, cleared)
        )
        # Three political blocks: 2024 lean, 2020 lean, one column per region.
        expected = 2 + len(set(fund.regions))
        assert covariate_matrix(fund, corr).shape[1] == expected


def test_a_zero_weight_adds_nothing_even_when_covariates_are_named(senate):
    """Weight is the switch, so naming covariates alone must not enable them."""
    fund, cfg = senate
    corr = cfg.election_day_error.correlation
    named = replace(
        corr, demographic_weight=0.0, demographic_covariates=("education", "urban")
    )
    assert np.array_equal(
        covariate_matrix(fund, named), covariate_matrix(fund, corr)
    )


# --- the enabled path -------------------------------------------------------

def test_enabling_adds_one_column_per_covariate(senate):
    fund, cfg = senate
    corr = cfg.election_day_error.correlation
    base = covariate_matrix(fund, corr).shape[1]
    one = replace(corr, demographic_weight=1.0, demographic_covariates=("urban",))
    two = replace(
        corr, demographic_weight=1.0, demographic_covariates=("urban", "education")
    )
    assert covariate_matrix(fund, one).shape[1] == base + 1
    assert covariate_matrix(fund, two).shape[1] == base + 2


def test_the_enabled_kernel_is_still_positive_definite(senate, house):
    """It is Cholesky-factored on every run, so this is not optional."""
    for fund, cfg in (senate, house):
        corr = replace(
            cfg.election_day_error.correlation,
            demographic_weight=1.5,
            demographic_covariates=("urban", "education"),
        )
        matrix = correlation_matrix(fund, corr)
        assert np.linalg.eigvalsh(matrix).min() > 0
        lower = cholesky_factor(matrix)
        assert np.allclose(lower @ lower.T, matrix, atol=1e-8)
        assert np.allclose(np.diag(matrix), 1.0)


def test_enabling_shrinks_the_kernel_toward_independence(house):
    """The consequence to know before turning it on.

    An extra column is an extra dimension of distance, so every pair moves
    apart. Independent error averages out over 435 seats and correlated error
    does not, so this narrows the seat interval -- which METHODOLOGY 11 warns
    against reading as an improvement. Asserted so that the direction is
    recorded rather than rediscovered.
    """
    fund, cfg = house
    corr = cfg.election_day_error.correlation
    off = correlation_matrix(fund, corr)
    on = correlation_matrix(
        fund,
        replace(corr, demographic_weight=1.5, demographic_covariates=("urban",)),
    )
    off_diagonal = ~np.eye(len(off), dtype=bool)
    assert on[off_diagonal].mean() < off[off_diagonal].mean()


def test_residualising_changes_the_columns_it_produces(senate):
    fund, cfg = senate
    corr = replace(
        cfg.election_day_error.correlation,
        demographic_weight=1.0,
        demographic_covariates=("education",),
    )
    residualised = covariate_matrix(fund, replace(corr, demographic_residualise=True))
    raw = covariate_matrix(fund, replace(corr, demographic_residualise=False))
    assert not np.allclose(residualised[:, -1], raw[:, -1])
    # The residualised column is orthogonal to the leans by construction; the
    # raw one is strongly collinear with them, which is the whole reason the
    # measurement preferred the residual specification.
    assert abs(np.corrcoef(residualised[:, -1], fund.lean_2024)[0, 1]) < 1e-8
    assert abs(np.corrcoef(raw[:, -1], fund.lean_2024)[0, 1]) > 0.5


# --- failing closed ---------------------------------------------------------

def test_an_unknown_covariate_name_is_refused(senate):
    fund, cfg = senate
    corr = replace(
        cfg.election_day_error.correlation,
        demographic_weight=1.0,
        demographic_covariates=("income",),
    )
    with pytest.raises(D.DemographicsError, match="unknown demographic covariate"):
        covariate_matrix(fund, corr)


def test_a_unit_with_no_demographics_is_refused_not_filled(monkeypatch):
    """Filling a gap would be a claim nobody made.

    A mean or a zero places that race at the centre of the demographic axis,
    which reads to the kernel as "average on this dimension" and changes how its
    election-day error correlates with every other race.
    """
    monkeypatch.setattr(D, "load", lambda path=None: {"GA": {"urban": 75.0}})
    with pytest.raises(D.DemographicsError, match="no demographics for"):
        D.values_for(("GA", "ZZ"), "urban")


def test_house_districts_resolve_to_their_state(monkeypatch):
    monkeypatch.setattr(D, "load", lambda path=None: {"AZ": {"urban": 89.8}})
    values = D.values_for(("AZ-01", "AZ-06"), "urban")
    # Every district in a state shares the state's value: the covariate adds
    # separation between states and none within one. Documented, not a bug.
    assert values.tolist() == [89.8, 89.8]


def test_state_of_unit_handles_both_chambers():
    assert D.state_of_unit("GA") == "GA"
    assert D.state_of_unit("AZ-01") == "AZ"
    assert D.state_of_unit("az-01") == "AZ"


def test_a_missing_csv_says_how_to_regenerate_it(tmp_path):
    with pytest.raises(D.DemographicsError, match="--refresh"):
        D.load(tmp_path / "absent.csv")
