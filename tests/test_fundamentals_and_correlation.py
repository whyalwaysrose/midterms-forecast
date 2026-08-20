"""Fundamentals prior and the election-day correlation kernel."""

from __future__ import annotations

import numpy as np
import pytest

from midterms import fundamentals as F
from midterms.config import ModelConfig, RaceSet
from midterms.model.correlation import (
    cholesky_factor,
    correlation_matrix,
    covariate_matrix,
    distance_matrix,
)


@pytest.fixture(scope="module")
def loaded():
    races = RaceSet.load()
    cfg = ModelConfig.load()
    return races, cfg, F.compute(races, cfg)


def test_logit_roundtrip():
    for p in (0.05, 0.3, 0.5, 0.72, 0.99):
        assert F.inv_logit(F.logit(p)) == pytest.approx(p, abs=1e-9)


def test_logit_is_clipped_not_infinite():
    assert np.isfinite(F.logit(0.0))
    assert np.isfinite(F.logit(1.0))


def test_margin_conversions_agree():
    assert F.logit_to_margin(0.0) == pytest.approx(0.0)
    assert F.share_to_margin(0.5) == pytest.approx(0.0)
    assert F.share_to_margin(0.55) == pytest.approx(10.0)


def test_one_logit_point_is_about_fifty_margin_points_near_a_tie():
    """The documented rule of thumb the config comments rely on."""
    assert F.logit_to_margin(0.02) == pytest.approx(1.0, abs=0.02)


def test_lean_is_relative_to_the_nation(loaded):
    races, cfg, _ = loaded
    # A state exactly at the national presidential result has zero lean.
    from midterms.config import Race

    neutral = Race(
        id="n", unit="NN", name="Neutral", special=False,
        incumbent_party="D", incumbent_status="open",
        pres_2024_dem_two_party=cfg.national.pres_2024_dem_two_party,
        pres_2020_dem_two_party=cfg.national.pres_2020_dem_two_party,
        region="south",
    )
    lean_2024, lean_2020 = F.race_lean(neutral, cfg)
    assert lean_2024 == pytest.approx(0.0, abs=1e-12)
    assert lean_2020 == pytest.approx(0.0, abs=1e-12)


def test_incumbency_bonus_is_signed_toward_the_incumbent_party(loaded):
    _races, cfg, _ = loaded
    from datetime import date

    from midterms.config import Control, Race, RaceSet

    def build(party, status):
        return Race(
            id=f"{party}-{status}", unit="NN", name="N", special=False,
            incumbent_party=party, incumbent_status=status,
            pres_2024_dem_two_party=cfg.national.pres_2024_dem_two_party,
            pres_2020_dem_two_party=cfg.national.pres_2020_dem_two_party,
            region="south",
        )

    control = Control(
        total_seats=4, seats_not_up={"D": 1, "R": 1}, seats_up={"D": 1, "R": 1},
        dem_seats_for_majority=3, rep_seats_for_majority=2, tiebreaker_party="R",
    )
    rs = RaceSet(
        cycle=2026, chamber="senate", election_date=date(2026, 11, 3),
        control=control,
        races=(build("D", "elected"), build("R", "elected")),
    )
    fund = F.compute(rs, cfg)
    dem_prior, rep_prior = fund.prior_mean
    bonus = cfg.fundamentals.incumbency_bonus["elected"]

    assert dem_prior == pytest.approx(bonus)
    assert rep_prior == pytest.approx(-bonus)


def test_open_seat_gets_no_incumbency_bonus(loaded):
    _races, cfg, _ = loaded
    assert cfg.fundamentals.incumbency_bonus["open"] == 0.0


def test_prior_ranks_states_sensibly(loaded):
    _races, _cfg, fund = loaded
    priors = dict(zip(fund.race_ids, fund.prior_mean, strict=True))
    # Wyoming should be far more Republican than Massachusetts, on fundamentals.
    assert priors["senate-2026-WY"] < priors["senate-2026-WV"] < priors["senate-2026-MA"]
    assert priors["senate-2026-MA"] > 0
    assert priors["senate-2026-WY"] < 0


# --------------------------------------------------------------- correlation


def test_correlation_matrix_is_a_valid_correlation_matrix(loaded):
    _races, cfg, fund = loaded
    corr = correlation_matrix(fund, cfg.election_day_error.correlation)

    n = len(fund.race_ids)
    assert corr.shape == (n, n)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert corr.min() >= 0.0
    assert corr.max() <= 1.0


def test_correlation_matrix_is_positive_definite(loaded):
    """It gets Cholesky-factored every run, so this must hold."""
    _races, cfg, fund = loaded
    corr = correlation_matrix(fund, cfg.election_day_error.correlation)
    eigenvalues = np.linalg.eigvalsh(corr)
    assert eigenvalues.min() > 0
    chol = cholesky_factor(corr)
    assert np.allclose(chol @ chol.T, corr, atol=1e-8)


def test_politically_similar_states_are_more_correlated(loaded):
    _races, cfg, fund = loaded
    corr = correlation_matrix(fund, cfg.election_day_error.correlation)
    idx = {rid: i for i, rid in enumerate(fund.race_ids)}

    # Two midwestern swing states should be more correlated with each other
    # than either is with Wyoming.
    mi_ia = corr[idx["senate-2026-MI"], idx["senate-2026-IA"]]
    mi_wy = corr[idx["senate-2026-MI"], idx["senate-2026-WY"]]
    assert mi_ia > mi_wy


def test_distance_matrix_properties(loaded):
    _races, cfg, fund = loaded
    cov = covariate_matrix(fund, cfg.election_day_error.correlation)
    d = distance_matrix(cov)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)
    assert (d >= 0).all()


def test_region_weight_of_zero_removes_the_region_signal(loaded):
    _races, cfg, fund = loaded
    import dataclasses

    no_region = dataclasses.replace(cfg.election_day_error.correlation, region_weight=0.0)
    cov = covariate_matrix(fund, no_region)
    # Region columns are all zero, so they contribute nothing to distance.
    assert np.allclose(cov[:, 2:], 0.0)
