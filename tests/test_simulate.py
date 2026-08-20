"""Seat arithmetic, correlated error, and the tipping-point calculation."""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pytest
import xarray as xr

from midterms import fundamentals as F
from midterms.config import Control, ModelConfig, Race, RaceSet
from midterms.model.simulate import _tipping_point, simulate_chamber

# ------------------------------------------------------------- tipping point


def test_tipping_point_picks_the_majority_making_race():
    # 3 contested races, 1 seat already held, majority needs 3 seats
    # => 2 of the 3 contested races must be won.
    # Ordered most-Dem first, the SECOND race in that order is decisive.
    simulated = np.array([[3.0, 1.0, -2.0]])   # order: 0, 1, 2
    tip = _tipping_point(simulated, seats_not_up_dem=1, dem_majority=3)
    assert tip[0] == 1


def test_tipping_point_handles_already_having_a_majority():
    simulated = np.array([[1.0, 2.0, 3.0]])
    tip = _tipping_point(simulated, seats_not_up_dem=5, dem_majority=3)
    assert tip[0] == 2  # the most-Democratic race


def test_tipping_point_handles_an_unreachable_majority():
    simulated = np.array([[1.0, 2.0, 3.0]])
    tip = _tipping_point(simulated, seats_not_up_dem=0, dem_majority=99)
    assert tip[0] == 0  # the least-Democratic race


def test_tipping_point_is_per_simulation():
    simulated = np.array([
        [3.0, 1.0, -2.0],   # order 0,1,2 -> decisive is 1
        [-2.0, 1.0, 3.0],   # order 2,1,0 -> decisive is 1
        [1.0, -2.0, 3.0],   # order 2,0,1 -> decisive is 0
    ])
    tip = _tipping_point(simulated, seats_not_up_dem=1, dem_majority=3)
    assert list(tip) == [1, 1, 0]


# ----------------------------------------------------------------- fixtures


def _mini_race_set(n_races=4):
    races = tuple(
        Race(
            id=f"r{i}", unit=f"S{i}", name=f"State {i}", special=False,
            incumbent_party="D" if i < 2 else "R",
            incumbent_status="open",
            pres_2024_dem_two_party=0.5, pres_2020_dem_two_party=0.5,
            region="south" if i % 2 == 0 else "west",
        )
        for i in range(n_races)
    )
    control = Control(
        total_seats=10,
        seats_not_up={"D": 3, "R": 3},
        seats_up={"D": 2, "R": 2},
        dem_seats_for_majority=6,
        rep_seats_for_majority=5,
        tiebreaker_party="R",
    )
    return RaceSet(
        cycle=2026, chamber="senate", election_date=date(2026, 11, 3),
        control=control, races=races,
    )


def _fake_idata(theta_final, n_grid=3, chains=2):
    """An InferenceData whose final grid point equals ``theta_final``."""
    import arviz as az

    theta_final = np.asarray(theta_final, dtype=float)
    n_draws, n_races = theta_final.shape
    per_chain = n_draws // chains

    theta = np.zeros((chains, per_chain, n_races, n_grid))
    reshaped = theta_final.reshape(chains, per_chain, n_races)
    for g in range(n_grid):
        theta[:, :, :, g] = reshaped

    eta = np.zeros((chains, per_chain, n_grid))

    posterior = xr.Dataset(
        {
            "theta": (("chain", "draw", "race", "grid"), theta),
            "eta": (("chain", "draw", "grid"), eta),
        },
        coords={
            "chain": np.arange(chains),
            "draw": np.arange(per_chain),
            "race": [f"r{i}" for i in range(n_races)],
            "grid": [f"2026-0{g+1}-01" for g in range(n_grid)],
        },
    )
    return az.InferenceData(posterior=posterior)


# -------------------------------------------------------------- simulation


def test_seat_totals_respect_the_chamber_baseline():
    races = _mini_race_set()
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)

    # Every race strongly Democratic -> Democrats win all 4 contested seats.
    idata = _fake_idata(np.full((100, 4), 5.0))
    sim = simulate_chamber(idata, races, cfg, fund, seed=1)

    assert sim.dem_seats.min() == 3 + 4      # not-up + all contested
    assert sim.dem_control_prob == 1.0
    assert sim.rep_control_prob == 0.0


def test_republicans_hold_the_chamber_on_a_tie():
    races = _mini_race_set()
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)

    # Democrats win exactly 2 of 4 -> 3 + 2 = 5 seats out of 10: a 5-5 tie.
    # Democrats need 6, so the tiebreaker hands control to Republicans.
    theta = np.tile(np.array([50.0, 50.0, -50.0, -50.0]), (100, 1))
    idata = _fake_idata(theta)
    sim = simulate_chamber(idata, races, cfg, fund, seed=1)

    assert set(np.unique(sim.dem_seats)) == {5}
    assert sim.dem_control_prob == 0.0
    assert sim.rep_control_prob == 1.0
    assert sim.tie_prob == 1.0


def test_simulation_expands_draws_by_sims_per_draw():
    races = _mini_race_set()
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)
    idata = _fake_idata(np.zeros((100, 4)))
    sim = simulate_chamber(idata, races, cfg, fund, seed=1)
    assert sim.n_sims == 100 * cfg.sims_per_draw


def test_election_day_error_widens_the_seat_distribution():
    """Without correlated error the seat distribution collapses; that is the
    whole reason the simulation layer exists."""
    races = _mini_race_set()
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)

    # Small but genuine leans. They must not be exactly zero: at exactly zero
    # the no-error case decides every race on floating-point sign noise, which
    # is maximally random rather than deterministic, and the comparison inverts.
    idata = _fake_idata(np.tile(np.array([0.06, -0.06, 0.10, -0.10]), (400, 1)))

    with_error = simulate_chamber(idata, races, cfg, fund, seed=2)

    no_error_cfg = dataclasses.replace(
        cfg,
        election_day_error=dataclasses.replace(
            cfg.election_day_error, national_sd=1e-9, state_sd=1e-9
        ),
    )
    without_error = simulate_chamber(idata, races, no_error_cfg, fund, seed=2)

    # With the leans fixed and no error, the outcome is fully determined.
    assert without_error.dem_seats.std() == pytest.approx(0.0)
    assert with_error.dem_seats.std() > 0.5


def test_correlated_error_is_actually_correlated():
    """Simulated outcomes in similar states should co-move."""
    races = _mini_race_set(n_races=4)
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)
    idata = _fake_idata(np.zeros((2000, 4)))
    sim = simulate_chamber(idata, races, cfg, fund, seed=3)

    empirical = np.corrcoef(sim.simulated_logit.T)
    # Off-diagonal correlation should be clearly positive: every race shares
    # the national error component.
    off_diag = empirical[np.triu_indices(4, k=1)]
    assert off_diag.min() > 0.3


def test_win_probabilities_are_in_range_and_consistent():
    races = _mini_race_set()
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)
    idata = _fake_idata(np.random.default_rng(0).normal(0, 0.3, size=(200, 4)))
    sim = simulate_chamber(idata, races, cfg, fund, seed=4)

    assert ((sim.dem_win_prob >= 0) & (sim.dem_win_prob <= 1)).all()
    assert sim.dem_control_prob + sim.rep_control_prob == pytest.approx(1.0)

    dist = sim.seat_distribution(races.control.total_seats)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_tipping_point_probabilities_sum_to_one():
    races = _mini_race_set()
    cfg = ModelConfig.load()
    fund = F.compute(races, cfg)
    idata = _fake_idata(np.random.default_rng(1).normal(0, 0.5, size=(200, 4)))
    sim = simulate_chamber(idata, races, cfg, fund, seed=5)
    assert sum(sim.tipping_point_probs().values()) == pytest.approx(1.0)
