"""From posterior to seats.

The PyMC model tells us where each race sits *given that the polls are
unbiased*. History says they are not. This module adds the missing piece — a
correlated election-day error — and propagates the whole thing through the
chamber's seat arithmetic.

Why this layer exists at all: a race-by-race model with independent errors
produces a wildly overconfident chamber forecast, because 35 independent coin
flips concentrate hard around their mean. Real polling misses are correlated,
which fattens the tails of the seat distribution enormously. That correlation
is the difference between a seat forecast that is merely arithmetic and one
that is honest.

Note on timing: the latent random walk in the model already runs to election
day, so drift between the last poll and the election is accounted for there.
What is added here is *bias*, not drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ModelConfig, RaceSet
from ..fundamentals import Fundamentals, inv_logit
from .correlation import cholesky_factor, correlation_matrix


@dataclass(frozen=True)
class SimulationResult:
    """Chamber-level and race-level output of the simulation."""

    race_ids: tuple[str, ...]
    #: (n_sims, n_races) simulated logit two-party Dem share on election day
    simulated_logit: np.ndarray
    #: (n_sims,) Democratic-caucus seat totals
    dem_seats: np.ndarray
    #: index into race_ids of the tipping-point race in each simulation
    tipping_point_idx: np.ndarray

    dem_win_prob: np.ndarray
    dem_control_prob: float
    rep_control_prob: float
    tie_prob: float

    correlation: np.ndarray

    @property
    def n_sims(self) -> int:
        return self.simulated_logit.shape[0]

    def margin_quantiles(self, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
        """Democratic margin (points) at the given quantiles, per race."""
        shares = inv_logit(self.simulated_logit)
        margins = 100.0 * (2.0 * shares - 1.0)
        return np.quantile(margins, quantiles, axis=0)

    def seat_distribution(self, total_seats: int) -> dict[int, float]:
        """Probability mass over Democratic-caucus seat totals."""
        counts = np.bincount(self.dem_seats, minlength=total_seats + 1)
        probs = counts / counts.sum()
        return {i: float(p) for i, p in enumerate(probs) if p > 0}

    def tipping_point_probs(self) -> dict[str, float]:
        counts = np.bincount(self.tipping_point_idx, minlength=len(self.race_ids))
        total = counts.sum()
        return {
            race_id: float(counts[i] / total)
            for i, race_id in enumerate(self.race_ids)
            if counts[i] > 0
        }


def _final_logit_draws(idata) -> np.ndarray:
    """Posterior draws of the latent state on election day, shape (draws, races)."""
    theta = idata.posterior["theta"]
    final = theta.isel(grid=-1)
    # stack chain and draw into a single sample dimension
    return final.stack(sample=("chain", "draw")).transpose("sample", "race").to_numpy()


def simulate_chamber(
    idata,
    races: RaceSet,
    cfg: ModelConfig,
    fundamentals: Fundamentals,
    seed: int | None = None,
) -> SimulationResult:
    """Draw the seat distribution implied by the posterior."""
    rng = np.random.default_rng(seed if seed is not None else cfg.sampler.seed)

    theta_final = _final_logit_draws(idata)          # (n_draws, n_races)
    n_draws, n_races = theta_final.shape
    reps = max(1, cfg.sims_per_draw)

    # Recycle each posterior draw with fresh election-day error. This keeps the
    # posterior fully represented while smoothing the seat histogram.
    base = np.repeat(theta_final, reps, axis=0)      # (n_sims, n_races)
    n_sims = base.shape[0]

    ede = cfg.election_day_error
    corr = correlation_matrix(fundamentals, ede.correlation)
    chol = cholesky_factor(corr)

    # A single national miss applied to every race alike...
    #
    # Widened by the standard error of the generic-ballot correction the model
    # applied inside `theta`. That correction is an estimate from seven cycles,
    # not a constant: subtracting it and then reporting the interval as though
    # it were exact would claim precision the seven cycles do not support. Added
    # in quadrature because the two are independent -- how far the generic
    # ballot sits from the national vote, and how far the polls miss on the day.
    bias_se = getattr(cfg.national_environment, "generic_ballot_bias_se", 0.0)
    national_sd = float(np.hypot(ede.national_sd, bias_se))
    national_error = rng.normal(0.0, national_sd, size=(n_sims, 1))
    # ...plus a state-level miss correlated across politically similar states.
    z = rng.standard_normal(size=(n_sims, n_races))
    state_error = ede.state_sd * (z @ chol.T)

    simulated = base + national_error + state_error

    dem_wins = simulated > 0.0
    seats_not_up_dem = races.control.seats_not_up.get("D", 0)
    dem_seats = seats_not_up_dem + dem_wins.sum(axis=1)

    threshold = races.control.dem_seats_for_majority
    dem_control_prob = float(np.mean(dem_seats >= threshold))
    rep_threshold = races.control.rep_seats_for_majority
    rep_seats = races.control.total_seats - dem_seats
    rep_control_prob = float(np.mean(rep_seats >= rep_threshold))
    # With a tiebreaker the two are exhaustive; a literal 50-50 split is still
    # worth reporting separately because it is the headline "tie" scenario.
    tie_prob = float(np.mean(dem_seats == races.control.total_seats - dem_seats))

    tipping = _tipping_point(simulated, seats_not_up_dem, threshold)

    return SimulationResult(
        race_ids=tuple(races.race_ids),
        simulated_logit=simulated,
        dem_seats=dem_seats.astype(int),
        tipping_point_idx=tipping,
        dem_win_prob=dem_wins.mean(axis=0),
        dem_control_prob=dem_control_prob,
        rep_control_prob=rep_control_prob,
        tie_prob=tie_prob,
        correlation=corr,
    )


def _tipping_point(
    simulated: np.ndarray, seats_not_up_dem: int, dem_majority: int
) -> np.ndarray:
    """Identify the decisive race in each simulation.

    Order the contested races from most to least Democratic in that simulation
    and walk down the list, accumulating Democratic seats. The race at which the
    running total reaches the majority threshold is the one that decided the
    chamber — the "tipping-point" race. It is the seat both parties would most
    want to win, and it is usually more informative than any single race's win
    probability.
    """
    n_sims, n_races = simulated.shape
    order = np.argsort(-simulated, axis=1)                    # most-Dem first
    seats_needed = dem_majority - seats_not_up_dem

    if seats_needed <= 0:
        return order[:, 0]
    if seats_needed > n_races:
        return order[:, -1]

    # The (seats_needed-1)-th race in the ordering is the one that supplies the
    # majority-making seat.
    return order[:, seats_needed - 1]


def national_environment_summary(idata) -> dict[str, float]:
    """Posterior for the national generic-ballot environment on election day."""
    eta = idata.posterior["eta"].isel(grid=-1).to_numpy().ravel()
    share = inv_logit(eta)
    margin = 100.0 * (2.0 * share - 1.0)
    return {
        "dem_margin_median": float(np.median(margin)),
        "dem_margin_p05": float(np.quantile(margin, 0.05)),
        "dem_margin_p95": float(np.quantile(margin, 0.95)),
        "dem_two_party_share_median": float(np.median(share)),
    }
