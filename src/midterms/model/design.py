"""Turn the normalised poll table into the arrays the PyMC model indexes into.

Separating this from the model itself keeps the model readable and makes the
design testable without sampling: every index array, time grid and observation
variance can be checked directly.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ..config import ModelConfig, RaceSet
from ..data.polls import (
    APPROVAL_RACE_ID,
    NATIONAL_RACE_ID,
    NormalisedPoll,
    PollTable,
)
from ..fundamentals import Fundamentals, logit

#: Population screens, in a fixed order so indices are stable across runs.
POPULATIONS = ("lv", "rv", "a")

#: House-effect bucket shared by pollsters too infrequent to estimate their own.
POOLED_POLLSTER = "(infrequent pollsters)"


@dataclass(frozen=True)
class ModelData:
    """Everything the model needs, as plain arrays."""

    # -- structure ---------------------------------------------------------
    race_ids: tuple[str, ...]
    grid_dates: tuple[date, ...]
    pollster_names: tuple[str, ...]
    populations: tuple[str, ...]
    reference_population_index: int

    # -- observations, race polls first then national polls -----------------
    y: np.ndarray                 # logit two-party Dem share
    sampling_var: np.ndarray      # delta-method sampling variance on logit scale
    pollster_idx: np.ndarray
    population_idx: np.ndarray
    partisan_sign: np.ndarray

    race_poll_race_idx: np.ndarray
    race_poll_time_idx: np.ndarray
    national_poll_time_idx: np.ndarray
    approval_poll_time_idx: np.ndarray

    # -- priors ------------------------------------------------------------
    fundamentals_prior_mean: np.ndarray
    #: Lower-triangular Cholesky of the race-movement correlation matrix.
    #: Multiplying i.i.d. innovations by this correlates drift across similar
    #: races while leaving each race's marginal variance at 1, because the rows
    #: of a correlation Cholesky have unit norm.
    movement_chol: np.ndarray

    # -- metadata ----------------------------------------------------------
    polls_in_order: tuple[NormalisedPoll, ...]

    @property
    def n_races(self) -> int:
        return len(self.race_ids)

    @property
    def n_grid(self) -> int:
        return len(self.grid_dates)

    @property
    def n_steps(self) -> int:
        return self.n_grid - 1

    @property
    def n_race_polls(self) -> int:
        return len(self.race_poll_time_idx)

    @property
    def n_national_polls(self) -> int:
        return len(self.national_poll_time_idx)

    @property
    def n_approval_polls(self) -> int:
        return len(self.approval_poll_time_idx)

    @property
    def nonreference_population_indices(self) -> np.ndarray:
        return np.array(
            [i for i in range(len(self.populations)) if i != self.reference_population_index],
            dtype=int,
        )


def build_time_grid(
    earliest: date, election_date: date, grid_days: int
) -> tuple[date, ...]:
    """A backward-anchored grid so the final point is exactly election day.

    Anchoring on the election rather than the first poll means the forecast
    target is a real grid point, not an interpolation, and the grid does not
    shift underneath us as new polls arrive.
    """
    span = (election_date - earliest).days
    n_steps = max(1, math.ceil(span / grid_days))
    return tuple(
        election_date - timedelta(days=(n_steps - i) * grid_days)
        for i in range(n_steps + 1)
    )


def _time_index(poll_date: date, grid: tuple[date, ...], grid_days: int) -> int:
    """Nearest grid point to a poll's field date.

    With a 7-day grid the worst-case snapping error is 3.5 days, far smaller
    than the week-to-week noise in the polls themselves.
    """
    offset = (poll_date - grid[0]).days / grid_days
    return int(np.clip(round(offset), 0, len(grid) - 1))


def _effective_sample_size(poll: NormalisedPoll) -> float:
    """Sample size backing the two-party comparison specifically.

    A poll of 1000 where 15% are undecided or backing a third candidate carries
    only ~850 respondents' worth of information about the D-vs-R split, so we
    scale the nominal sample size by the two-party share of responses.
    """
    two_party_response = (poll.dem_pct + poll.rep_pct) / 100.0
    return max(1.0, poll.sample_size * min(1.0, max(0.05, two_party_response)))


def _sampling_variance(poll: NormalisedPoll, design_effect: float) -> float:
    """Delta-method variance of ``logit(p_hat)``.

        Var(logit(p)) ~ Var(p) / (p(1-p))^2 = 1 / (n p (1-p))

    inflated by the design effect to account for weighting and clustering.
    """
    p = float(np.clip(poll.two_party_dem, 0.02, 0.98))
    n_eff = _effective_sample_size(poll)
    return design_effect / (n_eff * p * (1.0 - p))


def build_model_data(
    table: PollTable,
    races: RaceSet,
    cfg: ModelConfig,
    fundamentals: Fundamentals,
) -> ModelData:
    """Assemble the design arrays."""
    race_ids = tuple(races.race_ids)
    race_position = {rid: i for i, rid in enumerate(race_ids)}

    race_polls = sorted(table.race_polls, key=lambda p: (p.race_id, p.field_date))
    national_polls = sorted(table.national, key=lambda p: p.field_date)
    # The model only creates a latent approval series when the feature is on,
    # so the design must drop the polls when it is off. One switch, honoured in
    # both places, or the observation vector and the latent vector disagree.
    approval_polls = (
        sorted(table.approval, key=lambda p: p.field_date)
        if cfg.national_environment.approval.enabled
        else []
    )
    # The model concatenates its latent means in this order, so it must match.
    ordered = tuple(race_polls + national_polls + approval_polls)

    if not ordered:
        raise ValueError("no usable polls: cannot fit the model")

    earliest = min(p.field_date for p in ordered)
    grid = build_time_grid(earliest, races.election_date, cfg.grid_days)

    # House effects are estimated only for pollsters that appear often enough
    # for "house effect" to mean anything; the long tail shares one bucket.
    poll_counts_by_pollster = Counter(p.pollster for p in ordered)
    threshold = cfg.polls.min_polls_for_house_effect
    house_key = {
        name: (name if count >= threshold else POOLED_POLLSTER)
        for name, count in poll_counts_by_pollster.items()
    }
    pollster_names = tuple(sorted(set(house_key.values())))
    pollster_position = {name: i for i, name in enumerate(pollster_names)}

    reference_index = POPULATIONS.index(cfg.polls.population_reference)

    y = np.array([logit(p.two_party_dem) for p in ordered], dtype=float)
    sampling_var = np.array(
        [_sampling_variance(p, cfg.polls.design_effect) for p in ordered], dtype=float
    )
    pollster_idx = np.array(
        [pollster_position[house_key[p.pollster]] for p in ordered], dtype=int
    )
    population_idx = np.array(
        [POPULATIONS.index(p.population) if p.population in POPULATIONS else reference_index
         for p in ordered],
        dtype=int,
    )
    partisan_sign = np.array([p.partisan_sign for p in ordered], dtype=float)

    race_poll_race_idx = np.array(
        [race_position[p.race_id] for p in race_polls], dtype=int
    )
    race_poll_time_idx = np.array(
        [_time_index(p.field_date, grid, cfg.grid_days) for p in race_polls], dtype=int
    )
    national_poll_time_idx = np.array(
        [_time_index(p.field_date, grid, cfg.grid_days) for p in national_polls], dtype=int
    )
    approval_poll_time_idx = np.array(
        [_time_index(p.field_date, grid, cfg.grid_days) for p in approval_polls], dtype=int
    )

    # Fundamentals are computed over races in config order; assert alignment
    # rather than trusting it, because a silent misalignment would attach every
    # race's prior to the wrong race.
    if fundamentals.race_ids != race_ids:
        raise ValueError("fundamentals race order does not match the race set")

    assert all(p.race_id == NATIONAL_RACE_ID for p in national_polls)
    assert all(p.race_id == APPROVAL_RACE_ID for p in approval_polls)

    from .correlation import cholesky_factor, correlation_matrix

    movement_chol = cholesky_factor(
        correlation_matrix(fundamentals, cfg.race.movement_correlation)
    )

    return ModelData(
        race_ids=race_ids,
        grid_dates=grid,
        pollster_names=pollster_names,
        populations=POPULATIONS,
        reference_population_index=reference_index,
        y=y,
        sampling_var=sampling_var,
        pollster_idx=pollster_idx,
        population_idx=population_idx,
        partisan_sign=partisan_sign,
        race_poll_race_idx=race_poll_race_idx,
        race_poll_time_idx=race_poll_time_idx,
        national_poll_time_idx=national_poll_time_idx,
        approval_poll_time_idx=approval_poll_time_idx,
        fundamentals_prior_mean=fundamentals.prior_mean,
        movement_chol=movement_chol,
        polls_in_order=ordered,
    )
