"""Structural (non-poll) priors for each race.

The fundamentals answer one question: *before seeing a single poll, where would
we expect this race to sit if the national environment were exactly tied?*

Keeping the "if the national environment were tied" clause is what makes the
decomposition work. Each state's presidential result is converted into a
partisan **lean** — its logit share minus the nation's — so the national swing
is not baked into the race baseline. The environment then enters exactly once,
through the generic-ballot random walk in the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig, Race, RaceSet


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """Log-odds. Inputs are clipped away from 0 and 1 for safety."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(x: float | np.ndarray) -> float | np.ndarray:
    """Inverse of :func:`logit`."""
    return 1.0 / (1.0 + np.exp(-x))


def logit_to_margin(x: float | np.ndarray) -> float | np.ndarray:
    """Convert a logit two-party share to a Democratic margin in points."""
    return 100.0 * (2.0 * inv_logit(x) - 1.0)


def share_to_margin(p: float | np.ndarray) -> float | np.ndarray:
    """Convert a two-party share to a Democratic margin in points."""
    return 100.0 * (2.0 * np.asarray(p) - 1.0)


@dataclass(frozen=True)
class Fundamentals:
    """Per-race prior means and the covariates used for correlated error."""

    race_ids: tuple[str, ...]
    #: Prior mean of each race's baseline logit share under a tied environment.
    prior_mean: np.ndarray
    #: Presidential lean relative to the nation, 2024 and 2020.
    lean_2024: np.ndarray
    lean_2020: np.ndarray
    regions: tuple[str, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.race_ids, self.prior_mean.tolist(), strict=True))


def race_lean(race: Race, cfg: ModelConfig) -> tuple[float, float]:
    """Presidential lean of a race's state relative to the nation, 2024 and 2020."""
    lean_2024 = float(
        logit(race.pres_2024_dem_two_party) - logit(cfg.national.pres_2024_dem_two_party)
    )
    lean_2020 = float(
        logit(race.pres_2020_dem_two_party) - logit(cfg.national.pres_2020_dem_two_party)
    )
    return lean_2024, lean_2020


def compute(races: RaceSet, cfg: ModelConfig) -> Fundamentals:
    """Build the fundamentals prior for every race in ``races``."""
    f = cfg.fundamentals

    ids: list[str] = []
    prior: list[float] = []
    leans_2024: list[float] = []
    leans_2020: list[float] = []
    regions: list[str] = []

    for race in races.races:
        lean_2024, lean_2020 = race_lean(race, cfg)

        # Blend the two presidential cycles, then shrink slightly toward the
        # centre: Senate outcomes track presidential lean closely but not
        # perfectly, and safe states are a little less lopsided downballot.
        blended = f.weight_pres_2024 * lean_2024 + f.weight_pres_2020 * lean_2020
        baseline = f.lean_shrinkage * blended

        # Incumbency, signed toward the party that currently holds the seat.
        bonus = f.incumbency_bonus[race.incumbent_status]
        baseline += bonus if race.incumbent_party == "D" else -bonus

        ids.append(race.id)
        prior.append(baseline)
        leans_2024.append(lean_2024)
        leans_2020.append(lean_2020)
        regions.append(race.region)

    return Fundamentals(
        race_ids=tuple(ids),
        prior_mean=np.asarray(prior, dtype=float),
        lean_2024=np.asarray(leans_2024, dtype=float),
        lean_2020=np.asarray(leans_2020, dtype=float),
        regions=tuple(regions),
    )
