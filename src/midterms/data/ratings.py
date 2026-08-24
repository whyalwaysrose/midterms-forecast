"""Pollster quality: how much to trust each poll beyond its sample size.

The model already learns each pollster's *bias* from the data, as a house
effect. What it cannot learn well is each pollster's *precision*, because most
pollsters in a single cycle have too few polls to estimate a variance from. So
every poll currently shares one non-sampling noise term: a 400-person poll from
an F-rated shop and a 1,200-person poll from an A+ shop differ only by sample
size, which is exactly the mistake sample size alone invites.

FiveThirtyEight's pollster ratings measure this directly, across twenty-five
years and 500-plus organisations. This module turns a rating into a multiplier
on that noise term.

**The multiplier is centred.** ``polls.excess_sd_prior`` was fitted against all
historical polls, so it already describes an average pollster. Multiplying it by
raw ratings -- whose mean is +0.49 points, not zero -- would inflate every
poll's noise and quietly undo that calibration. Centring on the poll-weighted
mean of our own field keeps the overall scale where calibration put it and lets
the ratings do the one job they are here for: saying who is better than whom.

**Unrated pollsters get exactly 1.0.** 538 stopped updating in 2025, so the
2026 entrants -- Quantus, Verasight, Focaldata, Tavern Research and others --
have no rating and cannot get one. Treating them as average is a real
assumption, and probably a slightly generous one: a pollster with no record is
riskier than one with a good record. But inventing a penalty would be asserting
a number nothing measured, so they are treated as average and the count is
logged.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .pollsters import normalise

log = logging.getLogger(__name__)

#: Our canonical pollster name -> the name 538 files it under.
#:
#: Two independent vocabularies, so these are unavoidable. Only genuine
#: same-organisation pairs belong here. HarrisX is deliberately absent: 538
#: rates "Harris Insights & Analytics", which is a different firm, and guessing
#: otherwise would hand one pollster another's record.
RATINGS_ALIASES: dict[str, str] = {
    "university of new hampshire": "University of New Hampshire Survey Center",
    "glengariff group inc": "Glengariff Group",
}

#: Bounds on the raw multiplier, before centring.
#:
#: The rating range implies roughly 0.79x to 1.46x. Clamping stops a single
#: extreme or thinly-evidenced rating from dominating a race that only one
#: pollster has surveyed.
MIN_MULTIPLIER = 0.70
MAX_MULTIPLIER = 1.60


@dataclass(frozen=True)
class PollsterRating:
    """One row of the ratings file, reduced to what the model uses."""

    pollster: str
    polls_analyzed: int
    banned: bool
    grade: str
    #: Mean-reverted error relative to a comparable poll, in points. Negative is
    #: better. Mean-reverted matters: a shop with four lucky polls should not be
    #: read as excellent, and 538's raw plus-minus would say it was.
    predictive_plus_minus: float
    #: 538's estimate of the pollster's partisan lean, positive toward Democrats.
    mean_reverted_bias: float


def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


@dataclass
class PollsterRatings:
    """The vendored ratings file, indexed for lookup by pollster name."""

    by_key: dict[str, PollsterRating]
    #: Mean expected error across the file, in points. The denominator that
    #: converts a plus-minus in points into a proportional multiplier.
    mean_expected_error: float

    @classmethod
    def load(cls, path: Path | None = None) -> PollsterRatings:
        from ..paths import RATINGS_FILE

        path = path or RATINGS_FILE
        by_key: dict[str, PollsterRating] = {}
        expected: list[float] = []

        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Pollster") or "").strip()
                if not name:
                    continue
                by_key[normalise(name)] = PollsterRating(
                    pollster=name,
                    polls_analyzed=_int(row.get("Polls Analyzed", "")),
                    banned=(row.get("Banned by 538") or "").strip().lower() == "yes",
                    grade=(row.get("538 Grade") or "").strip(),
                    predictive_plus_minus=_float(row.get("Predictive Plus-Minus", "")),
                    mean_reverted_bias=_float(row.get("Mean-Reverted Bias", "")),
                )
                error = _float(row.get("Simple Expected Error", ""), 0.0)
                if error > 0:
                    expected.append(error)

        if not by_key:
            raise ValueError(f"no pollster ratings parsed from {path}")
        return cls(by_key=by_key, mean_expected_error=sum(expected) / len(expected))

    # -- lookup ------------------------------------------------------------

    def find(self, name: str) -> PollsterRating | None:
        """The rating for a pollster, or None if 538 never rated it.

        Tries the full name, then the alias table, then the components of a
        sponsor/pollster name: our feed carries "CNN/SSRS" where 538 rates
        "SSRS", and the pollster is the part that has a track record. The full
        name is always tried first, because 538 rates some partnerships in their
        own right -- "The New York Times/Siena College" is a rated name.
        """
        key = normalise(name)
        if key in self.by_key:
            return self.by_key[key]

        aliased = RATINGS_ALIASES.get(key)
        if aliased and normalise(aliased) in self.by_key:
            return self.by_key[normalise(aliased)]

        if "/" in name:
            for part in reversed(name.split("/")):
                part_key = normalise(part)
                if part_key in self.by_key:
                    return self.by_key[part_key]
        return None

    def banned_names(self, names: Iterable[str]) -> set[str]:
        """Which of ``names`` 538 refused to accept polls from."""
        return {n for n in names if (r := self.find(n)) is not None and r.banned}

    # -- the number the model consumes -------------------------------------

    def raw_multiplier(self, name: str) -> float | None:
        """Proportional noise multiplier implied by a rating, before centring."""
        rating = self.find(name)
        if rating is None:
            return None
        implied = (
            self.mean_expected_error + rating.predictive_plus_minus
        ) / self.mean_expected_error
        return min(MAX_MULTIPLIER, max(MIN_MULTIPLIER, implied))

    def noise_multipliers(self, pollsters: Sequence[str]) -> dict[str, float]:
        """Per-pollster noise multipliers, centred on the field given.

        ``pollsters`` is one entry **per poll**, not per pollster, so the
        centring is weighted by how much each pollster actually contributes.
        The result has a poll-weighted mean of 1.0 over rated polls, so adding
        ratings redistributes trust without changing its total.
        """
        raw = {name: self.raw_multiplier(name) for name in set(pollsters)}
        rated = [raw[n] for n in pollsters if raw[n] is not None]

        if not rated:
            log.warning("no polls matched a pollster rating; weighting is uniform")
            return dict.fromkeys(set(pollsters), 1.0)

        centre = sum(rated) / len(rated)
        multipliers = {
            name: (value / centre if value is not None else 1.0)
            for name, value in raw.items()
        }

        unrated = sorted(n for n, v in raw.items() if v is None)
        if unrated:
            n_unrated = sum(1 for n in pollsters if raw[n] is None)
            log.info(
                "pollster ratings: %d of %d polls unrated (treated as average); "
                "%d pollsters incl. %s",
                n_unrated, len(pollsters), len(unrated), ", ".join(unrated[:5]),
            )
        return multipliers
