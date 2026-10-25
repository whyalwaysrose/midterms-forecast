"""Other forecasters' published numbers, for the dashboard's comparison strip.

`data/rivals/forecasts.csv` is a dated, sourced registry maintained by hand,
because there is nothing to fetch: every rival publishes a rendered page and
none publishes a feed. Each row records when the rival said it, when we read
it, how we got at it, and a note naming the exact wording -- so a figure can be
checked rather than taken on trust, and a paywalled or prose-only number is
visibly that.

This module is the read side, shared by the payload. It deliberately does not
fetch anything: a rival's site being down, slow or newly paywalled must never
delay or fail a forecast.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .. import paths

log = logging.getLogger(__name__)

REGISTRY = paths.DATA_DIR / "rivals" / "forecasts.csv"

#: Beyond this a rival's number is not a comparison, it is a historical note.
#: Sixteen days is already a long time in a closing campaign; six months is the
#: Economist's April figure, which was also computed on district lines that ten
#: states have since redrawn.
MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class RivalForecast:
    forecaster: str
    chamber: str
    as_of: date
    dem_control_prob: float
    dem_seats_median: int | None
    access: str
    source_url: str
    note: str

    def age_days(self, today: date) -> int:
        return (today - self.as_of).days


def _row_to_forecast(row: dict) -> RivalForecast | None:
    try:
        prob = float(row["dem_control_prob"])
    except (TypeError, ValueError, KeyError):
        return None
    if not 0.0 <= prob <= 1.0:
        return None
    try:
        as_of = date.fromisoformat(row["as_of"].strip())
    except (TypeError, ValueError, KeyError):
        return None

    seats = (row.get("dem_seats_median") or "").strip()
    try:
        median = int(seats) if seats else None
    except ValueError:
        median = None

    return RivalForecast(
        forecaster=row["forecaster"].strip(),
        chamber=row["chamber"].strip(),
        as_of=as_of,
        dem_control_prob=prob,
        dem_seats_median=median,
        access=(row.get("access") or "").strip(),
        source_url=(row.get("source_url") or "").strip(),
        note=(row.get("note") or "").strip(),
    )


def load(path: Path | None = None) -> list[RivalForecast]:
    """Every readable row. Unparseable rows are skipped, not fatal."""
    path = path or REGISTRY
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            parsed = _row_to_forecast(row)
            if parsed is not None:
                out.append(parsed)
    return out


def latest_for_chamber(
    chamber: str, today: date | None = None, path: Path | None = None
) -> list[RivalForecast]:
    """The most recent usable figure per forecaster, newest first by probability.

    One row per forecaster: the registry keeps every reading ever taken, and
    showing six months of 50+1 next to each other would be a history, not a
    comparison. Anything older than :data:`MAX_AGE_DAYS` is dropped entirely
    rather than shown with a caveat -- a six-month-old number on superseded
    district lines is not a view about this election.
    """
    today = today or date.today()
    best: dict[str, RivalForecast] = {}
    for row in load(path):
        if row.chamber != chamber:
            continue
        if row.age_days(today) > MAX_AGE_DAYS or row.age_days(today) < 0:
            continue
        current = best.get(row.forecaster)
        if current is None or row.as_of > current.as_of:
            best[row.forecaster] = row
    return sorted(best.values(), key=lambda r: -r.dem_control_prob)
