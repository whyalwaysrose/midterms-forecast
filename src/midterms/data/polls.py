"""Normalise raw VoteHub records into the tidy table the model consumes.

Each usable poll becomes one row with a two-party Democratic share, an
effective sample size, a field date, and the covariates the measurement model
adjusts for (pollster, population screen, partisan sponsorship).

Rejections are counted and reported rather than silently dropped: a sudden jump
in the "no two-party matchup" count is exactly the kind of upstream change that
would otherwise quietly distort a forecast.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any

from ..config import ModelConfig, RaceSet
from .pollsters import unify as unify_pollsters
from .roster import SIDE_D, SIDE_OTHER, SIDE_R, Roster
from .votehub import is_primary_subject

log = logging.getLogger(__name__)

#: Sentinel race id for national generic-ballot polls.
NATIONAL_RACE_ID = "__national__"

#: Sentinel for presidential-approval polls, a second national indicator.
APPROVAL_RACE_ID = "__approval__"

#: Approval polls are collected for several subjects; only the president's
#: matter for a midterm.
APPROVAL_SUBJECT = "Donald Trump"


@dataclass(frozen=True)
class NormalisedPoll:
    """One poll, reduced to what the measurement model needs."""

    poll_id: str
    race_id: str
    pollster: str
    field_date: date
    start_date: date
    end_date: date
    sample_size: int
    population: str
    dem_pct: float
    rep_pct: float
    two_party_dem: float
    other_pct: float
    partisan_sign: int          # +1 D-sponsored, -1 R-sponsored, 0 neutral
    sponsors: tuple[str, ...]
    url: str
    dem_candidate: str
    rep_candidate: str

    @property
    def margin(self) -> float:
        """Democratic margin in percentage points of the two-party vote."""
        return 100.0 * (2.0 * self.two_party_dem - 1.0)


@dataclass
class PollTable:
    """Normalised polls plus a record of what was thrown away and why."""

    polls: list[NormalisedPoll]
    rejections: Counter = field(default_factory=Counter)
    unknown_candidates: dict[str, set[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.polls)

    def for_race(self, race_id: str) -> list[NormalisedPoll]:
        return [p for p in self.polls if p.race_id == race_id]

    @property
    def national(self) -> list[NormalisedPoll]:
        return self.for_race(NATIONAL_RACE_ID)

    @property
    def approval(self) -> list[NormalisedPoll]:
        return self.for_race(APPROVAL_RACE_ID)

    @property
    def race_polls(self) -> list[NormalisedPoll]:
        return [
            p for p in self.polls
            if p.race_id not in (NATIONAL_RACE_ID, APPROVAL_RACE_ID)
        ]

    def counts_by_race(self) -> Counter:
        return Counter(p.race_id for p in self.race_polls)

    def summary(self) -> str:
        lines = [
            f"{len(self.race_polls)} race polls, {len(self.national)} generic-ballot polls, "
            f"{len(self.approval)} approval polls",
        ]
        if self.rejections:
            lines.append("rejected: " + ", ".join(
                f"{reason}={n}" for reason, n in self.rejections.most_common()
            ))
        return "\n".join(lines)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _field_date(start: date | None, end: date | None) -> date | None:
    """Midpoint of the field period.

    A poll informs us about opinion while it was in the field, not on the day
    it was published, so the midpoint is the right time index.
    """
    if end is None:
        return start
    if start is None:
        return end
    return start + (end - start) / 2


def _partisan_sign(record: dict[str, Any]) -> int:
    """+1 for a Democratic-aligned sponsor, -1 for Republican, 0 for neutral."""
    partisan = record.get("partisan")
    if partisan is None:
        return 0
    token = str(partisan).strip().upper()
    if token in {"DEM", "D", "DEMOCRAT", "DEMOCRATIC"}:
        return 1
    if token in {"REP", "R", "REPUBLICAN", "GOP"}:
        return -1
    return 0


def _population(record: dict[str, Any], default: str) -> str:
    value = record.get("population")
    if value is None:
        return default
    token = str(value).strip().lower()
    return token if token in {"lv", "rv", "a"} else default


def _normalise_one(
    record: dict[str, Any],
    race_id: str,
    roster: Roster,
    cfg: ModelConfig,
    table: PollTable,
) -> NormalisedPoll | None:
    """Reduce one raw record to a :class:`NormalisedPoll`, or reject it."""
    poll_id = str(record.get("id", ""))

    start = _parse_date(record.get("start_date"))
    end = _parse_date(record.get("end_date"))
    fielded = _field_date(start, end)
    if fielded is None:
        table.rejections["no usable date"] += 1
        return None

    answers = record.get("answers") or []
    if not answers:
        table.rejections["no answers"] += 1
        return None

    # Group answers by party side. Multiple entries on one side means the poll
    # is testing several hypothetical matchups at once and cannot be reduced to
    # a single two-party number without guessing.
    sides: dict[str, list[tuple[str, float]]] = {SIDE_D: [], SIDE_R: [], SIDE_OTHER: []}
    for answer in answers:
        choice = str(answer.get("choice", ""))
        try:
            pct = float(answer.get("pct"))
        except (TypeError, ValueError):
            continue
        sides[roster.resolve(race_id, choice)].append((choice, pct))

    unknown = roster.unknown_names(race_id, [str(a.get("choice", "")) for a in answers])
    if unknown:
        table.unknown_candidates.setdefault(race_id, set()).update(unknown)

    if len(sides[SIDE_D]) != 1 or len(sides[SIDE_R]) != 1:
        table.rejections["no single D-vs-R matchup"] += 1
        return None

    dem_candidate, dem_pct = sides[SIDE_D][0]
    rep_candidate, rep_pct = sides[SIDE_R][0]
    two_party_total = dem_pct + rep_pct
    if two_party_total <= 0:
        table.rejections["zero two-party total"] += 1
        return None

    sample_size = record.get("sample_size")
    try:
        sample_size = int(sample_size) if sample_size else cfg.polls.default_sample_size
    except (TypeError, ValueError):
        sample_size = cfg.polls.default_sample_size
    if sample_size < cfg.polls.min_sample_size:
        table.rejections["sample too small"] += 1
        return None

    return NormalisedPoll(
        poll_id=poll_id,
        race_id=race_id,
        pollster=str(record.get("pollster") or "Unknown"),
        field_date=fielded,
        start_date=start or fielded,
        end_date=end or fielded,
        sample_size=sample_size,
        population=_population(record, cfg.polls.population_reference),
        dem_pct=dem_pct,
        rep_pct=rep_pct,
        two_party_dem=dem_pct / two_party_total,
        other_pct=sum(pct for _, pct in sides[SIDE_OTHER]),
        partisan_sign=_partisan_sign(record),
        sponsors=tuple(record.get("sponsors") or ()),
        url=str(record.get("url") or ""),
        dem_candidate=dem_candidate,
        rep_candidate=rep_candidate,
    )


def build_poll_table(
    raw_by_type: dict[str, Iterable[dict[str, Any]]],
    races: RaceSet,
    cfg: ModelConfig,
    roster: Roster,
    as_of: date | None = None,
    race_poll_type: str = "us-senator",
    national_poll_type: str = "generic-ballot",
    approval_poll_type: str | None = "approval",
) -> PollTable:
    """Turn raw API records into the model's input table.

    ``as_of`` lets a run be reproduced at an earlier date (used by the
    backtest): polls fielded after it are excluded.
    """
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=cfg.polls.max_age_days)
    table = PollTable(polls=[])

    subject_to_race = {
        race.subject_for(races.cycle): race.id for race in races.races
    }

    # --- race polls -------------------------------------------------------
    for record in raw_by_type.get(race_poll_type, []):
        subject = str(record.get("subject", ""))
        if is_primary_subject(subject):
            table.rejections["primary subject"] += 1
            continue
        race_id = subject_to_race.get(subject)
        if race_id is None:
            table.rejections[f"unmapped subject: {subject}"] += 1
            continue

        poll = _normalise_one(record, race_id, roster, cfg, table)
        if poll is None:
            continue
        if poll.field_date > as_of:
            table.rejections["fielded after as_of"] += 1
            continue
        if poll.field_date < cutoff:
            table.rejections["older than max_age_days"] += 1
            continue
        table.polls.append(poll)

    # --- national generic ballot -----------------------------------------
    for record in raw_by_type.get(national_poll_type, []):
        poll = _normalise_one(record, NATIONAL_RACE_ID, roster, cfg, table)
        if poll is None:
            continue
        if poll.field_date > as_of:
            table.rejections["fielded after as_of"] += 1
            continue
        if poll.field_date < cutoff:
            table.rejections["older than max_age_days"] += 1
            continue
        table.polls.append(poll)

    # --- presidential approval -------------------------------------------
    if approval_poll_type:
        for record in raw_by_type.get(approval_poll_type, []):
            if str(record.get("subject", "")) != APPROVAL_SUBJECT:
                continue
            poll = _normalise_one(record, APPROVAL_RACE_ID, roster, cfg, table)
            if poll is None:
                continue
            if poll.field_date > as_of or poll.field_date < cutoff:
                continue
            table.polls.append(poll)

    # One pollster, one identity. Done here rather than per record because
    # choosing between "GrayHouse" and "Grayhouse" needs to see the whole feed:
    # the majority spelling wins. Without this the same pollster is two, and its
    # house effect is estimated twice from half the evidence each time.
    mapping = unify_pollsters(p.pollster for p in table.polls)
    renamed = {raw: name for raw, name in mapping.items() if raw != name}
    if renamed:
        table.polls[:] = [
            replace(p, pollster=mapping[p.pollster]) if p.pollster in renamed else p
            for p in table.polls
        ]
        for raw, name in sorted(renamed.items()):
            log.info("pollster: merged %r into %r", raw, name)

    # Pollsters 538 refused to accept polls from, dropped here rather than
    # down-weighted. The ban list is mostly about suspected fabrication, and a
    # fabricated number cannot be weighted into usefulness: it looks precise,
    # not noisy, so a variance adjustment would make the model MORE confident.
    if cfg.polls.pollster_ratings.exclude_banned:
        from .ratings import PollsterRatings

        banned = PollsterRatings.load().banned_names({p.pollster for p in table.polls})
        if banned:
            dropped = Counter(p.pollster for p in table.polls if p.pollster in banned)
            table.polls[:] = [p for p in table.polls if p.pollster not in banned]
            for name, n in dropped.most_common():
                table.rejections[f"pollster banned by 538: {name}"] += n

    table.polls.sort(key=lambda p: (p.race_id, p.field_date))
    log.info("poll table: %s", table.summary().replace("\n", "; "))
    return table


def pollster_index(polls: Sequence[NormalisedPoll]) -> tuple[list[str], dict[str, int]]:
    """Stable alphabetical index of pollsters appearing in ``polls``."""
    names = sorted({p.pollster for p in polls})
    return names, {name: i for i, name in enumerate(names)}
