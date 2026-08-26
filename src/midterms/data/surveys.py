"""One survey, one observation.

A pollster who tests several hypothetical matchups publishes several rows, and
VoteHub passes each through as its own poll. They are not separate polls: they
are one sample of one set of people, asked more than one question. A TIPP survey
of 1,163 Michigan voters arrived as five records --

    El-Sayed  vs Rogers   D+1.2
    El-Sayed  vs Rogers    0.0
    Stevens   vs Rogers   D+7.9
    Stevens   vs Rogers   D+8.4
    McMorrow  vs Rogers   D+3.4

-- and the model treated all five as independent evidence. Two things go wrong.

**Bias**, which turned out to be the serious one. The matchups are not
equivalent. Stevens ran eight points better than El-Sayed in the same survey,
and El-Sayed is the nominee, so averaging them in credits the Democrats with a
candidate who is not on the ballot. New Hampshire's win probability jumped from
81.6% to 91.8% in one day on two records from a single Saint Anselm survey.
Correcting it moved Michigan from 64.1% to 58.9%, New Hampshire from 91.8% to
88.8%, and Democratic control of the chamber from 71.0% to 68.5%.

**Precision**, which matters less than the arithmetic suggests. Counting one
sample five times ought to shrink the apparent standard error by half, but
measured against the real feed the 90% interval on Maine barely moved -- 19.9 to
20.1 points -- despite its usable polls falling from 34 to 3. The election-day
error and the shared national environment dominate a race's final interval, and
a hierarchical model borrows strength across races by design, so the marginal
poll beyond the first few buys little width. Worth knowing before expecting this
change to visibly widen anything: it corrects where the estimate sits, not how
sure the model is.

Across the 2026 feed 90 of 252 race polls are repeat or superseded readings, and
they concentrate in the competitive races that decide control, because those are
the ones worth polling several ways.

The fix is to reduce each survey to one observation, preferring the matchup that
is actually happening. Which matchup that is comes from the polls themselves:
once a primary is settled, pollsters stop testing the losers, so whoever the
last few surveys still name is live. Where a primary genuinely has not happened
-- Florida, Massachusetts, Minnesota and New Hampshire in this cycle -- no
choice is available, and the survey's several readings are averaged into one.
That is the honest summary of "how a generic Democrat polls here", and
crucially it is still one observation rather than three.

The same test then applies to single-reading surveys, which is where most of
the effect lands: Maine's 34 records collapse to 3, because every survey before
mid-July tested Graham Platner or Janet Mills and the nominee is Troy Jackson.
That is a real loss of information, and the right one -- the model had been
treating a year of polling about other people as evidence about this race.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import date

from .polls import PLACEHOLDER_NAMES, NormalisedPoll

log = logging.getLogger(__name__)

#: What makes two records the same survey. Not the poll id, which differs per
#: matchup, and not the field date alone, which would merge two pollsters who
#: happened to be in the field together.
#:
#: The sponsor is part of it. One pollster can field for a Democratic client and
#: a Republican client over the same days, and those are two commissioned
#: surveys of two samples, not one survey read twice -- merging them would
#: average away the very partisan lean the model estimates separately. A single
#: survey testing several matchups always carries one sponsor, so including it
#: costs nothing.
SurveyKey = tuple[str, str, date, date, int, int]


def survey_key(poll: NormalisedPoll) -> SurveyKey:
    return (
        poll.race_id,
        poll.pollster,
        poll.start_date,
        poll.end_date,
        poll.sample_size,
        poll.partisan_sign,
    )


#: How many recent surveys decide who is still being polled.
#:
#: Surveys, not days. A fixed number of days is arbitrary relative to how often
#: a race is polled: three weeks covers eleven Michigan surveys and two Maine
#: ones. Counting surveys adapts on its own -- a heavily polled race resolves
#: within days of a primary, a sparse one takes as long as it takes.
#:
#: Three, because one is too fragile -- a single pollster testing an odd matchup
#: would redefine the race and discard everything else -- and because requiring
#: agreement among three independent pollsters is a real bar. Both switches in
#: this cycle are caught cleanly: Maine went from Graham Platner to Troy Jackson
#: between 25 June and 17 July, Michigan from a three-way field to Abdul
#: El-Sayed alone between 29 July and 8 August.
LIVE_SURVEYS = 3


def current_matchup(polls: Sequence[NormalisedPoll]) -> tuple[set[str], set[str]]:
    """Who pollsters are currently testing in a race, per side.

    Returns the candidates still being polled. One name on a side means the
    nomination is settled as far as pollsters are concerned; several means it is
    not, and the caller must not pick between them.

    Inferred from behaviour rather than declared anywhere: once a primary is
    decided, pollsters stop testing the losers. That makes the polls
    self-describing, so a result is picked up within days of happening and no
    roster needs editing to keep the model honest.
    """
    if not polls:
        return set(), set()

    recent_dates = sorted({p.field_date for p in polls}, reverse=True)[:LIVE_SURVEYS]
    recent = [p for p in polls if p.field_date in set(recent_dates)]

    def named(values: Iterable[str]) -> set[str]:
        """Real people only.

        A poll against a generic "Rep" is not evidence about a named matchup, so
        a placeholder must not join the live set -- otherwise Minnesota reads as
        having two live Republicans, Michele Tafoya and the word "Rep". When a
        side has *only* placeholders the set comes back empty, which the caller
        treats as "no opinion", and every reading is kept.
        """
        return {v for v in values if v and v.strip().lower() not in PLACEHOLDER_NAMES}

    return (
        named(p.dem_candidate for p in recent),
        named(p.rep_candidate for p in recent),
    )


def _average(records: Sequence[NormalisedPoll]) -> NormalisedPoll:
    """Collapse several readings of one survey into a single observation.

    Averages the two-party share and keeps everything else from the first
    record. The sample size is deliberately *not* summed: these readings share
    one set of respondents, and summing would reintroduce exactly the false
    precision this module exists to remove.
    """
    first = records[0]
    if len(records) == 1:
        return first
    mean_two_party = sum(p.two_party_dem for p in records) / len(records)
    return replace(
        first,
        two_party_dem=mean_two_party,
        dem_pct=sum(p.dem_pct for p in records) / len(records),
        rep_pct=sum(p.rep_pct for p in records) / len(records),
        other_pct=sum(p.other_pct for p in records) / len(records),
    )


def deduplicate(polls: Iterable[NormalisedPoll]) -> tuple[list[NormalisedPoll], Counter]:
    """Reduce each survey to one observation.

    Returns the surviving polls and a counter describing what happened, so the
    caller can report it rather than silently discarding a quarter of the feed.
    """
    polls = list(polls)
    by_race: dict[str, list[NormalisedPoll]] = defaultdict(list)
    for poll in polls:
        by_race[poll.race_id].append(poll)

    live_matchup = {race: current_matchup(rows) for race, rows in by_race.items()}

    surveys: dict[SurveyKey, list[NormalisedPoll]] = defaultdict(list)
    for poll in polls:
        surveys[survey_key(poll)].append(poll)

    kept: list[NormalisedPoll] = []
    counts: Counter = Counter()

    for key, records in surveys.items():
        counts["surveys"] += 1
        dem_live, rep_live = live_matchup[key[0]]

        # Keep only the readings whose candidates are still being polled, and
        # apply that to every survey rather than only to those with several
        # readings. Otherwise a poll testing one superseded candidate survives
        # while the same information published as two rows is dropped -- same
        # stale matchup, different treatment purely because of how the pollster
        # filed it. Maine is where this bites: 34 records, of which only 3
        # concern the nominee who emerged in July.
        live = [
            p for p in records
            if (not dem_live or p.dem_candidate in dem_live)
            and (not rep_live or p.rep_candidate in rep_live)
        ]

        if len(records) > 1:
            counts["surveys with several matchups"] += 1

        if not live:
            counts["surveys dropped: tests nobody still being polled"] += 1
            counts["records removed"] += len(records)
            continue

        if len(live) < len(records):
            counts["readings dropped: candidate superseded"] += len(records) - len(live)

        if len(live) > 1:
            counts["surveys averaged: primary unresolved"] += 1

        counts["records removed"] += len(live) - 1
        kept.append(_average(live))

    kept.sort(key=lambda p: (p.race_id, p.field_date))
    return kept, counts
