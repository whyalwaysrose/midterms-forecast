"""Who the dashboard says is on the ballot.

Names are cosmetic to the model and load-bearing for the reader: a race
labelled with someone who lost their primary is worse than a race labelled
"Democratic candidate". These tests pin the rule that we only name a candidate
when there is evidence of who the nominee is.

Polls here are synthetic. The live snapshot is not committed, so a test built
on it passes locally and errors in CI -- and the rule under test is about how
_candidates reads a table, not about any particular week's polling.
"""

from __future__ import annotations

from datetime import date

import pytest

from midterms import outputs as O
from midterms.config import load_all
from midterms.data.polls import NormalisedPoll, PollTable
from midterms.data.roster import Roster


@pytest.fixture(scope="module")
def context():
    races, _cfg = load_all()
    return races, Roster.load()


def poll(race_id: str, dem: str, rep: str, day: int = 1, **kwargs) -> NormalisedPoll:
    """A poll carrying nothing but a race, a date, and two names."""
    when = date(2026, 8, day)
    fields = dict(
        poll_id=f"{race_id}-{day}", race_id=race_id, pollster="Test Poll",
        field_date=when, start_date=when, end_date=when,
        sample_size=800, population="lv",
        dem_pct=50.0, rep_pct=48.0, two_party_dem=50.0 / 98.0, other_pct=2.0,
        partisan_sign=0, sponsors=(), url="",
        dem_candidate=dem, rep_candidate=rep,
    )
    fields.update(kwargs)
    return NormalisedPoll(**fields)


def test_a_polled_race_is_named_from_its_polls(context):
    _races, roster = context
    table = PollTable(polls=[poll("senate-2026-GA", "Jon Ossoff", "Mike Collins")])
    assert O._candidates(table, "senate-2026-GA", roster) == {
        "dem": "Jon Ossoff", "rep": "Mike Collins",
    }


def test_the_most_recent_poll_wins(context):
    """A primary result shows up as a name change, with no file to edit."""
    _races, roster = context
    table = PollTable(polls=[
        poll("senate-2026-GA", "Jon Ossoff", "Derek Dooley", day=1),
        poll("senate-2026-GA", "Jon Ossoff", "Mike Collins", day=20),
    ])
    assert O._candidates(table, "senate-2026-GA", roster)["rep"] == "Mike Collins"


@pytest.mark.parametrize("placeholder", ["Dem", "rep", "Democrat", "REPUBLICAN"])
def test_generic_placeholders_are_skipped_for_a_real_name(context, placeholder):
    """VoteHub uses bare party names before a nominee is settled."""
    _races, roster = context
    table = PollTable(polls=[
        poll("senate-2026-GA", "Jon Ossoff", "Mike Collins", day=1),
        poll("senate-2026-GA", placeholder, placeholder, day=20),
    ])
    names = O._candidates(table, "senate-2026-GA", roster)
    assert names["dem"] == "Jon Ossoff"
    assert names["rep"] == "Mike Collins"


def test_a_contested_primary_is_not_resolved_by_guessing(context):
    """The roster lists whole primary fields and is not ordered by winner.

    Taking the first entry would name a real person who may not be running.
    With no polls to settle it, the honest answer is no name at all.
    """
    races, roster = context
    contested = [
        race_id
        for race_id in races.race_ids
        if sum(1 for s in roster.by_race.get(race_id, {}).values() if s == "D") > 1
    ]
    assert contested, "fixture assumption broken: no contested D primaries in roster"

    for race_id in contested:
        assert O._candidates(PollTable(polls=[]), race_id, roster)["dem"] is None, (
            f"{race_id} named a Democrat despite a contested primary and no polls"
        )


def test_an_unpolled_race_with_one_nominee_a_side_is_still_named(context):
    """The other half of the rule: unambiguous rosters do get names."""
    races, roster = context
    named = 0
    for race_id in races.race_ids:
        sides = list(roster.by_race.get(race_id, {}).values())
        if sides.count("D") == 1 and sides.count("R") == 1:
            names = O._candidates(PollTable(polls=[]), race_id, roster)
            assert names["dem"] and names["rep"], race_id
            named += 1
    assert named, "no unambiguous rosters -- this test would prove nothing"


def test_roster_names_keep_their_original_spelling(context):
    """Roster keys are case-folded for matching; the page must not show that."""
    races, roster = context
    checked = 0
    for race_id in races.race_ids:
        for name in O._candidates(PollTable(polls=[]), race_id, roster).values():
            if name and " " in name:
                assert name != name.lower(), f"{race_id}: {name!r} lost its capitals"
                checked += 1
    assert checked, "no roster-derived names to check"


def test_an_unknown_race_names_nobody(context):
    _races, roster = context
    assert O._candidates(PollTable(polls=[]), "senate-2026-ZZ", roster) == {
        "dem": None, "rep": None,
    }
