"""Who the dashboard says is on the ballot.

Names are cosmetic to the model and load-bearing for the reader: a race
labelled with someone who lost their primary is worse than a race labelled
"Democratic candidate". These tests pin the rule that we only name a candidate
when there is evidence of who the nominee is.
"""

from __future__ import annotations

import pytest

from midterms import outputs as O
from midterms.config import load_all
from midterms.data.polls import PollTable, build_poll_table
from midterms.data.roster import Roster
from midterms.data.votehub import latest_snapshot, load_snapshot


@pytest.fixture(scope="module")
def context():
    races, cfg = load_all()
    roster = Roster.load()
    table = build_poll_table(load_snapshot(latest_snapshot()), races, cfg, roster)
    return races, roster, table


def test_polled_races_are_named_from_their_polls(context):
    races, roster, table = context
    counts = table.counts_by_race()
    polled = [r for r in races.race_ids if counts.get(r, 0) > 0]
    assert polled, "no polled races -- the snapshot is empty or misparsed"

    for race_id in polled:
        names = O._candidates(table, race_id, roster)
        assert names["dem"] and names["rep"], (
            f"{race_id} has polls but no candidate names"
        )


def test_placeholders_never_reach_the_page(context):
    races, roster, table = context
    generic = {"dem", "rep", "democrat", "republican", "democratic"}
    for race_id in races.race_ids:
        names = O._candidates(table, race_id, roster)
        for side, name in names.items():
            if name is None:
                continue
            assert name.strip().lower() not in generic, (
                f"{race_id} {side} is the placeholder {name!r}, not a person"
            )


def test_a_contested_primary_is_not_resolved_by_guessing(context):
    """The roster lists whole primary fields and is not ordered by winner.

    Taking the first entry would name a real person who may not be running.
    With no polls to settle it, the honest answer is no name at all.
    """
    races, roster, table = context
    contested = [
        race_id
        for race_id in races.race_ids
        if sum(1 for s in roster.by_race.get(race_id, {}).values() if s == "D") > 1
    ]
    assert contested, "fixture assumption broken: no contested D primaries in roster"

    empty = PollTable(polls=[])
    for race_id in contested:
        assert O._candidates(empty, race_id, roster)["dem"] is None, (
            f"{race_id} named a Democrat despite a contested primary and no polls"
        )


def test_unpolled_race_with_one_nominee_a_side_is_named(context):
    """The other half of the rule: unambiguous rosters still get names."""
    races, roster, table = context
    empty = PollTable(polls=[])
    named = 0
    for race_id in races.race_ids:
        sides = list(roster.by_race.get(race_id, {}).values())
        if sides.count("D") == 1 and sides.count("R") == 1:
            names = O._candidates(empty, race_id, roster)
            assert names["dem"] and names["rep"], race_id
            named += 1
    assert named, "no unambiguous rosters -- test proves nothing"


def test_names_keep_their_original_spelling(context):
    """Roster keys are case-folded for matching; the page must not show that."""
    races, roster, table = context
    for race_id in races.race_ids:
        for name in O._candidates(table, race_id, roster).values():
            if name and " " in name:
                assert name != name.lower(), f"{race_id}: {name!r} lost its capitals"
