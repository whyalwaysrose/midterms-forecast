"""One survey, one observation.

The defect these guard against was invisible in the output: the forecast looked
fine, the poll counts looked healthy, and the model was quietly treating one
sample of 1,163 Michigan voters as five independent polls.
"""

from __future__ import annotations

from datetime import date

import pytest

from midterms.data.polls import NormalisedPoll
from midterms.data.surveys import LIVE_SURVEYS, current_matchup, deduplicate


def poll(
    *,
    race="senate-2026-XX",
    pollster="Test Poll",
    day=1,
    n=800,
    dem="Alice Dem",
    rep="Bob Rep",
    dem_pct=50.0,
    rep_pct=50.0,
) -> NormalisedPoll:
    when = date(2026, 8, day)
    return NormalisedPoll(
        poll_id=f"{race}-{pollster}-{day}-{dem}-{rep}",
        race_id=race, pollster=pollster,
        field_date=when, start_date=when, end_date=when,
        sample_size=n, population="lv",
        dem_pct=dem_pct, rep_pct=rep_pct,
        two_party_dem=dem_pct / (dem_pct + rep_pct),
        other_pct=0.0, partisan_sign=0, sponsors=(), url="",
        dem_candidate=dem, rep_candidate=rep,
    )


# ------------------------------------------------------- what counts as one


def test_one_matchup_is_left_alone():
    polls = [poll(day=1), poll(day=8, pollster="Other")]
    kept, _ = deduplicate(polls)
    assert len(kept) == 2


def test_several_matchups_in_one_survey_become_one_observation():
    """The Michigan case: one sample, several hypothetical opponents."""
    survey = [
        poll(day=1, dem="Alice Dem", dem_pct=43, rep_pct=42),
        poll(day=1, dem="Carol Dem", dem_pct=48, rep_pct=41),
        poll(day=1, dem="Dana Dem", dem_pct=45, rep_pct=42),
    ]
    kept, counts = deduplicate(survey)
    assert len(kept) == 1
    assert counts["records removed"] == 2


def test_a_survey_is_identified_by_pollster_dates_and_size():
    """Two pollsters in the field the same week are two surveys, not one."""
    polls = [poll(day=1, pollster="A"), poll(day=1, pollster="B")]
    assert len(deduplicate(polls)[0]) == 2

    polls = [poll(day=1, n=800), poll(day=1, n=1200)]
    assert len(deduplicate(polls)[0]) == 2


def test_the_merged_sample_size_is_not_summed():
    """Summing would reintroduce exactly the false precision being removed.

    Three readings of 800 people is 800 people asked three questions, and the
    model's sampling variance must reflect 800.
    """
    survey = [poll(day=1, dem=f"D{i}", dem_pct=50, rep_pct=50) for i in range(3)]
    kept, _ = deduplicate(survey)
    assert kept[0].sample_size == 800


# ------------------------------------------------- picking the live matchup


def test_the_nominee_is_read_off_recent_polling():
    """Once a primary is settled, pollsters stop testing the losers."""
    polls = [
        poll(day=1, dem="Loser Dem"), poll(day=1, dem="Winner Dem"),
        poll(day=10, pollster="A", dem="Winner Dem"),
        poll(day=11, pollster="B", dem="Winner Dem"),
        poll(day=12, pollster="C", dem="Winner Dem"),
    ]
    dem_live, _ = current_matchup(polls)
    assert dem_live == {"Winner Dem"}


def test_a_superseded_candidate_is_dropped_even_alone_in_its_survey():
    """The Maine case.

    A poll testing only the candidate who lost the primary measures a race that
    is not happening. Keeping it because it happened to be filed as one row,
    while dropping the same information filed as two, would be arbitrary.
    """
    polls = [
        poll(day=1, pollster="Old", dem="Loser Dem", dem_pct=55, rep_pct=45),
        poll(day=10, pollster="A", dem="Winner Dem"),
        poll(day=11, pollster="B", dem="Winner Dem"),
        poll(day=12, pollster="C", dem="Winner Dem"),
    ]
    kept, counts = deduplicate(polls)
    assert {p.dem_candidate for p in kept} == {"Winner Dem"}
    assert counts["surveys dropped: tests nobody still being polled"] == 1


def test_an_unresolved_primary_averages_rather_than_guessing():
    """No nominee exists yet, so no reading can be preferred over another."""
    survey = [
        poll(day=1, dem="Alice Dem", dem_pct=52, rep_pct=48),
        poll(day=1, dem="Carol Dem", dem_pct=48, rep_pct=52),
    ]
    kept, counts = deduplicate(survey)
    assert len(kept) == 1
    assert kept[0].two_party_dem == pytest.approx(0.5)
    assert counts["surveys averaged: primary unresolved"] == 1


def test_a_changed_opponent_is_handled_on_the_republican_side_too():
    """New Hampshire: one Democrat, two possible Republicans."""
    survey = [
        poll(day=1, rep="Sun Rep", dem_pct=48, rep_pct=41),
        poll(day=1, rep="Brown Rep", dem_pct=49, rep_pct=37),
    ]
    kept, _ = deduplicate(survey)
    assert len(kept) == 1


def test_a_generic_placeholder_is_not_a_live_candidate():
    """Minnesota read as having two live Republicans: a person and the word "Rep"."""
    polls = [
        poll(day=10, pollster="A", rep="Real Rep"),
        poll(day=11, pollster="B", rep="Real Rep"),
        poll(day=12, pollster="C", rep="Rep"),
    ]
    _, rep_live = current_matchup(polls)
    assert rep_live == {"Real Rep"}


def test_a_side_with_only_placeholders_keeps_everything():
    """No opinion is available, so nothing should be discarded on that basis."""
    polls = [
        poll(day=10, pollster="A", rep="Rep", dem="Alice Dem"),
        poll(day=11, pollster="B", rep="Republican", dem="Alice Dem"),
    ]
    _, rep_live = current_matchup(polls)
    assert rep_live == set()
    assert len(deduplicate(polls)[0]) == 2


def test_races_are_resolved_independently():
    """A nominee in one state must not filter another state's polls."""
    polls = [
        poll(race="senate-2026-AA", day=10, dem="Alice Dem"),
        poll(race="senate-2026-BB", day=10, dem="Carol Dem"),
    ]
    kept, _ = deduplicate(polls)
    assert len(kept) == 2


def test_current_matchup_looks_at_several_surveys_not_one():
    """One pollster testing an odd matchup must not redefine the race."""
    polls = [
        poll(day=10, pollster="A", dem="Winner Dem"),
        poll(day=11, pollster="B", dem="Winner Dem"),
        poll(day=12, pollster="Odd", dem="Nobody Dem"),
    ]
    dem_live, _ = current_matchup(polls)
    assert "Winner Dem" in dem_live, (
        f"a single odd survey should not evict the rest; LIVE_SURVEYS={LIVE_SURVEYS}"
    )


def test_empty_input_is_safe():
    assert current_matchup([]) == (set(), set())
    assert deduplicate([]) == ([], {})


# ------------------------------------------------------ against the real feed


def test_the_live_feed_has_no_repeated_surveys_left():
    pytest.importorskip("midterms.data.votehub")
    from midterms.config import load_all
    from midterms.data.polls import build_poll_table
    from midterms.data.roster import Roster
    from midterms.data.surveys import survey_key
    from midterms.data.votehub import latest_snapshot, load_snapshot

    snapshot = latest_snapshot()
    if snapshot is None:
        pytest.skip("no local poll snapshot")

    races, cfg = load_all()
    table = build_poll_table(load_snapshot(snapshot), races, cfg, Roster.load())
    keys = [survey_key(p) for p in table.race_polls]
    assert len(keys) == len(set(keys)), "a survey still appears more than once"
