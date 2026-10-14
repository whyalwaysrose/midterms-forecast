"""Telling a new poll apart from one we had been rejecting all along.

The differ decides a poll is new by its absence from the previous run. That is
not the same thing. A poll can be in the feed for weeks, rejected because the
roster cannot resolve a candidate name, and then start counting the day the
roster is rebuilt -- without any pollster doing anything.

It happened the first time the House roster was rebuilt in CI: 30 polls started
counting at once, one of them fielded six weeks earlier, and the commentary
called every one of them new and reported 229 of 435 races as having moved. A
reader was being told fresh polling had changed the race when what had changed
was our own coverage.

The field date cannot make this distinction, which is the trap worth recording:
the median publication lag in the live feed is 12 days and the first quartile is
5, so a rule of "fielded before the last run" would mislabel most ordinary new
polling rather than the rare case. Only `created_at` -- when the feed first
carried the poll -- separates them.
"""

from __future__ import annotations

from midterms import commentary as C
from tests.test_commentary import make_payload, make_poll, make_race

PREV_RUN = "2026-08-19"
THIS_RUN = "2026-08-20"


def _commentary(
    polls_prev,
    polls_curr,
    prob_prev=0.60,
    prob_curr=0.60,
    race_prob_prev=0.40,
    race_prob_curr=0.55,
):
    """Commentary for one race, over two runs.

    The race's own probability moves by default and the chamber's does not:
    `race_changes` keeps only races the differ judged notable, so a race pinned
    at one probability never appears there however its polls changed, while a
    flat chamber probability is what a coverage correction actually looks like.
    """
    prev = make_payload(
        PREV_RUN, prob_prev, [make_race(prob=race_prob_prev, polls=polls_prev)]
    )
    curr = make_payload(
        THIS_RUN, prob_curr, [make_race(prob=race_prob_curr, polls=polls_curr)]
    )
    return C.generate(curr, prev)


def test_a_poll_the_feed_had_at_the_last_run_is_not_called_new():
    """The defect this file exists for.

    Fielded 1 August, in the feed from 2 August, and counted for the first time
    on 20 August: nothing about the race changed on the 20th, our roster did.
    """
    recovered = make_poll("b", date="2026-08-01", created_at="2026-08-02")
    c = _commentary([make_poll("a")], [make_poll("a"), recovered])

    assert "new poll" not in c.headline
    assert "already published" in c.headline
    assert "had not been counted until now" in c.headline

    change = c.race_changes[0]
    assert change.new_polls == []
    assert [p["id"] for p in change.newly_usable] == ["b"]
    assert "correction to our coverage" in change.describe()


def test_a_poll_published_since_the_last_run_is_new_even_if_fielded_before_it():
    """The ordinary case, and the one a field-date rule would get wrong.

    Fieldwork closed five days before the previous run; the feed only carried it
    afterwards. That is new polling, and the live feed's median lag of 12 days
    means this is the common shape, not an edge case.
    """
    fresh = make_poll("b", date="2026-08-14", created_at="2026-08-20")
    c = _commentary([make_poll("a")], [make_poll("a"), fresh])

    assert "1 new poll" in c.headline
    assert "already published" not in c.headline

    change = c.race_changes[0]
    assert [p["id"] for p in change.new_polls] == ["b"]
    assert change.newly_usable == []
    assert "Driven by 1 new poll" in change.describe()


def test_a_poll_created_on_the_day_of_the_last_run_counts_as_new():
    """`created_at` is a date, so same-day ordering is unknowable.

    The previous run may well have read the feed before the poll landed.
    Resolving the tie to "new" asserts nothing about our own coverage that
    cannot be shown; resolving it the other way would claim we had the poll and
    ignored it.
    """
    same_day = make_poll("b", date="2026-08-15", created_at=PREV_RUN)
    change = _commentary([make_poll("a")], [make_poll("a"), same_day]).race_changes[0]

    assert [p["id"] for p in change.new_polls] == ["b"]
    assert change.newly_usable == []


def test_a_payload_without_created_at_keeps_the_old_behaviour():
    """Archived payloads pre-date the field, and so may another feed.

    Absent evidence the poll was already in the feed, it is reported as new --
    which is both the claim that needs no evidence and what every run before
    this change did.
    """
    change = _commentary(
        [make_poll("a")], [make_poll("a"), make_poll("b", date="2026-06-01")]
    ).race_changes[0]

    assert [p["id"] for p in change.new_polls] == ["b"]
    assert change.newly_usable == []


def test_both_kinds_in_one_race_are_reported_separately():
    fresh = make_poll("b", date="2026-08-19", created_at="2026-08-20")
    recovered = make_poll("c", date="2026-07-04", created_at="2026-07-06")
    c = _commentary([make_poll("a")], [make_poll("a"), fresh, recovered])

    assert "1 new poll" in c.headline
    assert "plus 1 poll that was already published" in c.headline

    text = c.race_changes[0].describe()
    assert "Driven by 1 new poll" in text
    assert "This poll was already published" in text
    # Each poll appears under the right heading, not both under one.
    assert text.index("Driven by") < text.index("already published")


def test_a_run_whose_only_change_is_recovered_polls_is_not_called_quiet():
    """The headline must not say "no new polling" on the day coverage jumps.

    Control barely moves when a few districts start counting, so the quiet-day
    branch is exactly the one that would otherwise fire -- and it would report
    the largest coverage change in the project's history as nothing happening.
    """
    recovered = make_poll("b", date="2026-07-01", created_at="2026-07-03")
    c = _commentary(
        [make_poll("a")], [make_poll("a"), recovered], prob_prev=0.60, prob_curr=0.60
    )

    assert "No new polling" not in c.headline
    assert "already published" in c.headline


def test_the_national_paragraph_counts_this_chamber_s_contests():
    """It said "all 35 contests" on a 435-district page.

    Hardcoded from the Senate, and wrong on every House run the project has
    published.
    """
    races = [make_race(race_id=f"house-2026-X{i:02d}", name=f"X-{i:02d}", prob=0.5)
             for i in range(7)]
    prev = make_payload(PREV_RUN, 0.60, races)
    curr = make_payload(THIS_RUN, 0.62, races)
    for payload, margin in ((prev, 4.0), (curr, 6.0)):
        payload["national"]["generic_ballot"]["dem_margin_median"] = margin

    body = "\n".join(C.generate(curr, prev).body)
    assert "all 7 contests" in body
    assert "35 contests" not in body
