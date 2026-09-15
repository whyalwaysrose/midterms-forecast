"""Scoring each day's new polls against the run that had not seen them.

The value of this is that it accumulates without anyone doing anything, and
that is also its risk: a mistake here produces a plausible number every day for
weeks, and the first sign of trouble would be a config change made on the
strength of it.

So the tests pin the things that decide whether a row means anything at all --
whether the poll was genuinely unseen, whether the predictive was built from the
archived run's likelihood rather than today's, and whether the spread includes
everything standing between the latent state and a published number.
"""

from __future__ import annotations

import csv
from datetime import date

import pytest

from midterms import forward
from midterms.model.design import POOLED_POLLSTER


def trajectory(centre_pts: float = 0.0, half_width: float = 4.0) -> list[dict]:
    """A flat latent series, wide enough to be realistic."""
    return [
        {"date": d, "p05": centre_pts - half_width,
         "p50": centre_pts, "p95": centre_pts + half_width}
        for d in ("2026-08-25", "2026-09-01", "2026-09-08", "2026-09-15")
    ]


def payload(seen=("seen-1",), **predictive_overrides) -> dict:
    predictive = {
        "house_effect": {POOLED_POLLSTER: {"mean": 0.0, "sd": 0.02},
                         "Marist": {"mean": 0.03, "sd": 0.01}},
        "population_effect": {"lv": {"mean": 0.0, "sd": 0.0},
                              "rv": {"mean": 0.01, "sd": 0.005}},
        "partisan_effect": {"mean": 0.02, "sd": 0.004},
        "sigma_excess": {"mean": 0.03, "sd": 0.004},
        "reference_population": "lv",
        "design_effect": 1.18,
        "student_t_nu": 5.0,
        "match_student_t_variance": True,
        "national_poll_ids": [],
    }
    predictive.update(predictive_overrides)
    return {
        "run_date": "2026-09-15",
        "predictive": predictive,
        "national": {"latent": trajectory()},
        "races": [{"id": "senate-2026-XX", "all_poll_ids": list(seen),
                   "latent": trajectory(2.0)}],
    }


class Poll:
    """The fields the predictive construction reads.

    Including `dem_pct` and `rep_pct`: the effective sample size scales the
    nominal one by the two-party share of responses, so a poll with 4% going
    elsewhere carries slightly less information about the D-vs-R split than its
    headline n suggests.
    """

    def __init__(self, poll_id, race_id="senate-2026-XX", pollster="Marist",
                 two_party_dem=0.51, sample_size=800, population="lv",
                 partisan_sign=0, field_date=date(2026, 9, 16), other_pct=4.0):
        self.poll_id = poll_id
        self.race_id = race_id
        self.pollster = pollster
        self.two_party_dem = two_party_dem
        self.sample_size = sample_size
        self.population = population
        self.partisan_sign = partisan_sign
        self.field_date = field_date
        self.start_date = field_date
        self.end_date = field_date
        self.other_pct = other_pct
        two_party = 100.0 - other_pct
        self.dem_pct = two_party * two_party_dem
        self.rep_pct = two_party * (1.0 - two_party_dem)


# --- what counts as unseen --------------------------------------------------


def test_a_poll_the_run_already_used_is_not_scored():
    """The whole claim is that these polls are out of sample.

    Scoring a poll the model was fitted on would report the model's fit rather
    than its predictive accuracy, and would look flatteringly well calibrated.
    """
    scored = forward.score_against(payload(seen=("seen-1",)), [Poll("seen-1")])
    assert scored == []


def test_a_poll_the_run_never_saw_is_scored():
    scored = forward.score_against(payload(), [Poll("brand-new")])
    assert len(scored) == 1
    assert 0.0 < scored[0].pit < 1.0


def test_national_polls_are_recognised_as_seen_from_the_predictive_block():
    """Generic-ballot polls have no race to hang `all_poll_ids` off.

    Without `national_poll_ids` every national poll the run *did* use would look
    new, and since they outnumber race polls better than two to one, the record
    would be mostly in-sample while claiming otherwise.
    """
    seen_national = payload()
    seen_national["predictive"]["national_poll_ids"] = ["nat-1"]
    polls = [Poll("nat-1", race_id="__national__"),
             Poll("nat-2", race_id="__national__")]

    scored = forward.score_against(seen_national, polls)
    assert [s.poll_id for s in scored] == ["nat-2"]


def test_a_run_with_no_predictive_block_scores_nothing():
    """Archived runs from before the block existed must not be guessed at."""
    old = payload()
    del old["predictive"]
    assert forward.score_against(old, [Poll("new")]) == []


# --- the predictive itself --------------------------------------------------


def test_the_spread_is_wider_than_the_latent_alone():
    """A predictive built from the latent trajectory would be far too narrow.

    House effect, population effect, sampling noise and excess noise all sit
    between the latent state and a number a pollster publishes. Leaving them out
    would make a well-calibrated model look overconfident and invite exactly the
    wrong correction to `design_effect`.
    """
    scored = forward.score_against(payload(), [Poll("new")])[0]
    latent_sd = (forward.margin_to_logit(6.0) - forward.margin_to_logit(-2.0)) / (2 * forward.Z90)
    assert scored.sd > latent_sd * 1.2


def test_a_bigger_sample_gives_a_tighter_predictive():
    small = forward.score_against(payload(), [Poll("a", sample_size=400)])[0]
    large = forward.score_against(payload(), [Poll("b", sample_size=4000)])[0]
    assert large.sd < small.sd


def test_an_unknown_pollster_falls_back_to_the_pooled_bucket():
    """A pollster the archived run never saw has no house effect of its own and
    must not borrow another firm's. The pooled bucket is what the model uses for
    exactly this case."""
    known = forward.score_against(payload(), [Poll("a", pollster="Marist")])[0]
    unknown = forward.score_against(payload(), [Poll("b", pollster="Nobody Ever")])[0]
    assert known.centre != unknown.centre
    # Marist's +0.03 house effect shifts its expected reading up.
    assert known.centre > unknown.centre


def test_a_partisan_sponsor_shifts_the_centre_towards_its_own_side():
    left = forward.score_against(payload(), [Poll("a", partisan_sign=1)])[0]
    right = forward.score_against(payload(), [Poll("b", partisan_sign=-1)])[0]
    assert left.centre > right.centre


def test_the_archived_likelihood_settings_are_used_not_todays():
    """`design_effect` and `nu` have both been retuned mid-cycle.

    Reading the live config would score an old forecast against a likelihood it
    never used, which is not a calibration measurement of anything.
    """
    tight = forward.score_against(payload(design_effect=1.0), [Poll("a")])[0]
    loose = forward.score_against(payload(design_effect=4.0), [Poll("b")])[0]
    assert loose.sd > tight.sd


# --- placing a poll in time -------------------------------------------------


def test_a_poll_between_grid_points_is_interpolated():
    mid = forward._latent_at(trajectory(), date(2026, 9, 4))
    assert mid is not None and len(mid) == 2


def test_a_poll_beyond_the_archived_tail_clamps_to_the_nearest_point():
    """The archive keeps a short tail, so a poll fielded before it starts has to
    land somewhere sensible rather than being dropped or extrapolated."""
    early = forward._latent_at(trajectory(), date(2026, 1, 1))
    late = forward._latent_at(trajectory(), date(2026, 11, 1))
    assert early is not None and late is not None


def test_a_race_with_no_archived_latent_is_skipped():
    """Approval polls ride in the same table and have no race to predict."""
    assert forward.score_against(payload(), [Poll("x", race_id="__approval__")]) == []


# --- accumulation -----------------------------------------------------------


def test_rows_accumulate_without_double_counting(tmp_path, monkeypatch):
    """The same poll appears in several days' snapshots.

    Counting it once per day would inflate the sample and make a handful of
    polls look like weeks of evidence.
    """
    monkeypatch.setattr(forward, "_record_path", lambda chamber: tmp_path / "f.csv")

    scored = forward.score_against(payload(), [Poll("p1"), Poll("p2")])
    assert forward.append("senate", scored, "2026-09-15", date(2026, 9, 16)) == 2
    assert forward.append("senate", scored, "2026-09-16", date(2026, 9, 17)) == 0

    again = forward.score_against(payload(), [Poll("p3")])
    assert forward.append("senate", again, "2026-09-16", date(2026, 9, 17)) == 1

    with (tmp_path / "f.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["poll_id"] for r in rows] == ["p1", "p2", "p3"]


def test_the_summary_reports_coverage_and_the_row_count(tmp_path, monkeypatch):
    """Coverage without a count invites reading three polls as a finding."""
    monkeypatch.setattr(forward, "_record_path", lambda chamber: tmp_path / "f.csv")
    polls = [Poll(f"p{i}", two_party_dem=0.50 + i / 200) for i in range(12)]
    forward.append("senate", forward.score_against(payload(), polls),
                   "2026-09-15", date(2026, 9, 16))

    stats = forward.accumulated("senate")
    assert stats["n"] == 12
    assert stats["polls"] == 12
    assert {row["level"] for row in stats["coverage"]} == set(forward.CREDIBLE_LEVELS)
    assert 0.0 <= stats["ks"] <= 1.0
    assert 0.0 <= stats["pit_mean"] <= 1.0


def test_an_empty_record_summarises_to_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(forward, "_record_path", lambda chamber: tmp_path / "none.csv")
    assert forward.accumulated("senate") == {"n": 0}


def test_scoring_is_reproducible():
    """Two runs over the same inputs must agree, or the record is not evidence."""
    first = forward.score_against(payload(), [Poll("p")])[0]
    second = forward.score_against(payload(), [Poll("p")])[0]
    assert first.pit == pytest.approx(second.pit)
