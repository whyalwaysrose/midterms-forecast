"""The scorer, and the result ingestion that feeds it.

This is the one measurement the project only gets to make once, and every part
of it is written months before there is anything to measure. So the tests carry
more than usual: they are the only thing standing between a metric written in
September and a number printed in November.

Two failure modes matter most, and neither announces itself:

* a scoring function that is subtly wrong produces a plausible number, and
  nobody will have an independent one to check it against on the night;
* an ingester that silently mis-parses produces a *confident* wrong result,
  which is worse than no result at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from midterms import paths, score
from midterms.data import results as results_source

ROOT = Path(__file__).resolve().parents[1]


# --- the metrics, against values worked out by hand -------------------------


def test_brier_is_mean_squared_error():
    assert score.brier([1.0], [1.0]) == 0.0
    assert score.brier([0.0], [1.0]) == 1.0
    assert score.brier([0.5], [1.0]) == pytest.approx(0.25)
    # A forecast of p on an outcome that happens at rate p scores p(1-p).
    assert score.brier([0.7, 0.7], [1.0, 0.0]) == pytest.approx(0.29)


def test_log_score_punishes_confidence_and_stays_finite():
    assert score.log_score([0.5], [1.0]) == pytest.approx(math.log(2))
    # A confident miss must cost a large *finite* number, or one wrong call at
    # p=0 makes the whole score infinite and unreportable.
    assert math.isfinite(score.log_score([0.0], [1.0]))
    assert score.log_score([0.0], [1.0]) > 9.0


def test_skill_is_relative_improvement():
    assert score.skill(0.1, 0.25) == pytest.approx(0.6)
    assert score.skill(0.25, 0.25) == pytest.approx(0.0)
    assert score.skill(0.5, 0.25) == pytest.approx(-1.0)   # worse than baseline
    assert math.isnan(score.skill(0.1, 0.0))


def test_a_certain_forecast_lands_in_the_top_reliability_bucket():
    """The top bucket has to be closed, or p=1.0 falls out of the curve.

    Nearly a third of House districts are forecast above 95%, so a bucket rule
    that drops exactly 1.0 would quietly discard the safest seats from the one
    chart that shows whether the model is over-confident.
    """
    rows = score.reliability([1.0, 1.0, 0.99], [1.0, 1.0, 1.0])
    assert len(rows) == 1
    assert rows[0]["n"] == 3
    assert rows[0]["hi"] == 1.0


def test_reliability_reports_counts_so_thin_buckets_are_visible():
    rows = score.reliability([0.1, 0.1, 0.9], [0.0, 0.0, 1.0])
    assert {r["n"] for r in rows} == {2, 1}


def test_coverage_counts_only_races_with_a_known_margin():
    """A called race with no final margin still scores its win probability, but
    it cannot test an interval. Counting it as a miss would make the forecast
    look over-confident purely because the count was slow."""
    races = {"r1": {"margin": {"p05": -10.0, "p25": -5.0, "p75": 5.0, "p95": 10.0}}}
    known = [score.RaceResult("r1", True, 3.0)]
    unknown = [score.RaceResult("r1", True, None)]

    covered = score.interval_coverage(known, races)
    assert {r["level"]: (r["n"], r["covered"]) for r in covered} == {
        0.5: (1, 1.0), 0.9: (1, 1.0)
    }
    assert score.interval_coverage(unknown, races) == []


def test_coverage_detects_a_miss_outside_the_interval():
    races = {"r1": {"margin": {"p05": -10.0, "p25": -5.0, "p75": 5.0, "p95": 10.0}}}
    outside = [score.RaceResult("r1", True, 25.0)]
    rows = {r["level"]: r["covered"] for r in score.interval_coverage(outside, races)}
    assert rows == {0.5: 0.0, 0.9: 0.0}


# --- chamber scoring --------------------------------------------------------


def _chamber_forecast(**overrides):
    base = {
        "chamber": "house",
        "run_date": "2026-11-02",
        "races": [],
        "chamber_forecast": {
            "dem_control_prob": 0.72,
            "dem_seats_for_majority": 218,
            "seats_not_up": {"D": 0, "R": 0},
            "dem_seats": {"median": 236, "p05": 186, "p25": 214,
                          "p75": 259, "p95": 294},
            "seat_distribution": {"230": 0.4, "236": 0.3, "240": 0.3},
        },
    }
    base["chamber_forecast"].update(overrides)
    return base


def test_chamber_scoring_reports_the_probability_given_to_the_actual_winner():
    forecast = _chamber_forecast()
    won = score.score_chamber(forecast, score.ChamberResult("house", 240))
    lost = score.score_chamber(forecast, score.ChamberResult("house", 200))

    assert won["dem_won"] and won["prob_assigned_to_winner"] == pytest.approx(0.72)
    assert not lost["dem_won"]
    assert lost["prob_assigned_to_winner"] == pytest.approx(0.28)


def test_chamber_scoring_flags_a_result_outside_the_interval():
    forecast = _chamber_forecast()
    inside = score.score_chamber(forecast, score.ChamberResult("house", 236))
    outside = score.score_chamber(forecast, score.ChamberResult("house", 300))
    assert inside["inside_90"] and inside["inside_50"]
    assert not outside["inside_90"]
    assert outside["seat_error"] == 64


def test_the_seat_percentile_is_where_the_truth_fell():
    """The most informative number a single election can produce.

    One result cannot show an interval was too wide. A result at the 3rd
    percentile is still a fact about the shape of the forecast.
    """
    forecast = _chamber_forecast()
    assert score.score_chamber(
        forecast, score.ChamberResult("house", 230)
    )["seat_percentile"] == pytest.approx(0.4)
    assert score.score_chamber(
        forecast, score.ChamberResult("house", 240)
    )["seat_percentile"] == pytest.approx(1.0)


# --- rivals -----------------------------------------------------------------


def test_rivals_are_scored_on_their_most_recent_number():
    """A registry that accumulates over months must score the final call.

    Scoring whichever row happened to be first would grade 50+1 on an August
    forecast they had already superseded, which is not a comparison anybody
    should publish.
    """
    rows = [
        {"forecaster": "50+1", "chamber": "house", "as_of": "2026-08-03",
         "dem_control_prob": "0.85"},
        {"forecaster": "50+1", "chamber": "house", "as_of": "2026-09-15",
         "dem_control_prob": "0.97"},
        {"forecaster": "50+1", "chamber": "senate", "as_of": "2026-09-15",
         "dem_control_prob": "0.64"},
    ]
    scored = score.score_rivals(rows, score.ChamberResult("house", 240), dem_won=True)
    assert len(scored) == 1
    assert scored[0]["as_of"] == "2026-09-15"
    assert scored[0]["dem_control_prob"] == pytest.approx(0.97)
    assert scored[0]["brier"] == pytest.approx(0.0009)


def test_the_registry_on_disk_scores_without_error():
    """Guards the real file, not a fixture: a malformed row would only surface
    on election night otherwise."""
    rows = score._load_rivals("house")
    assert rows, "no House rows in the rival registry"
    scored = score.score_rivals(rows, score.ChamberResult("house", 240), dem_won=True)
    assert all(0.0 <= r["brier"] <= 1.0 for r in scored)


# --- the dry run ------------------------------------------------------------


def test_the_dry_run_is_deterministic_and_covers_every_race():
    forecast = json.loads(
        (paths.SITE_DATA_DIR / "forecast_house.json").read_text(encoding="utf-8")
    )
    first, chamber = score.simulate_outcome(forecast, seed=7)
    again, chamber_again = score.simulate_outcome(forecast, seed=7)

    assert len(first) == len(forecast["races"]) == 435
    assert [r.dem_won for r in first] == [r.dem_won for r in again]
    assert chamber.dem_seats == chamber_again.dem_seats
    assert chamber.dem_seats == sum(r.dem_won for r in first)


def test_the_dry_run_scores_end_to_end():
    """The pipeline has to work before there is a result, or it gets debugged
    on the night."""
    forecast = json.loads(
        (paths.SITE_DATA_DIR / "forecast_house.json").read_text(encoding="utf-8")
    )
    races, chamber = score.simulate_outcome(forecast, seed=7)
    result = score.score_races(forecast, races)

    assert result["n"] == 435
    assert 0.0 <= result["brier"] <= 1.0
    # Drawn from the forecast's own distribution, so it must beat a coin by a
    # wide margin. This checks the plumbing, never the model.
    assert result["skill_vs_coin"] > 0.4
    assert {row["level"] for row in result["coverage"]} == {0.5, 0.9}


# --- result ingestion -------------------------------------------------------

#: Modelled on the real 2026 articles: the fields exist and are empty until the
#: night. Anything that treats "empty" as "zero" scores a landslide for the
#: Republicans in September.
NOT_YET = """
{{Infobox election
| majority_seats = 218
| party1 = Republican Party (US)
| seats_before1 = 218
| seats1 =
| party2 = Democratic Party (US)
| seats_before2 = 214
| seats2 =
}}
"""

#: Modelled on 2022's Senate, whose seat field carries a footnote long enough
#: to explain a party switch. This is the case that broke the first parser.
WITH_FOOTNOTE = """
{{Infobox election
| majority_seats = 51
| party1 = Democratic Party (United States)
| seats_after1 = 49{{efn|name=Sinema|Kyrsten Sinema left the Democratic Party in
December 2022, after the election but before the swearing in.<ref></ref>}}
| party2 = Republican Party (United States)
| seats_after2 = 49
| party4 = Independent
| seats_after4 = 2
}}
"""

#: A plausible mis-parse: the numbers are real but they do not add to a chamber.
DOES_NOT_ADD_UP = """
{{Infobox election
| majority_seats = 218
| party1 = Republican Party (US)
| seats1 = 222
| party2 = Democratic Party (US)
| seats2 = 190
}}
"""


def test_an_election_that_has_not_happened_returns_nothing():
    assert results_source.parse_seats(NOT_YET, "house", "test") is None


def test_a_seat_count_survives_a_footnote():
    seats = results_source.parse_seats(WITH_FOOTNOTE, "senate", "test")
    assert seats is not None, "the footnote defeated the parser again"
    assert (seats.dem, seats.rep, seats.ind) == (49, 49, 2)
    assert seats.total == 100
    assert seats.majority_seats == 51
    assert seats.field == "seats_after"


def test_a_parse_that_does_not_add_up_is_refused():
    """The check that catches nearly everything.

    A mis-parse almost never sums to exactly 435, so validating the total is
    what separates "no result yet" from "a confidently wrong result", and the
    second would corrupt the one measurement this project gets.
    """
    assert results_source.parse_seats(DOES_NOT_ADD_UP, "house", "test") is None


def test_both_party_name_spellings_are_recognised():
    """Wikipedia writes "(US)" on the House article and "(United States)" on the
    Senate's, and has switched between them across cycles."""
    for spelling in ("Democratic Party (US)", "Democratic Party (United States)"):
        assert results_source.PARTY_ALIASES[spelling.lower()] == "D"


def test_independents_are_never_folded_into_either_party():
    """The forecast counts the Democratic caucus; Wikipedia counts parties, and
    on the night nobody knows who caucuses with whom. Keeping them separate is
    what stops a silent assumption ending up under the final score."""
    seats = results_source.parse_seats(WITH_FOOTNOTE, "senate", "test")
    assert seats.ind == 2
    assert seats.dem == 49, "independents were folded into the Democratic count"
