"""Day-over-day commentary: detection, attribution, and the slim archive."""

from __future__ import annotations

import json

import pytest

from midterms import commentary as C
from midterms.outputs import slim_payload


def make_poll(poll_id, margin=2.0, pollster="Test Pollster", date="2026-08-18", partisan=0):
    return {
        "id": poll_id, "pollster": pollster, "date": date,
        "start_date": date, "end_date": date,
        "sample_size": 800, "population": "lv",
        "dem_pct": 50.0, "rep_pct": 48.0, "margin": margin,
        "partisan_sign": partisan, "sponsors": [], "url": "",
        "dem_candidate": "D", "rep_candidate": "R",
    }


def make_payload(run_date, dem_control, races):
    return {
        "schema_version": 2,
        "run_date": run_date,
        "generated_at": f"{run_date}T12:00:00+00:00",
        "chamber_forecast": {
            "dem_control_prob": dem_control,
            "rep_control_prob": 1 - dem_control,
            "dem_seats": {"mean": 51.0, "median": 51, "p05": 45, "p95": 57},
            "seat_distribution": {}, "seats_not_up": {"D": 34, "R": 31},
            "dem_seats_for_majority": 51, "tiebreaker_party": "R",
            "n_simulations": 40000, "exact_tie_prob": 0.05,
        },
        "national": {
            "generic_ballot": {
                "dem_margin_median": 6.0, "dem_margin_p05": 4.0,
                "dem_margin_p95": 8.0, "dem_two_party_share_median": 0.53,
            },
            "trajectory": [],
        },
        "races": races,
        "diagnostics": {"max_r_hat": 1.01, "min_ess_bulk": 700, "divergences": 0, "n_draws": 4000},
        "poll_summary": {
            "n_race_polls": sum(r["poll_count"] for r in races),
            "n_national_polls": 400, "n_pollsters": 50,
            "rejections": {}, "unknown_candidates": {},
        },
    }


def make_race(race_id="senate-2026-GA", name="Georgia", prob=0.6, margin_p50=3.0, polls=None):
    polls = polls if polls is not None else []
    return {
        "id": race_id, "unit": race_id[-2:], "name": name, "special": False,
        "incumbent_party": "D", "incumbent_status": "elected",
        "dem_win_prob": prob, "rep_win_prob": 1 - prob,
        "margin": {"p05": margin_p50 - 10, "p25": margin_p50 - 4, "p50": margin_p50,
                   "p75": margin_p50 + 4, "p95": margin_p50 + 10},
        "fundamentals_prior_margin": 1.0, "tipping_point_prob": 0.1,
        "poll_count": len(polls), "latest_poll_date": "2026-08-18",
        "trajectory": [], "polls": polls,
        "all_poll_ids": [p["id"] for p in polls], "notes": [],
    }


# ------------------------------------------------------------------ first run


def test_first_run_has_no_previous():
    current = make_payload("2026-08-20", 0.58, [make_race(polls=[make_poll("a")])])
    note = C.generate(current, None)
    assert note.previous_run_date is None
    assert "First run" in note.headline
    assert note.race_changes == []


def test_first_run_pluralises_poll_counts():
    current = make_payload("2026-08-20", 0.58, [make_race(polls=[make_poll("a")])])
    note = C.generate(current, None)
    text = "\n".join(note.body)
    assert "(1 poll)" in text
    assert "(1 polls)" not in text


# ------------------------------------------------------------------- diffing


def test_new_poll_is_attributed_to_the_race_that_moved():
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.55, margin_p50=2.0,
                                                        polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.62, [make_race(prob=0.66, margin_p50=5.0,
                                                        polls=[make_poll("old"),
                                                               make_poll("new", margin=8.0,
                                                                         pollster="Fresh Poll")])])
    note = C.generate(curr, prev)

    assert note.new_poll_count == 1
    assert len(note.race_changes) == 1
    change = note.race_changes[0]
    assert change.prob_delta == pytest.approx(0.11)
    assert [p["id"] for p in change.new_polls] == ["new"]
    assert "Fresh Poll" in change.describe()


def test_move_without_new_polls_is_labelled_as_such():
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.55, polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.60, [make_race(prob=0.62, polls=[make_poll("old")])])
    note = C.generate(curr, prev)

    assert note.new_poll_count == 0
    assert "No new polls in this race" in note.race_changes[0].describe()


def test_quiet_day_is_reported_as_quiet():
    race = make_race(prob=0.6, polls=[make_poll("old")])
    prev = make_payload("2026-08-19", 0.600, [race])
    curr = make_payload("2026-08-20", 0.601, [race])
    note = C.generate(curr, prev)
    assert "No new polling" in note.headline


def test_tiny_moves_are_not_reported():
    prev = make_payload("2026-08-19", 0.60, [make_race(prob=0.600, margin_p50=3.00,
                                                        polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.60, [make_race(prob=0.602, margin_p50=3.05,
                                                        polls=[make_poll("old")])])
    note = C.generate(curr, prev)
    assert note.race_changes == []


def test_safe_seat_margin_drift_is_not_reported():
    """A 0.4-point margin wiggle in a race at 0.9% is arithmetic, not news."""
    prev = make_payload("2026-08-19", 0.55, [make_race(
        race_id="senate-2026-KY", name="Kentucky", prob=0.011, margin_p50=-23.0,
        polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.55, [make_race(
        race_id="senate-2026-KY", name="Kentucky", prob=0.009, margin_p50=-22.6,
        polls=[make_poll("old")])])
    note = C.generate(curr, prev)
    assert note.race_changes == []


def test_competitive_race_margin_drift_is_reported():
    """The same shift in a contested race is worth saying."""
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.52, margin_p50=0.5,
                                                        polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.55, [make_race(prob=0.525, margin_p50=0.9,
                                                        polls=[make_poll("old")])])
    note = C.generate(curr, prev)
    assert len(note.race_changes) == 1


def test_large_probability_move_is_reported_even_in_a_safe_seat():
    """The competitiveness guard applies to margin drift only."""
    prev = make_payload("2026-08-19", 0.55, [make_race(
        race_id="senate-2026-KY", name="Kentucky", prob=0.005, margin_p50=-23.0,
        polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.55, [make_race(
        race_id="senate-2026-KY", name="Kentucky", prob=0.10, margin_p50=-12.0,
        polls=[make_poll("old")])])
    note = C.generate(curr, prev)
    assert len(note.race_changes) == 1


def test_national_environment_shift_is_explained():
    prev = make_payload("2026-08-19", 0.55, [make_race(polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.60, [make_race(polls=[make_poll("old")])])
    curr["national"]["generic_ballot"]["dem_margin_median"] = 7.5
    note = C.generate(curr, prev)
    assert any("National environment" in line for line in note.body)


def test_partisan_sponsor_is_flagged_in_the_poll_line():
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.55, polls=[])])
    curr = make_payload("2026-08-20", 0.62, [make_race(prob=0.65, polls=[
        make_poll("new", pollster="Partisan Co", partisan=-1)])])
    note = C.generate(curr, prev)
    assert "R-aligned sponsor" in note.race_changes[0].describe()


def test_backfilled_older_poll_is_not_double_counted():
    """`polls` is capped for display, so diffing must use `all_poll_ids`."""
    displayed = [make_poll(f"p{i}") for i in range(3)]
    prev_race = make_race(polls=displayed)
    # The previous run knew about two extra polls that fell outside the display cap.
    prev_race["all_poll_ids"] = [p["id"] for p in displayed] + ["older1", "older2"]
    prev = make_payload("2026-08-19", 0.55, [prev_race])

    curr_race = make_race(polls=displayed + [make_poll("older1")])
    curr = make_payload("2026-08-20", 0.55, [curr_race])

    note = C.generate(curr, prev)
    assert note.new_poll_count == 0


# ------------------------------------------------------------------ markdown


def test_markdown_renders_with_a_date_heading():
    curr = make_payload("2026-08-20", 0.58, [make_race(polls=[make_poll("a")])])
    md = C.generate(curr, None).to_markdown()
    assert md.startswith("## 2026-08-20")


def test_commentary_is_json_serialisable():
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.55, polls=[make_poll("old")])])
    curr = make_payload("2026-08-20", 0.62, [make_race(prob=0.66, polls=[
        make_poll("old"), make_poll("new")])])
    payload = C.generate(curr, prev).to_dict()
    json.dumps(payload)   # must not raise


# ---------------------------------------------------------------- slim archive


def test_slim_payload_keeps_everything_the_differ_needs():
    polls = [make_poll(f"p{i}") for i in range(4)]
    full = make_payload("2026-08-20", 0.58, [make_race(polls=polls)])
    slim = slim_payload(full)

    # A diff against the slim archive must behave identically to one against
    # the full payload.
    later = make_payload("2026-08-21", 0.60, [make_race(prob=0.66, polls=polls + [
        make_poll("brand-new")])])

    from_full = C.generate(later, full)
    from_slim = C.generate(later, slim)

    assert from_full.new_poll_count == from_slim.new_poll_count == 1
    assert from_full.headline == from_slim.headline


def test_slim_payload_is_substantially_smaller():
    race = make_race(polls=[make_poll(f"p{i}") for i in range(25)])
    race["trajectory"] = [{"date": "2026-01-01", "p05": -5, "p50": 0, "p95": 5}] * 69
    full = make_payload("2026-08-20", 0.58, [race])
    assert len(json.dumps(slim_payload(full))) < 0.35 * len(json.dumps(full))


# ------------------------------------------------- model-change attribution


def test_model_change_is_reported_instead_of_blamed_on_polls():
    """Regression: a model revision was reported as though polls caused it.

    After recalibration every race moved at once. The differ, which assumes a
    fixed model, announced a 5.6-point jump "on 1 new poll" and credited 26
    races to "the national environment" — a claim that was simply false.
    """
    race = make_race(prob=0.42, polls=[make_poll("old")])
    prev = make_payload("2026-08-19", 0.55, [race])
    prev["model_fingerprint"] = "aaaaaaaaaaaa"

    curr = make_payload("2026-08-20", 0.63, [make_race(prob=0.63, polls=[make_poll("old")])])
    curr["model_fingerprint"] = "bbbbbbbbbbbb"

    note = C.generate(curr, prev)
    assert "model changed" in note.headline.lower()
    text = " ".join(note.body) + " ".join(c.describe() for c in note.race_changes)
    assert "model revision" in text
    assert "comes from the national environment" not in text


def test_identical_fingerprints_keep_normal_attribution():
    polls = [make_poll("old")]
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.55, polls=polls)])
    curr = make_payload("2026-08-20", 0.60, [make_race(prob=0.62, polls=polls)])
    for payload in (prev, curr):
        payload["model_fingerprint"] = "same00000000"

    note = C.generate(curr, prev)
    assert "model changed" not in note.headline.lower()
    assert "comes from the national environment" in note.race_changes[0].describe()


def test_missing_fingerprint_does_not_claim_a_model_change():
    """Runs archived before fingerprinting existed must not all look changed."""
    polls = [make_poll("old")]
    prev = make_payload("2026-08-19", 0.55, [make_race(prob=0.55, polls=polls)])
    curr = make_payload("2026-08-20", 0.60, [make_race(prob=0.62, polls=polls)])
    curr["model_fingerprint"] = "bbbbbbbbbbbb"   # previous has none
    note = C.generate(curr, prev)
    assert "model changed" not in note.headline.lower()


def test_fingerprint_changes_when_the_config_changes(tmp_path, monkeypatch):
    from midterms import outputs, paths

    before = outputs.model_fingerprint()
    original = paths.MODEL_CONFIG.read_bytes()
    try:
        paths.MODEL_CONFIG.write_bytes(original + b"\n# nudge\n")
        assert outputs.model_fingerprint() != before
    finally:
        paths.MODEL_CONFIG.write_bytes(original)
    assert outputs.model_fingerprint() == before
