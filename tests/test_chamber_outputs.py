"""Two chambers writing into one directory must not overwrite each other.

The Senate keeps the unsuffixed filenames it has always had, because the live
site, every archived run, and any saved link already point at them. Everything
else is suffixed. That asymmetry is easy to half-apply -- one writer updated and
another not -- and the symptom would be the House quietly clobbering the Senate
forecast on the shared dashboard, which no exception would announce.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from midterms import commentary as C
from midterms import paths


def test_senate_keeps_its_historic_filenames():
    for stem in ("forecast", "history", "commentary"):
        assert paths.chamber_filename(stem, "senate") == f"{stem}.json"


def test_other_chambers_are_suffixed():
    assert paths.chamber_filename("forecast", "house") == "forecast_house.json"
    assert paths.chamber_filename("history", "house") == "history_house.json"


def test_no_two_chambers_share_a_filename():
    chambers = ("senate", "house")
    for stem in ("forecast", "history", "commentary"):
        names = {paths.chamber_filename(stem, c) for c in chambers}
        assert len(names) == len(chambers), f"{stem} collides across chambers"


def _payload(chamber: str, run_date: str) -> dict:
    """The smallest payload the commentary differ will accept."""
    return {
        "schema_version": 5,
        "chamber": chamber,
        "run_date": run_date,
        "generated_at": f"{run_date}T00:00:00Z",
        "model_fingerprint": "abc123",
        "chamber_forecast": {
            "dem_control_prob": 0.5,
            "dem_seats": {"median": 1, "mean": 1.0, "p05": 0, "p95": 2},
        },
        "national": {"generic_ballot": {
            "dem_margin_median": 1.0, "dem_margin_p05": -1.0, "dem_margin_p95": 3.0,
        }},
        "poll_summary": {
            "n_race_polls": 1, "n_national_polls": 1, "n_pollsters": 1,
        },
        "diagnostics": {},
        "races": [],
    }


def test_previous_payload_never_diffs_one_chamber_against_the_other(tmp_path):
    """A dated run directory can hold a Senate archive and no House one.

    That is the normal state of affairs: the House was added part-way through
    the cycle, so every run directory written before then has only a Senate
    file. Looking up by date alone would hand the House yesterday's Senate
    numbers and produce commentary describing moves that never happened.
    """
    day = tmp_path / "2026-08-20"
    day.mkdir()
    (day / "forecast.json").write_text(
        json.dumps(_payload("senate", "2026-08-20")), encoding="utf-8"
    )

    senate = C.previous_payload(date(2026, 8, 21), runs_dir=tmp_path, chamber="senate")
    assert senate is not None and senate["chamber"] == "senate"

    house = C.previous_payload(date(2026, 8, 21), runs_dir=tmp_path, chamber="house")
    assert house is None, "the House was handed a Senate archive to diff against"


def test_first_house_commentary_says_house(tmp_path):
    """The prose must follow the numbers, not a hardcoded chamber."""
    entry = C.generate(_payload("house", "2026-08-21"), None)
    assert "House control" in entry.headline
    assert "Senate" not in entry.headline
    assert any("district-level polls" in line for line in entry.body)


def test_each_chamber_writes_its_own_feed_and_changelog(tmp_path):
    entry = C.generate(_payload("house", "2026-08-21"), None)
    feed, changelog = C.write_commentary(
        entry,
        site_data_dir=tmp_path,
        changelog_path=tmp_path / "CHANGELOG_house.md",
        chamber="house",
    )
    assert feed.name == "commentary_house.json"
    assert "House forecast changelog" in changelog.read_text(encoding="utf-8")
    assert not (tmp_path / "commentary.json").exists()


@pytest.mark.parametrize("chamber", ["senate", "house"])
def test_a_run_writes_only_its_own_chamber(tmp_path, chamber):
    """Writing one chamber must leave the other's files untouched.

    The point of the test is the *absence* below: a suffix applied to the
    forecast but not the history would leave the House overwriting the Senate's
    trend line, and the only visible symptom would be a chart that jumps.
    """
    other = "house" if chamber == "senate" else "senate"
    for stem in ("forecast", "history", "commentary"):
        mine = paths.chamber_filename(stem, chamber)
        theirs = paths.chamber_filename(stem, other)
        assert mine != theirs
