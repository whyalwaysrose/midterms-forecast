"""Which forecast gets graded.

METHODOLOGY 8c pre-registered in September exactly how the 2026 forecast would
be scored. It did not say *which* forecast, and the scorer read the live
payload -- the one the daily workflow had refit minutes earlier, in the same
run, before scoring.

So on the morning after the election it would have refit the model, then graded
that brand-new forecast against the result it already knew. And again the next
day, on a different forecast, and the next. A pre-registered metric applied to
a moving target is not a measurement.

The rule now is the last run published on or before election day. It needs no
action on the night and freezes itself: before the election the answer is the
latest run, and from election day onward nothing new can qualify.
"""

from __future__ import annotations

import json

import pytest

from midterms import paths as midterms_paths
from midterms import score

ELECTION = "2026-11-03"


def archive(tmp_path, monkeypatch, runs: dict[str, str]):
    """Build a fake outputs/runs tree. Maps run date -> election_date."""
    root = tmp_path / "outputs"
    for run_date, election in runs.items():
        directory = root / "runs" / run_date
        directory.mkdir(parents=True)
        payload = {
            "run_date": run_date,
            "chamber": "senate",
            "races": [],
            "chamber_forecast": {
                "dem_control_prob": 0.6, "dem_seats_for_majority": 51,
                "seats_not_up": {"D": 34, "R": 31},
                "dem_seats": {"median": 52, "p05": 45, "p25": 49,
                              "p75": 54, "p95": 57},
                "seat_distribution": {"52": 1.0},
            },
        }
        if election is not None:
            payload["election_date"] = election
        (directory / "forecast.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(midterms_paths, "OUTPUTS_DIR", root)
    return root


def chosen(tmp_path, monkeypatch, runs) -> str | None:
    archive(tmp_path, monkeypatch, runs)
    found = score.forecast_of_record("senate")
    return None if found is None else found[0]["run_date"]


# --- before the election ----------------------------------------------------


def test_before_the_election_it_is_simply_the_latest_run(tmp_path, monkeypatch):
    runs = {d: ELECTION for d in ("2026-10-11", "2026-10-12", "2026-10-13")}
    assert chosen(tmp_path, monkeypatch, runs) == "2026-10-13"


def test_the_run_on_election_day_itself_counts(tmp_path, monkeypatch):
    """The daily job fires in the morning, hours before the polls close, so
    election day's own run is a legitimate forecast of the result."""
    runs = {d: ELECTION for d in ("2026-11-01", "2026-11-02", ELECTION)}
    assert chosen(tmp_path, monkeypatch, runs) == ELECTION


# --- after the election, which is the whole point ---------------------------


def test_a_run_made_after_the_result_is_never_graded(tmp_path, monkeypatch):
    """The defect this closes.

    Grading the 4 November run would grade a forecast fitted with the answer
    already public, and would grade a different one every day after.
    """
    runs = {d: ELECTION for d in
            ("2026-11-02", ELECTION, "2026-11-04", "2026-11-05", "2026-11-20")}
    assert chosen(tmp_path, monkeypatch, runs) == ELECTION


def test_the_answer_stops_moving_once_the_election_has_passed(tmp_path, monkeypatch):
    """Scoring a week later must give the same verdict as scoring on the night.

    This is what makes the number quotable: if it drifted, every retelling of
    "the model scored X" would be true on a different day and false on the
    rest.
    """
    base = {"2026-11-01": ELECTION, "2026-11-02": ELECTION, ELECTION: ELECTION}
    night = chosen(tmp_path / "a", monkeypatch, base)
    later = chosen(tmp_path / "b", monkeypatch,
                   {**base, "2026-11-04": ELECTION, "2026-11-11": ELECTION})
    assert night == later == ELECTION


def test_it_still_works_when_only_later_runs_remain(tmp_path, monkeypatch):
    """If nothing was archived before the election there is nothing honest to
    grade, and saying so beats grading a hindsight forecast."""
    runs = {"2026-11-05": ELECTION, "2026-11-06": ELECTION}
    assert chosen(tmp_path, monkeypatch, runs) is None


# --- degradation ------------------------------------------------------------


def test_archives_without_an_election_date_fall_back_and_say_so(
    tmp_path, monkeypatch, caplog
):
    """Runs archived before the field existed cannot be placed against the
    election. Using the newest is the only option, and it must be logged rather
    than assumed, because after the election that choice would be wrong."""
    runs = {"2026-10-12": None, "2026-10-13": None}
    with caplog.at_level("WARNING"):
        assert chosen(tmp_path, monkeypatch, runs) == "2026-10-13"
    assert any("election_date" in r.message for r in caplog.records)


def test_no_archive_at_all_returns_nothing(tmp_path, monkeypatch):
    assert chosen(tmp_path, monkeypatch, {}) is None


def test_a_non_date_directory_is_ignored(tmp_path, monkeypatch):
    root = archive(tmp_path, monkeypatch, {"2026-10-13": ELECTION})
    (root / "runs" / "scratch").mkdir()
    (root / "runs" / "scratch" / "forecast.json").write_text("{}", encoding="utf-8")
    assert score.forecast_of_record("senate")[0]["run_date"] == "2026-10-13"


# --- the archive must carry what scoring needs ------------------------------


def test_the_archived_run_carries_the_fundamentals_baseline():
    """One of the two pre-registered baselines lives in this field.

    Without it score_races falls back to 0.5 for every race and the "did it
    beat the fundamentals" comparison silently becomes a comparison with a coin
    -- a number that still prints, and means nothing.
    """
    from midterms.outputs import slim_payload

    full = {
        "schema_version": 5, "model_fingerprint": "x", "run_date": "2026-10-13",
        "generated_at": "2026-10-13T00:00:00", "election_date": ELECTION,
        "chamber_forecast": {}, "poll_summary": {}, "diagnostics": {},
        "national": {"generic_ballot": {}, "trajectory": []},
        "predictive": {},
        "races": [{
            "id": "senate-2026-XX", "name": "XX", "dem_win_prob": 0.7,
            "margin": {"p50": 2.0}, "poll_count": 3, "all_poll_ids": ["a"],
            "trajectory": [], "fundamentals_prior_margin": 1.5,
        }],
    }
    slim = slim_payload(full)
    assert slim["election_date"] == ELECTION
    assert slim["races"][0]["fundamentals_prior_margin"] == 1.5


def test_scoring_reads_the_archive_rather_than_the_live_payload():
    """Pins the fix itself. The live payload is rewritten by every run; the
    archive is not."""
    import inspect

    source = inspect.getsource(score.run_score)
    assert "forecast_of_record" in source
    assert "SITE_DATA_DIR" not in source, (
        "run_score reads the live payload again -- on the day after the "
        "election that is a forecast made after the result was known"
    )


@pytest.mark.parametrize("chamber", ["senate", "house"])
def test_both_chambers_have_an_archive_name(chamber):
    assert score.ARCHIVE_NAME[chamber].endswith(".json")
