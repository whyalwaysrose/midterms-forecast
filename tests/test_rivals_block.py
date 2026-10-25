"""The comparison strip under the headline: which rival numbers it shows.

A comparison that only appears when it flatters is not one, so the rules about
what gets shown have to be fixed in advance rather than adjusted when the
ordering is unflattering. On the House this model is currently last of six.

Two judgements are worth pinning. One reading per forecaster, because the
registry keeps every reading ever taken and six months of 50+1 side by side is
a history rather than a comparison. And nothing older than thirty days at all:
The Economist's April figure was computed on district lines that ten states
have since redrawn, so it is not a view about this election and a caveat would
not make it one.
"""

from __future__ import annotations

from datetime import date

import pytest

from midterms.data import rivals

TODAY = date(2026, 10, 25)

HEADER = (
    "as_of,forecaster,chamber,dem_control_prob,dem_seats_median,dem_seats_p05,"
    "dem_seats_p95,counts_osborn,access,source_url,retrieved,note\n"
)


def _registry(tmp_path, rows):
    path = tmp_path / "forecasts.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def _row(as_of, who, chamber, prob, seats="", access="free", note=""):
    return (
        f'{as_of},{who},{chamber},{prob},{seats},,,n/a,{access},'
        f'https://example.test,{as_of},"{note}"\n'
    )


def test_one_reading_per_forecaster_the_newest(tmp_path):
    path = _registry(tmp_path, [
        _row("2026-10-01", "50+1", "senate", 0.64),
        _row("2026-10-20", "50+1", "senate", 0.73),
        _row("2026-10-10", "50+1", "senate", 0.70),
    ])
    rows = rivals.latest_for_chamber("senate", today=TODAY, path=path)

    assert [(r.forecaster, r.dem_control_prob) for r in rows] == [("50+1", 0.73)]


def test_a_stale_reading_is_dropped_not_captioned(tmp_path):
    """The Economist's April number, on superseded district lines."""
    path = _registry(tmp_path, [
        _row("2026-10-20", "DDHQ", "house", 0.80),
        _row("2026-04-29", "The Economist", "house", 0.979),
    ])
    rows = rivals.latest_for_chamber("house", today=TODAY, path=path)

    assert [r.forecaster for r in rows] == ["DDHQ"]


def test_the_cutoff_is_inclusive_at_the_boundary(tmp_path):
    path = _registry(tmp_path, [
        _row("2026-09-25", "Exactly30", "house", 0.5),   # 30 days
        _row("2026-09-24", "Exactly31", "house", 0.5),   # 31 days
    ])
    rows = rivals.latest_for_chamber("house", today=TODAY, path=path)

    assert [r.forecaster for r in rows] == ["Exactly30"]


def test_the_other_chamber_is_not_mixed_in(tmp_path):
    path = _registry(tmp_path, [
        _row("2026-10-20", "DDHQ", "house", 0.80),
        _row("2026-10-20", "DDHQ", "senate", 0.56),
    ])
    assert [r.dem_control_prob for r in
            rivals.latest_for_chamber("senate", today=TODAY, path=path)] == [0.56]


def test_rows_come_back_most_confident_first(tmp_path):
    path = _registry(tmp_path, [
        _row("2026-10-20", "Low", "house", 0.60),
        _row("2026-10-20", "High", "house", 0.99),
        _row("2026-10-20", "Mid", "house", 0.80),
    ])
    rows = rivals.latest_for_chamber("house", today=TODAY, path=path)

    assert [r.forecaster for r in rows] == ["High", "Mid", "Low"]


@pytest.mark.parametrize("bad", [
    _row("2026-10-20", "NoProb", "house", ""),
    _row("2026-10-20", "Bad", "house", "not-a-number"),
    _row("not-a-date", "BadDate", "house", "0.8"),
    _row("2026-10-20", "TooBig", "house", "1.4"),
])
def test_an_unreadable_row_is_skipped_not_fatal(tmp_path, bad):
    """A hand-maintained CSV will eventually get a typo.

    One malformed row must not take the whole comparison off the dashboard,
    and must certainly not fail a forecast.
    """
    path = _registry(tmp_path, [bad, _row("2026-10-20", "Good", "house", 0.8)])
    rows = rivals.latest_for_chamber("house", today=TODAY, path=path)

    assert [r.forecaster for r in rows] == ["Good"]


def test_a_missing_seat_median_is_allowed(tmp_path):
    """Silver Bulletin publishes a probability and paywalls the seats."""
    path = _registry(tmp_path, [_row("2026-10-20", "Silver Bulletin", "house", 0.94)])
    assert rivals.latest_for_chamber("house", today=TODAY, path=path)[0] \
        .dem_seats_median is None


def test_a_missing_registry_is_an_empty_list(tmp_path):
    assert rivals.load(tmp_path / "absent.csv") == []


# --- the payload block ------------------------------------------------------

def test_the_block_is_withheld_when_there_is_nothing_to_compare(monkeypatch):
    """One other forecaster is an anecdote, not a comparison."""
    from midterms import outputs

    monkeypatch.setattr(
        rivals, "latest_for_chamber", lambda *a, **k: [object()]
    )
    assert outputs._rivals_block("house", TODAY) is None


def test_a_broken_registry_never_breaks_a_forecast(monkeypatch):
    from midterms import outputs

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(rivals, "latest_for_chamber", boom)
    assert outputs._rivals_block("house", TODAY) is None


def test_the_committed_registry_gives_at_least_three_per_chamber():
    """The dashboard promises a comparison, so there has to be one.

    If this fails the registry needs refreshing, not the threshold lowering:
    rivals go stale at thirty days and the whole point of the strip is that it
    is current.
    """
    for chamber in ("senate", "house"):
        rows = rivals.latest_for_chamber(chamber, today=TODAY)
        assert len(rows) >= 3, (
            f"only {len(rows)} current forecasts for the {chamber}; "
            "refresh data/rivals/forecasts.csv"
        )


def test_the_block_carries_what_the_page_needs():
    from midterms import outputs

    block = outputs._rivals_block("senate", TODAY)
    assert block["max_age_days"] == rivals.MAX_AGE_DAYS
    for row in block["forecasters"]:
        assert set(row) >= {
            "name", "dem_control_prob", "dem_seats_median", "as_of",
            "age_days", "access", "source_url",
        }
        assert 0.0 <= row["dem_control_prob"] <= 1.0
        assert row["age_days"] >= 0
