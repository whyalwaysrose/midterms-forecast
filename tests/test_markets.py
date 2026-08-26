"""Prediction-market odds shown beside the forecast.

Two properties matter more than the parsing. The markets must never influence
the model, and a market outage must never stop a forecast publishing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from midterms.data import markets

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = {
    "markets": [
        {
            "groupItemTitle": "Democrats Sweep",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.485", "0.515"]',
        },
        {
            "groupItemTitle": "Republicans Sweep",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.115", "0.885"],
        },
    ]
}


# --------------------------------------------------------------- parsing


def test_only_the_yes_leg_becomes_a_probability():
    """Each outcome is a binary market; the No leg is its complement."""
    outcomes = markets._outcomes(SAMPLE)
    assert [o.label for o in outcomes] == ["Democrats Sweep", "Republicans Sweep"]
    assert outcomes[0].probability == pytest.approx(0.485)


def test_json_encoded_lists_are_handled():
    """Gamma returns some list fields as strings and some as lists."""
    assert markets._maybe_json('["a", "b"]') == ["a", "b"]
    assert markets._maybe_json(["a", "b"]) == ["a", "b"]
    assert markets._maybe_json("not json") == "not json"


def test_outcomes_are_ordered_most_likely_first():
    outcomes = markets._outcomes(SAMPLE)
    assert outcomes == sorted(outcomes, key=lambda o: -o.probability)


def test_malformed_entries_are_skipped_not_fatal():
    junk = {"markets": [
        {"groupItemTitle": "Fine", "outcomes": ["Yes"], "outcomePrices": ["0.5"]},
        {"groupItemTitle": "", "outcomes": ["Yes"], "outcomePrices": ["0.5"]},
        {"groupItemTitle": "Bad price", "outcomes": ["Yes"], "outcomePrices": ["abc"]},
        {"groupItemTitle": "Out of range", "outcomes": ["Yes"], "outcomePrices": ["7"]},
        {"groupItemTitle": "No lists", "outcomes": None, "outcomePrices": None},
    ]}
    assert [o.label for o in markets._outcomes(junk)] == ["Fine"]


def test_an_event_with_no_markets_yields_nothing():
    assert markets._outcomes({"markets": []}) == []
    assert markets._outcomes({}) == []


# ------------------------------------------------------------- snapshots


def test_a_snapshot_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("midterms.paths.RAW_DIR", tmp_path)
    event = markets.MarketEvent(
        slug="s", title="T", volume=1234.0,
        outcomes=[markets.MarketOutcome("A", 0.6), markets.MarketOutcome("B", 0.4)],
    )
    path = markets.write_snapshot({"s": event})
    loaded, fetched_at = markets.load_snapshot(path)
    assert loaded["s"].title == "T"
    assert loaded["s"].outcomes[0].probability == pytest.approx(0.6)
    assert fetched_at


def test_probability_of_is_case_insensitive():
    event = markets.MarketEvent(
        slug="s", title="T", volume=0.0,
        outcomes=[markets.MarketOutcome("Democratic Party", 0.495)],
    )
    assert event.probability_of("democratic party") == pytest.approx(0.495)
    assert event.probability_of("nobody") is None


# ------------------------------------------- the model must stay unaffected


def test_nothing_in_the_model_imports_markets():
    """Traders read forecasts, so feeding prices back would be circular.

    It would also wreck the calibration the election-day error scales were
    fitted for. This is a one-way street and the test is what keeps it one.
    """
    offenders = []
    for path in (ROOT / "src" / "midterms" / "model").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "markets" in text and "import" in text:
            for line in text.splitlines():
                if "import" in line and "markets" in line:
                    offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, (
        f"the model imports market data: {offenders}. Prediction markets are "
        "for display only."
    )


def test_the_payload_survives_markets_being_unavailable(monkeypatch):
    """A market outage must not delay or fail a forecast."""
    from midterms import outputs

    monkeypatch.setattr(markets, "latest_snapshot", lambda: None)
    assert outputs._markets_block() is None


def test_the_payload_survives_a_corrupt_snapshot(tmp_path, monkeypatch):
    from midterms import outputs

    bad = tmp_path / "markets-2026-01-01.json"
    bad.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(markets, "latest_snapshot", lambda: bad)
    assert outputs._markets_block() is None, "a bad snapshot must not raise"


def test_the_payload_carries_the_no_input_note(tmp_path, monkeypatch):
    """Readers should be told these are not an input, not left to assume."""
    from midterms import outputs

    monkeypatch.setattr("midterms.paths.RAW_DIR", tmp_path)
    event = markets.MarketEvent(
        slug="s", title="T", volume=1.0, outcomes=[markets.MarketOutcome("A", 0.5)]
    )
    path = markets.write_snapshot({"s": event})
    monkeypatch.setattr(markets, "latest_snapshot", lambda: path)

    block = outputs._markets_block()
    assert block is not None
    assert "never an input" in block["note"]
    assert block["source"] == "Polymarket"


def test_vanishingly_small_outcomes_are_dropped_from_the_payload(tmp_path, monkeypatch):
    """A 0.02% leg is noise and would just crowd the card."""
    from midterms import outputs

    monkeypatch.setattr("midterms.paths.RAW_DIR", tmp_path)
    event = markets.MarketEvent(
        slug="s", title="T", volume=1.0,
        outcomes=[markets.MarketOutcome("Real", 0.9), markets.MarketOutcome("Dust", 0.0002)],
    )
    path = markets.write_snapshot({"s": event})
    monkeypatch.setattr(markets, "latest_snapshot", lambda: path)
    labels = [o["label"] for o in outputs._markets_block()["events"][0]["outcomes"]]
    assert labels == ["Real"]


# ------------------------------------------------------------ the page


def test_the_page_says_the_markets_are_not_an_input():
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    section = html[html.index('id="markets"'):]
    section = section[: section.index("</section>")]
    assert "never an input" in section
    assert "circular" in section


def test_the_committed_snapshot_parses_if_present():
    """The live snapshot is committed, so a broken one would ship."""
    snapshot = markets.latest_snapshot()
    if snapshot is None:
        pytest.skip("no committed market snapshot")
    events, fetched_at = markets.load_snapshot(snapshot)
    assert events and fetched_at
    for event in events.values():
        assert event.outcomes
        for outcome in event.outcomes:
            assert 0.0 <= outcome.probability <= 1.0


def test_the_snapshot_is_json_and_small():
    """It is committed on every run, so its size compounds.

    Unthinned it was 206 KB -- thirteen months of daily readings for every leg
    of every market, including a seat-count market with eleven of them. At one
    a day that is roughly 14 MB by election day, for resolution no chart a few
    hundred pixels wide can show. This test is what caught that.
    """
    snapshot = markets.latest_snapshot()
    if snapshot is None:
        pytest.skip("no committed market snapshot")
    json.loads(snapshot.read_text(encoding="utf-8"))
    size = snapshot.stat().st_size
    assert size < 80_000, (
        f"{snapshot.name} is {size / 1024:.0f} KB; history is probably unthinned "
        "or being kept for outcomes that are never charted"
    )


def test_history_is_kept_only_where_it_is_charted():
    """The seat-count market is a distribution; eleven lines would be spaghetti."""
    snapshot = markets.latest_snapshot()
    if snapshot is None:
        pytest.skip("no committed market snapshot")
    events, _ = markets.load_snapshot(snapshot)
    for slug, event in events.items():
        has_history = any(o.history for o in event.outcomes)
        if slug in markets.CHARTED_OVER_TIME:
            assert has_history, f"{slug} is charted over time but has no series"
        else:
            assert not has_history, f"{slug} carries history it will never plot"


def test_thinning_keeps_the_endpoints():
    """The last point is the price the card prints, so it must survive."""
    series = [(f"2026-01-{i:02d}", i / 400) for i in range(1, 400)]
    thinned = markets.thin(series)
    assert len(thinned) <= markets.MAX_HISTORY_POINTS
    assert thinned[0] == series[0]
    assert thinned[-1] == series[-1]
    assert markets.thin(series[:20]) == series[:20], "short series must be untouched"
