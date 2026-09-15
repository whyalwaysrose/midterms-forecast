"""The registry of other people's forecasts, and the sensitivity it feeds.

Contrasting this model against rival models is the only cheap way to learn it is
wrong before November says so. Two things have to hold for that contrast to mean
anything, and both fail silently:

* Every rival number must be one somebody can point at. A probability with no
  source URL and no date is a rumour, and a rumour that disagrees with the model
  is worse than no comparison at all.

* The re-simulation behind the Nebraska decomposition must still be measuring
  the model. It stopped being so once, undetectably, and that is the specific
  regression the last test here exists to catch.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "rivals" / "forecasts.csv"
MODEL_DIR = ROOT / "src" / "midterms" / "model"


def rows() -> list[dict]:
    with REGISTRY.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --- the registry -----------------------------------------------------------


def test_the_registry_exists_and_is_populated():
    assert REGISTRY.exists(), "no rival registry to compare against"
    assert rows(), "the registry is empty, so the comparison reports nothing"


def test_every_number_can_be_pointed_at():
    """A rival probability with no source and no date is not evidence.

    The whole value of this file is that a disagreement can be chased back to
    what the other forecaster actually published, on what day. Without that, a
    9-point gap is just a number somebody typed.
    """
    for row in rows():
        who = f"{row['forecaster']}/{row['chamber']}@{row['as_of']}"
        assert row["source_url"].startswith("https://"), f"{who} has no source URL"
        date.fromisoformat(row["as_of"])       # raises if malformed
        date.fromisoformat(row["retrieved"])
        assert date.fromisoformat(row["retrieved"]) >= date.fromisoformat(
            row["as_of"]
        ), f"{who} was recorded before it was published"


def test_probabilities_and_chambers_are_well_formed():
    for row in rows():
        who = f"{row['forecaster']}/{row['chamber']}@{row['as_of']}"
        prob = float(row["dem_control_prob"])
        assert 0.0 < prob < 1.0, f"{who} has an out-of-range probability"
        assert row["chamber"] in {"senate", "house"}, f"{who} has an unknown chamber"
        assert row["counts_osborn"] in {"yes", "no", "unknown", "n/a"}, (
            f"{who} has an unrecognised Osborn treatment; the Senate "
            f"decomposition depends on this field"
        )
        assert row["access"] in {"free", "paywalled-preview", "paywalled"}, (
            f"{who} has an unrecognised access note"
        )


def test_seat_bounds_bracket_the_median_where_given():
    """An interval that does not contain its own median was mistyped."""
    for row in rows():
        if not row["dem_seats_median"]:
            continue
        who = f"{row['forecaster']}/{row['chamber']}@{row['as_of']}"
        lo, mid, hi = (
            int(row["dem_seats_p05"]),
            int(row["dem_seats_median"]),
            int(row["dem_seats_p95"]),
        )
        assert lo <= mid <= hi, f"{who} has a median outside its own interval"


# --- the diagnostic stays a diagnostic --------------------------------------


def test_no_rival_number_can_reach_the_model():
    """Rival forecasts are a contrast, never an input.

    A model that reads other forecasts stops being independent evidence and
    starts being a weighted average of everyone else, which is both worse and
    impossible to score honestly.
    """
    for path in MODEL_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "rivals" not in source and "compare_to_forecasters" not in source, (
            f"{path.relative_to(ROOT)} references the rival registry; nothing "
            f"under model/ may read another forecaster's numbers"
        )


# --- the re-simulation must still measure the model -------------------------


def test_the_sensitivity_resimulation_reproduces_the_published_headline():
    """The regression that shipped, and was invisible for weeks.

    The Nebraska decomposition re-simulates the chamber from each race's
    published median margin plus the config's election-day error. Those
    published margins already contain the election-day error, so rebuilding a
    distribution that way gives it the election-day variance *alone* and throws
    away the posterior spread underneath -- a median 4.5 points of SD per race.

    Too narrow a distribution pushes a probability away from 50%, so the error
    is nearly invisible at a coin flip and grows as the forecast moves away from
    one. It was inside the documented 0.1-point tolerance when written, with the
    Senate near 56%. At 67.6% it was 2.0 points off and reported the
    definitional share of the Polymarket gap as 0.0 points when it was 0.4.

    Nothing failed. The script ran, printed a plausible number, and was wrong.
    """
    pytest.importorskip("numpy")
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from _sensitivity import (  # noqa: PLC0415
        HEADLINE_TOLERANCE,
        reproduces_headline,
        senate_both_ways,
    )

    payload = json.loads(
        (ROOT / "site" / "data" / "forecast.json").read_text(encoding="utf-8")
    )
    both = senate_both_ways(payload)
    assert both is not None, "roster mismatch; the sensitivity cannot be checked"

    ok, drift = reproduces_headline(payload, both[0])
    assert ok, (
        f"the re-simulation lands {drift:.1%} from the published headline, "
        f"outside the {HEADLINE_TOLERANCE:.0%} tolerance -- it is no longer "
        f"measuring the model, so any decomposition built on it is fiction"
    )


def test_the_posterior_spread_is_actually_restored():
    """Guards the fix itself, not just its current numerical outcome.

    The tolerance test above passes today at 67.6%. It would also have passed in
    August at 56% with the bug present, because the error scales with distance
    from a coin flip. So pin the mechanism too.
    """
    source = (ROOT / "scripts" / "_sensitivity.py").read_text(encoding="utf-8")
    assert re.search(r"posterior_sd\s*=\s*np\.sqrt", source), (
        "the re-simulation no longer recovers the posterior spread from the "
        "published interval; it will drift as the forecast moves from 50%"
    )
    assert "senate_without_osborn" not in (
        (ROOT / "scripts" / "compare_to_markets.py").read_text(encoding="utf-8")
    ), "compare_to_markets has its own copy again; the fix will diverge"
