"""Where does this model disagree with the people betting real money?

A forecast cannot be scored before the election, but it can be *contrasted*.
Polymarket's 2026 balance-of-power market has turned over ten million dollars
and prices the same two questions this model answers. Where the two agree,
neither is evidence for the other -- traders read forecasts. Where they
disagree, one of them knows something, and it is worth finding out which.

This is a diagnostic, never an input. Nothing under ``model/`` imports it, and
the market numbers are not fed back into anything.

READ THE DISAGREEMENTS CAREFULLY: SOME ARE DEFINITIONAL
------------------------------------------------------
The Senate gap is the clearest example and the reason this script exists rather
than a mental comparison. Polymarket asks whether the **Democratic Party** wins
the Senate. The model asks whether the **Democratic caucus** reaches 51, and
counts Dan Osborn -- an independent who has said he would not automatically
caucus with either party -- on the Democratic side, because chamber arithmetic
forces a choice and that is the one the config makes.

So part of the gap is the two forecasts answering different questions. This
reports the Senate both ways, and the difference between them is the part that
is definitional rather than substantive.

METHOD
------
Chamber probabilities are read out of the balance-of-power event, whose four
outcomes are mutually exclusive and jointly cover the chamber combinations:

    P(D House)  = P(Democrats Sweep) + P(R Senate, D House)
    P(D Senate) = P(Democrats Sweep) + P(D Senate, R House)

Normalised, because a market's outcomes sum to slightly over one -- the spread
between bid and ask on each leg.

The Nebraska sensitivity is computed by re-simulating from the *published*
per-race margins with the config's own error scales, rather than by refitting.
See ``scripts/_sensitivity.py``: the re-simulation has to put back the posterior
spread that the published margins' median leaves out, and it checks itself
against the payload's own headline rather than assuming it worked.

Usage:
    python scripts/compare_to_markets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from midterms import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sensitivity import senate_both_ways  # noqa: E402

BALANCE_SLUG = "balance-of-power-2026-midterms"

#: Which market outcomes imply a Democratic win in each chamber.
IMPLIES_DEM = {
    "house": ("Democrats Sweep", "R Senate, D House"),
    "senate": ("Democrats Sweep", "D Senate, R House"),
}

DRAWS = 200_000


def market_probabilities(payload: dict) -> tuple[dict[str, float], float] | None:
    """Chamber probabilities implied by the balance-of-power market."""
    markets = payload.get("markets") or {}
    for event in markets.get("events", []):
        if event.get("slug") != BALANCE_SLUG:
            continue
        prices = {o["label"]: float(o["probability"]) for o in event.get("outcomes", [])}
        total = sum(prices.values())
        if total <= 0:
            return None
        # Normalise away the spread before combining legs, or the sum of two
        # legs inherits it twice.
        prices = {k: v / total for k, v in prices.items()}
        return (
            {
                chamber: sum(prices.get(label, 0.0) for label in labels)
                for chamber, labels in IMPLIES_DEM.items()
            },
            float(event.get("volume") or 0.0),
        )
    return None


def main() -> int:
    site = paths.SITE_DATA_DIR
    senate_path, house_path = site / "forecast.json", site / "forecast_house.json"
    if not senate_path.exists():
        print("No forecast yet; run `midterms run` first.")
        return 2

    senate = json.loads(senate_path.read_text(encoding="utf-8"))
    house = (
        json.loads(house_path.read_text(encoding="utf-8"))
        if house_path.exists()
        else None
    )

    implied = market_probabilities(senate)
    if implied is None:
        print("No balance-of-power market in the payload; run `midterms fetch-markets`.")
        return 2
    market, volume = implied

    print()
    print("=" * 78)
    print("  MODEL vs PREDICTION MARKET — where the two disagree, and why")
    print("=" * 78)
    print(f"  Polymarket balance-of-power event, ${volume:,.0f} traded")
    print(f"  Market snapshot: {senate['markets'].get('fetched_at', 'unknown')}")
    print(f"  Model run:       {senate['run_date']}")
    print()
    print(f"  {'':10s} {'model':>8s} {'market':>8s} {'gap':>8s}")

    model_probs = {"senate": senate["chamber_forecast"]["dem_control_prob"]}
    if house is not None:
        model_probs["house"] = house["chamber_forecast"]["dem_control_prob"]

    for chamber in ("senate", "house"):
        if chamber not in model_probs:
            print(f"  {chamber:10s} {'--':>8s} {100 * market[chamber]:7.1f}%"
                  f"  (no forecast for this chamber yet)")
            continue
        gap = model_probs[chamber] - market[chamber]
        print(f"  {chamber:10s} {100 * model_probs[chamber]:7.1f}% "
              f"{100 * market[chamber]:7.1f}% {100 * gap:+7.1f}")

    # --- how much of the Senate gap is a difference of definition? ---------
    with_osborn, without = senate_both_ways(senate)
    published = senate["chamber_forecast"]["dem_control_prob"]
    drift = abs(with_osborn - published)

    print()
    print("  SENATE: how much of that gap is Nebraska rather than disagreement?")
    print(f"    re-simulated, Osborn counted as Democratic : {100 * with_osborn:5.1f}%"
          f"   (published {100 * published:.1f}%, drift {100 * drift:.1f})")
    print(f"    re-simulated, Osborn not counted           : {100 * without:5.1f}%")
    print(f"    definitional share of the gap              : "
          f"{100 * (with_osborn - without):5.1f} points")
    remaining = without - market["senate"]
    print(f"    genuine disagreement, after that           : "
          f"{100 * remaining:+5.1f} points")
    if drift > 0.02:
        print("    WARNING: the re-simulation no longer reproduces the published "
              "number; treat the split above as unreliable.")

    print()
    print("  Neither side is evidence for the other where they agree -- traders read")
    print("  forecasts. Where they disagree by more than a few points, one of them")
    print("  knows something. The model's own documented blind spots are candidate")
    print("  quality, scandal and recruitment, none of which it can see and all of")
    print("  which a market prices.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
