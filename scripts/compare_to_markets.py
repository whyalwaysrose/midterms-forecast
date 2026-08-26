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
That reproduces the published headline to within 0.1 points, which is the check
that it is measuring the model and not an approximation of it.

Usage:
    python scripts/compare_to_markets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from midterms import fundamentals as F  # noqa: E402
from midterms import paths  # noqa: E402
from midterms.config import load_all  # noqa: E402
from midterms.model.correlation import cholesky_factor, correlation_matrix  # noqa: E402

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


def senate_without_osborn() -> tuple[float, float]:
    """P(D Senate) counting Nebraska's independent, and not counting him.

    Re-simulated from the published margins rather than refitted: the election-
    day error is applied at simulation time, so everything needed is in the
    payload plus the config. The first number is checked against the published
    headline by the caller.
    """
    races, cfg = load_all(chamber="senate")
    fundamentals = F.compute(races, cfg)
    chol = cholesky_factor(
        correlation_matrix(fundamentals, cfg.election_day_error.correlation)
    )

    payload = json.loads(
        (paths.SITE_DATA_DIR / "forecast.json").read_text(encoding="utf-8")
    )
    by_id = {r["id"]: r for r in payload["races"]}
    base = np.array([by_id[r.id]["margin"]["p50"] for r in races.races]) / 50.0

    ede = cfg.election_day_error
    rng = np.random.default_rng(9)
    national = rng.normal(0.0, ede.national_sd, (DRAWS, 1))
    z = rng.standard_normal((DRAWS, len(base)))
    wins = (base + national + ede.state_sd * (z @ chol.T)) > 0

    control = payload["chamber_forecast"]
    held = control["seats_not_up"]["D"]
    need = control["dem_seats_for_majority"]

    keep = [i for i, r in enumerate(races.races) if r.unit != "NE"]
    with_osborn = float(((held + wins.sum(axis=1)) >= need).mean())
    without = float(((held + wins[:, keep].sum(axis=1)) >= need).mean())
    return with_osborn, without


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
    with_osborn, without = senate_without_osborn()
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
