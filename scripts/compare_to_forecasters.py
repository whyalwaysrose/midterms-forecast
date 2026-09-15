"""Where does this model disagree with other people's models?

``compare_to_markets.py`` contrasts this forecast against Polymarket. That is a
market, not a model: it prices what traders will pay, and traders read
forecasts. This script does the other half -- it contrasts the forecast against
other *forecasts*, which is the only cheap way to find out that something here
is wrong before November says so.

This is a diagnostic, never an input. Nothing under ``model/`` imports it, and
no rival number is fed back into anything.

WHY THE NUMBERS ARE HAND-ENTERED
--------------------------------
Every serious 2026 model is either paywalled or has no machine-readable feed.
Measured on 2026-09-15: Silver Bulletin publishes the Senate probability in a
free preview and puts the House behind a subscription, The Economist paywalls
its interactive, 270toWin returns 403 to an automated fetch, and Race to the WH
prints no probability in its page source at all.

So there is no daily scrape to write, and for the paywalled ones there should
not be. ``data/rivals/forecasts.csv`` is a dated, sourced registry instead:
every row carries the date the forecaster published it, the URL it came from,
and the date it was copied down. A number nobody can point at does not go in.

COMPARE LIKE WITH LIKE
----------------------
A rival's number from 3 August against this model's number from today is not a
disagreement, it is five weeks of polling. Because every daily run is archived
under ``outputs/runs/<date>/``, each rival snapshot is compared against *this
model's own number on that date*, and the script refuses the comparison when no
archived run falls within the tolerance rather than quietly using a near one.

That refusal is doing real work, not being fussy. The House archive only begins
on 2026-08-26, and The Economist's public figure is from April -- four months
before ten states redrew 181 seats, which the article itself flags as an open
caveat. Aligning those to the nearest run would manufacture a disagreement out
of a redistricting.

SOME OF THE GAP IS DEFINITIONAL
-------------------------------
This model asks whether the Democratic *caucus* reaches 51, counting Nebraska's
Dan Osborn -- an independent who has said he would not automatically caucus with
either party -- on the Democratic side, because chamber arithmetic forces a
choice. A rival asking whether the Democratic *Party* wins the Senate is
answering a different question.

Where a rival's treatment is not known to match, this reports the model both
ways. The spread between them is the part of any gap that is definitional
rather than substantive, and it has to be subtracted by eye before the
remainder means anything.

Usage:
    python scripts/compare_to_forecasters.py [--tolerance DAYS]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sensitivity import reproduces_headline, senate_both_ways  # noqa: E402

REGISTRY = REPO / "data" / "rivals" / "forecasts.csv"
RUNS = REPO / "outputs" / "runs"
FILENAME = {"senate": "forecast.json", "house": "forecast_house.json"}

#: Beyond this, two forecasts are telling materially different stories and the
#: difference is worth an afternoon. Chosen as roughly the spread between
#: serious models on a chamber nobody has much information about. Not fitted --
#: there is nothing to fit it against until November.
NOTABLE = 0.08

def load_registry() -> list[dict]:
    if not REGISTRY.exists():
        print(f"No registry at {REGISTRY.relative_to(REPO)}")
        return []
    with REGISTRY.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def archived_runs(chamber: str) -> dict[date, Path]:
    """Every archived run for a chamber, keyed by its run date."""
    out: dict[date, Path] = {}
    for directory in sorted(RUNS.glob("*/")):
        payload = directory / FILENAME[chamber]
        if not payload.exists():
            continue
        try:
            out[date.fromisoformat(directory.name)] = payload
        except ValueError:
            continue
    return out


def nearest_run(runs: dict[date, Path], target: date, tolerance: int):
    """The archived run closest to ``target``, or None if all are too far off."""
    if not runs:
        return None, None
    best = min(runs, key=lambda d: abs((d - target).days))
    gap = (best - target).days
    if abs(gap) > tolerance:
        return None, gap
    return best, gap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contrast this forecast against other published models."
    )
    parser.add_argument(
        "--tolerance", type=int, default=3,
        help="how many days apart two forecasts may be and still be compared "
             "(default: 3)",
    )
    args = parser.parse_args()

    rows = load_registry()
    if not rows:
        return 1

    runs = {chamber: archived_runs(chamber) for chamber in FILENAME}
    forecasters = {row["forecaster"] for row in rows}
    print(f"Registry: {len(rows)} rows from {len(forecasters)} forecasters\n")

    compared, skipped, notable = [], [], []

    for row in sorted(rows, key=lambda r: (r["chamber"], r["as_of"])):
        chamber, who = row["chamber"], row["forecaster"]
        as_of = date.fromisoformat(row["as_of"])
        theirs = float(row["dem_control_prob"])

        run_date, gap = nearest_run(runs[chamber], as_of, args.tolerance)
        if run_date is None:
            reason = (
                f"nearest archived {chamber} run is {abs(gap)} days away"
                if gap is not None else f"no archived {chamber} runs"
            )
            skipped.append((who, chamber, row["as_of"], reason))
            continue

        payload = json.loads(runs[chamber][run_date].read_text(encoding="utf-8"))
        ours = payload["chamber_forecast"]["dem_control_prob"]
        delta = ours - theirs
        compared.append((who, chamber, row, run_date, gap, ours, theirs, delta))
        if abs(delta) >= NOTABLE:
            notable.append((who, chamber, delta))

    if compared:
        print(f"{'Forecaster':<17}{'Chamber':<9}{'Their date':<13}{'Our run':<14}"
              f"{'Them':>7}{'Us':>8}{'Diff':>9}")
        print("-" * 77)
        for who, chamber, row, run_date, gap, ours, theirs, delta in compared:
            stamp = str(run_date) if gap == 0 else f"{run_date} ({gap:+d}d)"
            print(f"{who:<17}{chamber:<9}{row['as_of']:<13}{stamp:<14}"
                  f"{theirs:>6.1%}{ours:>8.1%}{delta:>+9.1%}")

    if skipped:
        print("\nNot compared:")
        for who, chamber, as_of, reason in skipped:
            print(f"  {who} / {chamber} @ {as_of} -- {reason}")

    # --- how much of any Senate gap is definitional rather than substantive ---
    senate_rows = [
        item for item in compared
        if item[1] == "senate" and item[2]["counts_osborn"] != "yes"
    ]
    if senate_rows:
        print("\nNebraska sensitivity (how much of a Senate gap is definitional):")
        for who, _, row, run_date, _, _ours, _theirs, delta in senate_rows:
            payload = json.loads(
                runs["senate"][run_date].read_text(encoding="utf-8")
            )
            both = senate_both_ways(payload)
            if both is None:
                print(f"  {who}: archived roster differs from today's; not computed")
                continue

            with_ne, without_ne = both
            ok, drift = reproduces_headline(payload, with_ne)
            if not ok:
                # Refuse rather than mislead. A decomposition built on a
                # re-simulation that has drifted off the model is worse than no
                # decomposition: it looks like a measurement.
                print(f"  {who}: re-simulation is {drift:.1%} off the "
                      f"{run_date} headline; decomposition suppressed")
                continue

            spread = abs(with_ne - without_ne)
            print(f"  {who} treats Osborn: {row['counts_osborn']}")
            print(f"    us counting him {with_ne:.1%} | not counting him "
                  f"{without_ne:.1%} | definitional spread {spread:.1%}")
            print(f"    gap vs {who} is {abs(delta):.1%}; at most "
                  f"{max(0.0, abs(delta) - spread):.1%} of it is substantive")

    print()
    if notable:
        print(f"NOTABLE: {len(notable)} comparison(s) differ by {NOTABLE:.0%} or more:")
        for who, chamber, delta in notable:
            direction = "more" if delta > 0 else "less"
            print(f"  {chamber}: this model is {abs(delta):.1%} {direction} "
                  f"Democratic than {who}")
        print("\nThat is a prompt to look, not a verdict -- two models can differ")
        print("honestly. Check the definitional spread above first, then whether")
        print("the rival's date really is comparable.")
    else:
        print(f"No comparison differs by more than {NOTABLE:.0%}. "
              "Nothing here says the model is off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
