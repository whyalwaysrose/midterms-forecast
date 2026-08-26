"""Is the fundamentals prior too kind to underdogs in lopsided states?

CONCLUSION (2026-08-26): **no.** The model is well calibrated here, and if
anything conservative. The suspicion was wrong and the numbers stand.

The question came from the model giving Democrats 25% in an unpolled Montana
and 34% in an unpolled South Carolina, which looked generous. Measured against
253 historical Senate generals it is not, once the comparison is made properly.

What made it look wrong was comparing against an *unconditional* base rate.
Democrats win about 17.6% of Senate races in states leaning R+15 to R+25 --
averaged over every year. But that average hides almost everything, because the
rate depends enormously on the national environment:

    lean band          weak/neutral year      strong D year (D+6 or better)
    R+25 or worse           0.0% (n=24)              14.3% (n=14)
    R+15 to R+25            5.6% (n=18)              31.2% (n=16)
    R+8 to R+15             7.7% (n=13)              58.3% (n=12)

2026 is running at D+6.7 on the generic ballot, so the right column is the
comparison. South Carolina at 33.9% against 31.2% is nearly exact; Montana at
25.2%, Louisiana at 19.6% and Kentucky at 6.0% are all below their band's
strong-year rate. The model is not being generous, it is reading a strong
Democratic year -- which is what the correlated national environment is *for*.

The trap is worth naming, because acting on the unconditional number would have
"corrected" a correctly calibrated model and made it wrong.

Two smaller findings that did survive:

**`lean_shrinkage` is right.** History gives a slope of 0.956 (SE 0.055) against
the configured 0.95, drifting to 1.005 for 2016 onward as ticket-splitting
collapsed. No change warranted; a case for 1.0 in a later cycle.

**Thin-data pooling moves unpolled red states by 2-4 points.**
`scripts/ablate_pooling.py` refits with races removed. Dropping the six
Republican-leaning states carrying one or two polls each moves unpolled Montana
2.6 points and Oklahoma 4.0, while Colorado moves 0.1 -- so it is specifically
red-state pooling, not a general effect. Nebraska alone is worth about 1 point,
which is Dan Osborn's personal support as an independent leaking into states he
is not running in. Real, modest, and documented rather than fixed.

An attempt to fit the incumbency bonus alongside the lean is NOT reported here
and should not be revived in the same form: FEC `weball` marks a sitting senator
as an incumbent filer in cycles when they are not on the ballot, so the derived
variable was "which party holds a seat in this state" and produced 2 open seats
out of 253. Any incumbency term needs a real seat-level source.

Usage:
    python scripts/fit_fundamentals.py
"""

from __future__ import annotations

import csv
import gzip
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW_POLLS = REPO / "data" / "history" / "raw_polls.csv.gz"

#: Presidential cycles usable as a lean, and which two precede each midterm or
#: on-cycle Senate race.
PRES_CYCLES = (2000, 2004, 2008, 2012, 2016, 2020)


def load_rows() -> list[dict]:
    with gzip.open(RAW_POLLS, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def two_party_dem(row: dict) -> float | None:
    """Actual Democratic share of the two-party vote, from the recorded result."""
    if {row["cand1_party"], row["cand2_party"]} != {"DEM", "REP"}:
        return None
    try:
        first, second = float(row["cand1_actual"]), float(row["cand2_actual"])
    except (TypeError, ValueError):
        return None
    if first <= 0 or second <= 0:
        return None
    dem, rep = (first, second) if row["cand1_party"] == "DEM" else (second, first)
    return dem / (dem + rep)


def presidential_results(rows: list[dict]) -> dict[int, dict[str, float]]:
    """Democratic two-party share by cycle and state, from recorded results."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["type_simple"] != "Pres-G":
            continue
        share = two_party_dem(row)
        if share is None:
            continue
        out[int(row["cycle"])][row["location"].strip().upper()] = share
    return dict(out)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def margin_of(share: float) -> float:
    return 100.0 * (2.0 * share - 1.0)


def senate_races(rows: list[dict]) -> dict[tuple[int, str], float]:
    """Actual Democratic two-party share of each Senate general, by cycle/state."""
    out: dict[tuple[int, str], float] = {}
    for row in rows:
        if row["type_simple"] != "Sen-G":
            continue
        share = two_party_dem(row)
        if share is None:
            continue
        out[(int(row["cycle"]), row["location"].strip().upper())] = share
    return out


def lean_for(pres: dict[int, dict[str, float]], cycle: int, state: str):
    """Partisan lean going into a cycle, and the two elections it came from.

    Mirrors fundamentals.py: each state's presidential logit share minus the
    nation's, so the national swing is not baked into the race baseline, blended
    0.75/0.25 across the two most recent presidential elections *before* the
    race.
    """
    prior_cycles = [c for c in PRES_CYCLES if c < cycle]
    if len(prior_cycles) < 2:
        return None
    recent, previous = prior_cycles[-1], prior_cycles[-2]

    leans = []
    for pres_cycle in (recent, previous):
        table = pres.get(pres_cycle, {})
        if state not in table or "US" not in table:
            return None
        leans.append(logit(table[state]) - logit(table["US"]))
    return 0.75 * leans[0] + 0.25 * leans[1]


def main() -> int:
    rows = load_rows()
    pres = presidential_results(rows)
    senate = senate_races(rows)

    records = []
    for (cycle, state), actual_share in sorted(senate.items()):
        lean = lean_for(pres, cycle, state)
        if lean is None:
            continue
        records.append({
            "cycle": cycle,
            "state": state,
            "lean_logit": lean,
            # Logit lean expressed in points of margin, the same 50:1
            # convention the config uses (1 logit ~ 50 points).
            "lean_margin": lean * 50,
            "actual_margin": margin_of(actual_share),
            "dem_won": actual_share > 0.5,
        })

    print(f"{len(records)} Senate generals with a reconstructable lean "
          f"({min(r['cycle'] for r in records)}-{max(r['cycle'] for r in records)})\n")

    # --- 1. how often does the underdog actually win? ---------------------
    print("=" * 74)
    print("  1. Democratic win rate by how Republican the state leans")
    print("=" * 74)
    print(f"  {'lean (pts)':>16}  {'races':>6} {'D wins':>7} {'rate':>7}")
    bands = [
        ("R+25 or worse", -1e9, -25), ("R+15 to R+25", -25, -15),
        ("R+8 to R+15", -15, -8), ("R+8 to D+8", -8, 8),
        ("D+8 to D+15", 8, 15), ("D+15 or better", 15, 1e9),
    ]
    for label, low, high in bands:
        band = [r for r in records if low <= r["lean_margin"] < high]
        if not band:
            continue
        wins = sum(r["dem_won"] for r in band)
        print(f"  {label:>16}  {len(band):6d} {wins:7d} {wins / len(band) * 100:6.1f}%")

    # --- 1b. the same rates, conditioned on the national environment ------
    #
    # This is the part that matters. Each cycle's environment is the mean
    # shortfall of the actual result against what the lean alone predicts --
    # the same quantity the model calls eta. Splitting on it turns an
    # unconditional 17.6% into 5.6% in a flat year and 31.2% in a strong one,
    # and a model that reads the environment should be compared against the
    # column it is actually in.
    from collections import defaultdict as _dd

    by_cycle = _dd(list)
    for r in records:
        by_cycle[r["cycle"]].append(r)
    env = {
        cycle: statistics.mean(r["actual_margin"] - 0.95 * r["lean_margin"] for r in band)
        for cycle, band in by_cycle.items()
    }

    print()
    print("=" * 74)
    print("  1b. The same rates, split by how strong the year was for Democrats")
    print("=" * 74)
    print(f"  {'lean band':>16} {'weak/neutral year':>22} {'strong D year (D+6+)':>24}")
    for label, low, high in bands[:3]:
        line = f"  {label:>16}"
        for keep in (lambda e: e < 6, lambda e: e >= 6):
            band = [r for r in records
                    if low <= r["lean_margin"] < high and keep(env[r["cycle"]])]
            if band:
                wins = sum(r["dem_won"] for r in band)
                line += f"{wins / len(band) * 100:16.1f}% (n={len(band):2d})"
            else:
                line += f"{'-':>24}"
        print(line)
    strong = sorted(c for c in env if env[c] >= 6)
    print(f"\n  strong-Democratic cycles in this data: {strong}")
    print("  2026 is running at D+6.7 on the generic ballot, so it is the right column.")

    # --- 2. is the prediction compressed? ---------------------------------
    print()
    print("=" * 74)
    print("  2. Does the Senate margin track the presidential lean one-for-one?")
    print("=" * 74)
    x = np.array([r["lean_margin"] for r in records])
    y = np.array([r["actual_margin"] for r in records])
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = len(y) - 2
    s2 = float(resid @ resid) / dof
    se = float(np.sqrt(s2 * np.linalg.inv(design.T @ design)[1, 1]))
    print(f"  actual = {beta[0]:+.2f} + {beta[1]:.3f} x lean     (SE {se:.3f})")
    print(f"  residual sd {math.sqrt(s2):.2f} pts of margin")
    print()
    print("  config uses lean_shrinkage = 0.95, i.e. a slope of 0.95.")
    print(f"  history says {beta[1]:.3f}"
          f" ({(beta[1] - 0.95) / se:+.1f} standard errors away).")

    # Recent cycles only: ticket-splitting collapsed after 2014, so an average
    # over 2004-2020 understates how tightly the two now move together.
    print()
    for lo in (2004, 2012, 2016):
        recent = [r for r in records if r["cycle"] >= lo]
        if len(recent) < 30:
            continue
        xr = np.array([r["lean_margin"] for r in recent])
        yr = np.array([r["actual_margin"] for r in recent])
        d = np.column_stack([np.ones_like(xr), xr])
        b, *_ = np.linalg.lstsq(d, yr, rcond=None)
        res = yr - d @ b
        print(f"    {lo}+ ({len(recent):3d} races): slope {b[1]:.3f}, "
              f"residual sd {statistics.stdev(res):.2f} pts")

    # --- 3. what should prior_sd be? --------------------------------------
    print()
    print("=" * 74)
    print("  3. What spread does the fundamentals prediction actually have?")
    print("=" * 74)
    print(f"  config: prior_sd 0.15 logit = {0.15 * 50:.1f} pts of margin")
    print(f"  fitted: residual sd {math.sqrt(s2):.2f} pts = "
          f"{math.sqrt(s2) / 50:.3f} logit")
    print()
    print("  Selection bias runs one way here: this file only has races somebody")
    print("  polled, which skews competitive and flatters the underdog. A real")
    print("  effect is therefore at least this large.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
