"""Where the 2026 result fell in the forecast's own distribution.

METHODOLOGY 8c pre-registered, seven weeks before the election, that the
chamber outcome would be scored partly on "where in the predicted distribution
it fell". `midterms score` answers that for races and chambers. This answers a
different question the result raised on the night: **the misses were not spread
evenly, they clustered by state.** Four of the first six House misses were in
North Carolina.

Clustered misses have two very different explanations and they call for
opposite responses:

  a bad lean      the district's own baseline was wrong, as Oregon's was in
                  2024 when OR-03 came out 30 points off and sat live in a
                  published forecast for weeks. That is a data fault, and the
                  fix is to the data.

  a correlated    the districts are politically near-identical, the national
  environment     environment moved, and they all went together. That is not a
                  fault at all -- it is the event the correlated election-day
                  error of METHODOLOGY 5 exists to price -- and "fixing" it by
                  shrinking the kernel would make the seat interval far too
                  narrow, which is the trap 11b and fit_correlation.py's
                  finding 3 both warn about.

They are distinguishable, which is the point of this script.

FINDINGS (2026-11-04, on the races called by then):

1. **North Carolina's lean data is sound.** Its fourteen districts average
   48.50% Democratic two-party for 2024 against an actual statewide 48.35%.
   Congressional districts are equal-population by construction, so their
   unweighted mean has to land on the statewide figure; it does, to 0.15
   points. Whatever went wrong in North Carolina, it was not the baseline --
   and this is the independent check METHODOLOGY 9.0 said was missing for the
   181 redrawn seats, arriving from the election itself.

2. **The map is built to fail all at once.** Three districts are packed at
   D+18 to D+24 and the other eleven are drawn in a band five points wide,
   R+4.0 to R+9.0. A map like that wins eleven of fourteen under the conditions
   it was drawn for and loses the lot together when the environment moves,
   because those eleven are politically almost the same district.

3. **The kernel saw that, and the clustered misses are one event.** Mean
   election-day correlation among the four missed districts is 0.940, against
   0.314 across all 435. Each was given about 30%. Under that correlation the
   chance of all four flipping is near 21%; under independence it would be
   0.8%. A one-in-five outcome occurred. Counting it as four independent
   failures overstates the damage by a factor of about twenty-five.

4. **What was actually wrong was the centre, not the spread.** Those eleven
   marginals needed roughly a seven-point Democratic shift from 2024 to fall,
   and got it. The same signal appears in the Senate reliability curve, where
   every competitive race broke Democratic. The model's national term pointed
   the right way and not far enough -- which is a statement about the central
   estimate, and the opposite of the correction the headline comparison with
   more confident forecasters appears to invite.

Usage:
    python scripts/postmortem_2026.py
    python scripts/postmortem_2026.py --chamber senate
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: A cluster needs this many called misses in one state before its correlation
#: is worth reporting. Two races going the same way is not a pattern.
MIN_CLUSTER = 3

#: Draws for the joint-flip estimate. Large enough that the reported
#: probability is stable to a tenth of a point.
N_DRAWS = 200_000


def joint_flip_probability(
    marginals: list[float], rho: float, rng: np.random.Generator
) -> tuple[float, float]:
    """P(all of them flip), under the kernel's correlation and independently.

    An equicorrelated Gaussian copula: each race flips when its latent draw
    clears the threshold its own marginal implies. This is the kernel's own
    structure, not a new assumption -- the point is to express what the model
    already believed about these races jointly, which a per-race probability
    cannot show.
    """
    k = len(marginals)
    thresholds = np.array([_z_for(p) for p in marginals])

    common = rng.standard_normal((N_DRAWS, 1))
    idio = rng.standard_normal((N_DRAWS, k))
    z = np.sqrt(rho) * common + np.sqrt(max(0.0, 1.0 - rho)) * idio
    together = float((z > thresholds).all(axis=1).mean())

    independent = float(np.prod(marginals))
    return together, independent


def _z_for(p: float) -> float:
    """The standard-normal threshold a marginal probability implies."""
    from math import erf, sqrt

    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        tail = 0.5 * (1.0 - erf(mid / sqrt(2.0)))
        if tail > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chamber", default="house", choices=["house", "senate"])
    args = parser.parse_args()

    from midterms import fundamentals as F
    from midterms.config import load_all
    from midterms.data import results as R
    from midterms.model.correlation import correlation_matrix
    from midterms.score import forecast_of_record

    races, cfg = load_all(chamber=args.chamber)
    fund = F.compute(races, cfg)
    corr = correlation_matrix(fund, cfg.election_day_error.correlation)
    index = {rid: i for i, rid in enumerate(fund.race_ids)}

    payload = forecast_of_record(args.chamber)
    if isinstance(payload, tuple):
        payload = payload[0]
    forecast = {r["id"]: r for r in payload["races"]}

    outcomes = (
        R.fetch_house_districts() if args.chamber == "house"
        else R.fetch_senate_races()
    )
    if not outcomes:
        print("No results published yet. Nothing to examine.")
        return 0

    called = [(rid, o) for rid, o in outcomes.items() if rid in forecast]
    misses = [
        (rid, o) for rid, o in called
        if (forecast[rid]["dem_win_prob"] >= 0.5) != o.dem_won
    ]
    print("=" * 72)
    print(f"  WHERE THE 2026 {args.chamber.upper()} RESULT FELL -- "
          f"forecast of record {payload['run_date']}")
    print("=" * 72)
    print(f"  {len(called)} races called, {len(called) - len(misses)} called correctly, "
          f"{len(misses)} missed")

    # --- did the misses cluster? ------------------------------------------
    by_state: dict[str, list[tuple[str, object]]] = {}
    for rid, o in misses:
        by_state.setdefault(_state_of(rid), []).append((rid, o))

    print("\n  Misses by state: "
          + ", ".join(f"{st} {len(v)}" for st, v in
                      sorted(by_state.items(), key=lambda kv: -len(kv[1])))
          or "  none")

    rng = np.random.default_rng(20261104)
    clusters = {st: v for st, v in by_state.items() if len(v) >= MIN_CLUSTER}
    if not clusters:
        print(f"\n  No state has {MIN_CLUSTER} or more missed races yet; "
              "nothing to decompose.")
    for state, group in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        ids = [rid for rid, _ in group]
        rows = [index[r] for r in ids]
        pairs = [corr[a, b] for n, a in enumerate(rows) for b in rows[n + 1:]]
        rho = statistics.fmean(pairs)
        # The probability the model gave the side that actually won. For a
        # race we called for the Republicans and the Democrats took, that is
        # `dem_win_prob`; for one we called the other way it is its complement.
        # Writing it as `dem_win_prob` outright happens to be right for North
        # Carolina and would be silently wrong the first time a cluster broke
        # the other way.
        marginals = [
            forecast[r]["dem_win_prob"] if o.dem_won
            else 1.0 - forecast[r]["dem_win_prob"]
            for r, o in group
        ]
        together, independent = joint_flip_probability(marginals, rho, rng)

        off = ~np.eye(len(corr), dtype=bool)
        print(f"\n  --- {state}: {len(ids)} missed races ---")
        print(f"      {', '.join(r.split('-', 2)[-1] for r in ids)}")
        print(f"      mean correlation between them : {rho:.3f}")
        print(f"      mean across the whole chamber : {corr[off].mean():.3f}")
        print(f"      each given                    : "
              f"{', '.join(f'{100*p:.0f}%' for p in marginals)}")
        print(f"      P(all of them) at that rho    : {together:6.2%}")
        print(f"      P(all of them) if independent : {independent:6.2%}")
        print(f"      -> counting these as {len(ids)} independent failures "
              f"overstates it {together / independent:.0f}x")

    # --- is the baseline sound? -------------------------------------------
    if args.chamber == "house" and clusters:
        print()
        print("  Lean check (districts are equal-population, so their unweighted")
        print("  mean must land on the statewide 2024 share):")
        for state in sorted(clusters):
            shares = [
                r.pres_2024_dem_two_party for r in races.races
                if _state_of(r.id) == state
            ]
            print(f"      {state}: {len(shares)} districts average "
                  f"{100 * statistics.fmean(shares):.2f}% D two-party, 2024")
        print("      Compare against the state's actual 2024 result; a gap of more")
        print("      than about a point is a data fault, not an election result.")

    print("=" * 72)
    return 0


def _state_of(race_id: str) -> str:
    """'house-2026-NC-09' -> 'NC';  'senate-2026-GA' -> 'GA'."""
    tail = race_id.split("-", 2)[-1]
    return tail[:2]


if __name__ == "__main__":
    sys.exit(main())
