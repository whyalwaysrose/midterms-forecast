"""Does a fundraising edge tell us anything the model does not already know?

CONCLUSION (2026-08-24): not usable as it stands. Two findings, one clean and
one contaminated.

**Against polling error, it is entirely incumbency.** Individual contributions
predict beating your polls at t=+3.18 on their own, but adding incumbency
collapses the money coefficient to +0.071 (t=+0.27) while incumbency itself sits
at +2.505 (t=+4.59). Leave-one-cycle-out has the money term wandering between
-0.06 and +0.22 -- no stable signal. The fundamentals prior already carries
incumbency, so this adds nothing. That negative result is robust: the lookahead
described below would have *helped* the money term, and it still vanished.

**Against the fundamentals baseline, the apparent effect is lookahead.** Money
predicts the result beyond state and incumbency at t=+7.72, cutting residual sd
from 8.59 to 7.05 points, which looks decisive until you notice what the
covariate is. FEC `weball` files are end-of-cycle: Warnock's 2022 total covers
through 2022-12-31. A live forecast in August has filings through June 30 only,
and a candidate's fundraising accelerates once they start visibly winning. So
the fit is partly using the outcome to predict the outcome.

Testing it honestly needs cumulative totals as of June 30 of each election year,
which means per-report data from `/committee/{id}/reports/` for roughly 350
committees across seven cycles. That needs an FEC API key -- DEMO_KEY allows 30
requests an hour. Until then this is unmeasured, not measured-and-rejected.

--------------------------------------------------------------------------

Original question: does a fundraising edge predict where the polls will be wrong?

That is the only question worth asking here. The model already uses polls, so a
covariate earns its place only if it says something the polls do not. Regressing
the actual result on fundraising would mostly rediscover that well-funded
candidates are ahead in the polls too; regressing the *polling error* on
fundraising asks whether money knows something the polls miss.

    margin_actual - margin_poll_average  ~  fundraising edge

Positive coefficient means a candidate who out-raises their opponent tends to
beat their polls. That is Silver's finding, and this checks it against the same
races this project already uses to calibrate its error scales.

Inputs are FEC bulk "all candidates" files (public domain, one per cycle) and the
vendored FiveThirtyEight poll file, which carries both the poll margin and the
actual result. Nothing here is fetched at forecast time.

Usage:
    python scripts/fit_fundraising.py --fec-dir <dir with weballNN.zip>
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW_POLLS = REPO / "data" / "history" / "raw_polls.csv.gz"

CYCLES = (2010, 2012, 2014, 2016, 2018, 2020, 2022)

# weball column positions (the file has no header row).
CAND_ID, CAND_NAME, CAND_ICI, _PTY_CD, PARTY = 0, 1, 2, 3, 4
TTL_RECEIPTS = 5
TTL_INDIV_CONTRIB = 17
CAND_OFFICE_ST = 18

DEM = {"DEM", "D", "DFL"}
REP = {"REP", "R", "GOP"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "mr", "mrs", "ms", "dr", "hon"}


def surname(name: str) -> str:
    """Family name, from either "SMITH, JOHN A JR" or "John A. Smith"."""
    name = name.strip()
    if "," in name:
        family = name.split(",")[0]
    else:
        parts = [p for p in re.split(r"\s+", name) if p]
        parts = [p for p in parts if p.strip(".").lower() not in SUFFIXES]
        family = parts[-1] if parts else name
    return re.sub(r"[^a-z]", "", family.lower())


@dataclass
class Filing:
    name: str
    party: str
    state: str
    receipts: float
    individual: float
    incumbent: bool


def load_filings(path: Path) -> dict[tuple[str, str], list[Filing]]:
    """Senate filings from one cycle, keyed by (state, surname)."""
    with zipfile.ZipFile(path) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", "replace")

    by_key: dict[tuple[str, str], list[Filing]] = defaultdict(list)
    for line in text.splitlines():
        row = line.split("|")
        if len(row) <= CAND_OFFICE_ST or not row[CAND_ID].startswith("S"):
            continue
        party = row[PARTY].strip().upper()
        if party not in DEM | REP:
            continue
        try:
            receipts = float(row[TTL_RECEIPTS] or 0)
            individual = float(row[TTL_INDIV_CONTRIB] or 0)
        except ValueError:
            continue
        filing = Filing(
            name=row[CAND_NAME].strip(),
            party="D" if party in DEM else "R",
            state=row[CAND_OFFICE_ST].strip().upper(),
            receipts=receipts,
            individual=individual,
            incumbent=row[CAND_ICI].strip().upper() == "I",
        )
        by_key[(filing.state, surname(filing.name))].append(filing)
    return by_key


def load_races() -> dict[str, dict]:
    """One row per historical Senate general, with poll average and result."""
    with gzip.open(RAW_POLLS, "rt", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("type_simple") == "Sen-G"]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)

    races: dict[str, dict] = {}
    for race_id, polls in grouped.items():
        cycle = int(polls[0]["cycle"])
        if cycle not in CYCLES:
            continue
        # Late polls only: an average dominated by year-old polls would make the
        # "polling error" mostly staleness, which fundraising should not be
        # expected to explain.
        late = [p for p in polls if 0 <= float(p["time_to_election"] or 999) <= 45]
        if len(late) < 3:
            continue

        first = late[0]
        # margin_* is signed toward cand1; flip when cand1 is the Republican so
        # every race is on one axis.
        sign = 1.0 if first["cand1_party"] == "DEM" else -1.0
        if {first["cand1_party"], first["cand2_party"]} != {"DEM", "REP"}:
            continue

        races[race_id] = {
            "cycle": cycle,
            "state": first["location"].strip().upper(),
            "dem": first["cand1_name"] if sign > 0 else first["cand2_name"],
            "rep": first["cand2_name"] if sign > 0 else first["cand1_name"],
            "poll_margin": sign * statistics.mean(
                float(p["margin_poll"]) for p in late
            ),
            "actual_margin": sign * float(first["margin_actual"]),
            "n_polls": len(late),
        }
    return races


def match(filings, state: str, name: str, party: str) -> Filing | None:
    """The filing for one candidate, or None.

    Picks the best-funded match when several people share a surname in a state:
    the general-election candidate is essentially always the one who raised most.
    """
    candidates = [
        f for f in filings.get((state, surname(name)), []) if f.party == party
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.receipts)


def edge(dem: float, rep: float) -> float | None:
    """Log ratio of the two candidates' money, positive toward the Democrat.

    A log ratio rather than a share because money is heavy-tailed: the
    difference between $1M and $2M matters about as much as between $10M and
    $20M, which a raw share would not capture.
    """
    if dem <= 0 or rep <= 0:
        return None
    return float(np.log(dem / rep))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="write the matched table here")
    args = parser.parse_args()

    races = load_races()
    print(f"{len(races)} historical Senate generals with 3+ polls inside 45 days\n")

    matched: list[dict] = []
    unmatched = 0
    for race_id, race in sorted(races.items()):
        path = args.fec_dir / f"weball{str(race['cycle'])[2:]}.zip"
        if not path.exists():
            continue
        filings = load_filings(path)
        dem = match(filings, race["state"], race["dem"], "D")
        rep = match(filings, race["state"], race["rep"], "R")
        if dem is None or rep is None:
            unmatched += 1
            continue

        total = edge(dem.receipts, rep.receipts)
        individual = edge(dem.individual, rep.individual)
        if total is None or individual is None:
            unmatched += 1
            continue

        matched.append({
            "race_id": race_id,
            "cycle": race["cycle"],
            "state": race["state"],
            "poll_margin": race["poll_margin"],
            "actual_margin": race["actual_margin"],
            "poll_error": race["actual_margin"] - race["poll_margin"],
            "receipts_edge": total,
            "individual_edge": individual,
            "dem_incumbent": int(dem.incumbent),
            "rep_incumbent": int(rep.incumbent),
        })

    print(f"matched {len(matched)} races to FEC filings ({unmatched} unmatched)\n")
    if len(matched) < 40:
        print("too few matches to fit anything trustworthy")
        return 1

    y = np.array([m["poll_error"] for m in matched])
    print(f"polling error: mean {y.mean():+.2f} pts, sd {y.std(ddof=1):.2f}\n")

    print("=" * 70)
    print("  Does a fundraising edge predict beating your polls?")
    print("=" * 70)

    for label, key in (("total receipts", "receipts_edge"),
                       ("individual contributions", "individual_edge")):
        x = np.array([m[key] for m in matched])
        # Simple OLS with an intercept, plus the standard error, so "no effect"
        # is distinguishable from "an effect we cannot measure".
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        dof = len(y) - 2
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(design.T @ design)
        se = float(np.sqrt(cov[1, 1]))
        t = beta[1] / se
        r = float(np.corrcoef(x, y)[0, 1])

        print(f"\n  {label}")
        print(f"    slope     {beta[1]:+.3f} pts per log-unit of money  (SE {se:.3f})")
        print(f"    t         {t:+.2f}")
        print(f"    r         {r:+.3f}")
        print(f"    residual  sd {np.sqrt(s2):.2f} pts, against {y.std(ddof=1):.2f} unexplained")
        verdict = (
            "signal" if abs(t) >= 2
            else "nothing measurable" if abs(t) < 1
            else "suggestive, not significant"
        )
        print(f"    verdict   {verdict}")

    # ---------------------------------------------------------------- confounds
    #
    # The obvious alternative explanation is incumbency: incumbents out-raise
    # challengers enormously, so a "money effect" could just be an incumbency
    # effect wearing a disguise -- and the model already has incumbency in its
    # fundamentals prior. If the slope survives controlling for it, the money is
    # telling us something incumbency does not.
    print()
    print("=" * 70)
    print("  Is it just incumbency?")
    print("=" * 70)

    incumbency = np.array(
        [m["dem_incumbent"] - m["rep_incumbent"] for m in matched], dtype=float
    )
    x = np.array([m["individual_edge"] for m in matched])

    for label, columns in (
        ("incumbency alone", [incumbency]),
        ("money alone", [x]),
        ("both", [x, incumbency]),
    ):
        design = np.column_stack([np.ones(len(y)), *columns])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        dof = len(y) - design.shape[1]
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(design.T @ design)
        terms = []
        for i, name in enumerate(["money", "incumbency"][: len(columns)], start=1):
            se = float(np.sqrt(cov[i, i]))
            terms.append(f"{name} {beta[i]:+.3f} (t {beta[i] / se:+.2f})")
        print(f"    {label:18s} {'  '.join(terms):52s} resid sd {np.sqrt(s2):.2f}")

    print(
        "\n  If the money coefficient holds up in the 'both' row, it is carrying\n"
        "  information the incumbency term in fundamentals does not already have."
    )

    # --- stability across cycles -----------------------------------------
    print()
    print("=" * 70)
    print("  Leave-one-cycle-out: is it one cycle doing all the work?")
    print("=" * 70)
    for held in CYCLES:
        keep = [m for m in matched if m["cycle"] != held]
        if len(keep) < 40:
            continue
        yk = np.array([m["poll_error"] for m in keep])
        xk = np.array([m["individual_edge"] for m in keep])
        ik = np.array([m["dem_incumbent"] - m["rep_incumbent"] for m in keep], float)
        design = np.column_stack([np.ones(len(yk)), xk, ik])
        beta, *_ = np.linalg.lstsq(design, yk, rcond=None)
        n_held = sum(1 for m in matched if m["cycle"] == held)
        print(f"    without {held} (n={n_held:2d} dropped): money {beta[1]:+.3f}")

    # A 2x money advantage is the scale a reader can picture.
    design = np.column_stack([np.ones(len(y)), x, incumbency])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    print(f"\n  Controlling for incumbency, a candidate raising twice their "
          f"opponent\n  from individuals beats their polls by "
          f"{beta[1] * np.log(2):+.2f} pts.")

    # ------------------------------------------------- the actual use case
    #
    # The test above asks whether money beats the POLLS, which is the right
    # question for a polled race. But fundamentals matter most where there are
    # no polls, and there the competition is not the polls -- it is presidential
    # lean plus incumbency, which is all the prior currently knows. So: does
    # money predict the RESULT beyond the state and who holds the seat?
    #
    # State fixed effects stand in for partisan lean. They absorb it more
    # completely than a presidential-lean covariate would, which makes this a
    # conservative test: anything money explains here is on top of a very well
    # controlled baseline.
    print()
    print("=" * 70)
    print("  Does money predict the RESULT, beyond state and incumbency?")
    print("=" * 70)

    result = np.array([m["actual_margin"] for m in matched])
    states = sorted({m["state"] for m in matched})
    dummies = np.array(
        [[1.0 if m["state"] == s else 0.0 for s in states[1:]] for m in matched]
    )

    base = np.column_stack([np.ones(len(result)), incumbency, dummies])
    full = np.column_stack([np.ones(len(result)), incumbency, dummies, x])

    for label, design in (("state + incumbency", base), ("+ money", full)):
        beta, *_ = np.linalg.lstsq(design, result, rcond=None)
        resid = result - design @ beta
        dof = len(result) - design.shape[1]
        s2 = float(resid @ resid) / dof
        line = f"    {label:22s} residual sd {np.sqrt(s2):5.2f} pts"
        if label == "+ money":
            cov = s2 * np.linalg.pinv(design.T @ design)
            se = float(np.sqrt(cov[-1, -1]))
            line += f"   money {beta[-1]:+.3f} (t {beta[-1] / se:+.2f})"
        print(line)

    print(
        f"\n  {len(states)} states, {len(matched)} races. If the residual sd barely\n"
        "  moves and the t is small, money is not adding to the fundamentals\n"
        "  prior either -- it was proxying for the incumbency term already there."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(matched[0]))
            writer.writeheader()
            writer.writerows(matched)
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
