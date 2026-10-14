"""Do demographic covariates improve the correlation kernel? They do not.

METHODOLOGY 5 called this "the single highest-value improvement available", on
the reasoning that the kernel's covariates are political -- 2024 lean, 2020
lean, region -- while "education and urbanicity have been the axes along which
recent polling errors actually correlated". It was deferred rather than
fabricated from unverified numbers. This measures it against real ones.

The measurement is fit_correlation.py's, reused wholesale: 13 cycles of Senate
state polling error in the 0-21 day window, cycle-demeaned to strip the national
miss the model carries separately, giving 135 state pairs with at least four
shared cycles. The score is agreement between the kernel's ranking of pairs and
the measured one, which is the part the demeaning cannot distort. Baseline, at
the committed parameters with political covariates only: +0.173.

FINDINGS (2026-10-14):

1. **Education does not help; it actively hurts.** Appending it raw takes
   agreement from +0.173 to +0.102 at weight 1.5, monotonically. The reason is
   collinearity: across the 35 states racing in 2026, BA+ share correlates
   +0.79 with 2024 lean and +0.78 with 2020 lean. The kernel already has this
   axis. Appending it again mostly stretches the one it has.

2. **Residualising makes it worse, which is the informative version.** Regress
   education on both leans and keep the residual -- the part that is genuinely
   new information -- and only 38% of its variance survives. Adding that takes
   agreement to +0.079. So the new information is not merely redundant, it is
   not predictive of which states miss together.

3. **Urbanicity is the one thing that helps, by +0.005, which is noise.**
   Residualised urban share (62% of variance surviving) reaches +0.178 against
   the +0.173 baseline. Bootstrapping over whole cycles -- the unit of
   independence, since every pair within a cycle shares that year's polling
   environment -- it wins in 62% of resamples with a 90% interval of -0.108 to
   +0.115. That is a weaker result than the region_weight change finding 2 of
   fit_correlation.py rejected, which won 71% with an interval of -0.030 to
   +0.063. By the project's own standard it is not committable.

4. **The claim is about recent cycles, and recent cycles cannot answer it.**
   Restricting to 2012 and later leaves 14 usable pairs; to 2016 and later,
   zero. There is no power here at all, in either direction. The honest
   statement is not "education does not matter" but "this dataset cannot see
   whether it does in the cycles where the realignment is claimed".

5. **At district level, where it would matter most, the measurement does not
   exist.** 51 House seats sit within five points of the line against ten in
   the Senate, so the kernel matters more for the House. But House district
   polling concentrates on whichever handful of seats is competitive in a given
   year, and those change. Within each stable-boundary era -- districts are
   redrawn every decade, so "AZ-01" in 2004 is not the same geography as in
   2014 -- the counts are:

       2002-2010   70 district-cycles   0 pairs with >=4 shared cycles
       2012-2020   45 district-cycles   0 pairs with >=4 shared cycles
       2022-        4 district-cycles   0 pairs with >=4 shared cycles

   Zero, in every era. The House kernel is not merely unconfirmed, as
   METHODOLOGY 11 says; from this dataset it is unconfirmable at district level.

6. **And the covariate values themselves would be partly invented.** 181 of 435
   districts have new 2026 lines (METHODOLOGY 9.0). The ACS publishes no
   estimates on boundaries that have never held an election, so district
   demographics for those seats would have to be apportioned from tracts -- a
   defensible method, but one whose error would land precisely on the redrawn
   seats, which are the ones the forecast is least sure about already.

CONCLUSION: the kernel is unchanged. The reason 9.1 gave for deferring this --
not fabricating from unverified numbers -- turns out to be the whole finding:
with verified numbers, at the only level and window where the data has power,
the demographic covariates do not improve it.

Usage:
    python scripts/fit_demographic_covariates.py
    python scripts/fit_demographic_covariates.py --refresh   # re-derive the CSV
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

VENDORED = REPO / "data" / "history" / "state_demographics.csv"

#: Keyless Census bulk sources. The data API now requires a key; these do not,
#: which keeps this script runnable by anyone who clones the repo.
FIPS_URL = "https://www2.census.gov/geo/docs/reference/state.txt"
EDU_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
    "table-based-SF/data/5YRData/acsdt5y2023-b15003.dat"
)
URBAN_URL = "https://www2.census.gov/geo/docs/reference/ua/PctUrbanRural_State.txt"

#: The education table is 92 MB nationally and we need 52 rows of it. Rows are
#: ordered by summary level and the state block (`0400000US`) sits inside the
#: first 400 KB, so a range request is enough. Checked, not assumed: the parser
#: fails loudly below if it does not find 51 states.
EDU_RANGE = "bytes=0-700000"

#: B15003 line numbers for bachelor's, master's, professional, doctorate.
BA_PLUS = (22, 23, 24, 25)


def _get(url: str, byte_range: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "midterms-forecast"})
    if byte_range:
        request.add_header("Range", byte_range)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def refresh() -> None:
    """Re-derive the vendored CSV from the Census bulk files."""
    fips_to_postal = {
        row["STATE"]: row["STUSAB"]
        for row in csv.DictReader(
            io.StringIO(_get(FIPS_URL).decode("utf-8")), delimiter="|"
        )
    }

    raw = _get(EDU_URL, EDU_RANGE).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(raw), delimiter="|"))
    idx = {name: i for i, name in enumerate(rows[0])}
    total_col = idx["B15003_E001"]
    ba_cols = [idx[f"B15003_E{n:03d}"] for n in BA_PLUS]

    education: dict[str, float] = {}
    for row in rows[1:]:
        if not row or not row[0].startswith("0400000US"):
            continue
        postal = fips_to_postal.get(row[0][-2:])
        try:
            total = float(row[total_col])
            ba = sum(float(row[i]) for i in ba_cols)
        except (ValueError, IndexError):
            continue
        if postal and total > 0:
            education[postal] = 100.0 * ba / total

    urban: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(_get(URBAN_URL).decode("utf-8"))):
        postal = fips_to_postal.get(row["STATE"])
        try:
            value = float(row["POPPCT_URBAN"])
        except (TypeError, ValueError):
            continue
        if postal:
            urban[postal] = value

    shared = sorted(set(education) & set(urban))
    if len(shared) < 51:
        raise RuntimeError(
            f"only {len(shared)} states resolved (expected 51 with DC). "
            "The range request may no longer reach the state block, or a "
            "source's format changed."
        )

    VENDORED.parent.mkdir(parents=True, exist_ok=True)
    with VENDORED.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["state", "ba_plus_pct", "urban_pct"])
        for postal in shared:
            writer.writerow([
                postal, f"{education[postal]:.2f}", f"{urban[postal]:.2f}"
            ])
    print(f"wrote {VENDORED} ({len(shared)} states)")


def load_demographics() -> dict[str, dict[str, float]]:
    if not VENDORED.exists():
        raise SystemExit(
            f"{VENDORED} is missing; run with --refresh to derive it from Census."
        )
    out = {}
    with VENDORED.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["state"]] = {
                "education": float(row["ba_plus_pct"]),
                "urban": float(row["urban_pct"]),
            }
    return out


def residualise_against(target: np.ndarray, *controls: np.ndarray) -> np.ndarray:
    """``target`` with the controls' linear explanation removed.

    The part of a demographic covariate the kernel does not already carry.
    Skipping this is the badly-posed version of the experiment: appending a
    column 0.79 correlated with 2024 lean mostly re-weights the existing axis
    rather than adding a new one.
    """
    design = np.column_stack([np.ones(len(target)), *controls])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ beta


def _demeaned(corr, residuals, pairs, index) -> np.ndarray:
    """fit_correlation.demeaned_prediction, parameterised so it can resample.

    See that function for why the comparison must be demeaned on both sides.
    """
    per_pair = defaultdict(list)
    for errors in residuals.values():
        states = [s for s in sorted(errors) if s in index]
        if len(states) < 4:
            continue
        rows = [index[s] for s in states]
        sub = corr[np.ix_(rows, rows)]
        n = len(states)
        centring = np.eye(n) - np.ones((n, n)) / n
        covariance = centring @ sub @ centring.T
        sd = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        implied = covariance / np.outer(sd, sd)
        for i, a in enumerate(states):
            for j, b in enumerate(states):
                if i < j:
                    per_pair[(a, b)].append(float(implied[i, j]))
    return np.array([
        float(np.mean(per_pair[(a, b)])) if per_pair.get((a, b)) else 0.0
        for a, b in pairs
    ])


def district_pair_counts() -> dict[str, tuple[int, int]]:
    """District-cycles and recurring pairs per redistricting era.

    Finding 5. Boundaries change every decade, so a pair of district labels
    only means one thing within an era.
    """
    import gzip
    import statistics
    from itertools import combinations

    path = REPO / "data" / "history" / "raw_polls.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("type_simple") == "House-G"]

    eras = {"2002-2010": range(2002, 2011),
            "2012-2020": range(2012, 2021),
            "2022-": range(2022, 2027)}
    out = {}
    for era, years in eras.items():
        grouped = defaultdict(list)
        for row in rows:
            try:
                cycle, days = int(row["cycle"]), float(row["time_to_election"])
            except (TypeError, ValueError):
                continue
            if cycle not in years or not 0 <= days <= 21:
                continue
            if {row["cand1_party"], row["cand2_party"]} != {"DEM", "REP"}:
                continue
            sign = 1.0 if row["cand1_party"] == "DEM" else -1.0
            try:
                error = sign * (
                    float(row["margin_poll"]) - float(row["margin_actual"])
                )
            except (TypeError, ValueError):
                continue
            grouped[(cycle, row["location"].strip().upper())].append(error)

        by_cycle = defaultdict(dict)
        for (cycle, loc), errors in grouped.items():
            if len(errors) >= 3:
                by_cycle[cycle][loc] = statistics.mean(errors)

        shared = defaultdict(int)
        for errors in by_cycle.values():
            for a, b in combinations(sorted(errors), 2):
                shared[(a, b)] += 1
        out[era] = (
            sum(len(v) for v in by_cycle.values()),
            sum(1 for n in shared.values() if n >= 4),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-derive the vendored CSV from Census bulk files")
    parser.add_argument("--trials", type=int, default=400)
    args = parser.parse_args()

    if args.refresh:
        refresh()

    import fit_correlation as FC

    from midterms import fundamentals as F
    from midterms.config import load_all
    from midterms.model.correlation import _standardise, distance_matrix

    residuals = FC.residualise(FC.state_errors())
    empirical = FC.empirical_correlations(residuals)

    races, cfg = load_all()
    fund = F.compute(races, cfg)
    corr_cfg = cfg.election_day_error.correlation
    state_of = {race.id: race.unit for race in races.races}
    index = {state_of[rid]: i for i, rid in enumerate(fund.race_ids)}
    pairs = [(a, b) for a, b in empirical if a in index and b in index]
    observed = np.array([empirical[(a, b)][0] for a, b in pairs])

    demo = load_demographics()
    units = [state_of[rid] for rid in fund.race_ids]
    missing = [u for u in units if u not in demo]
    if missing:
        print(f"no demographics for {missing}; refusing to score with gaps")
        return 1
    education = np.array([demo[u]["education"] for u in units])
    urban = np.array([demo[u]["urban"] for u in units])

    print(f"{len(residuals)} cycles, {len(pairs)} usable state pairs, "
          f"{len(units)} races")
    print(f"  BA+ share   {education.min():5.1f}-{education.max():5.1f}%   "
          f"r(lean 2024) {np.corrcoef(education, fund.lean_2024)[0, 1]:+.2f}")
    print(f"  urban share {urban.min():5.1f}-{urban.max():5.1f}%   "
          f"r(lean 2024) {np.corrcoef(urban, fund.lean_2024)[0, 1]:+.2f}")

    education_r = residualise_against(education, fund.lean_2024, fund.lean_2020)
    urban_r = residualise_against(urban, fund.lean_2024, fund.lean_2020)
    print(f"  variance surviving residualisation: education "
          f"{np.var(education_r) / np.var(education):.0%}, urban "
          f"{np.var(urban_r) / np.var(urban):.0%}")
    print()

    def kernel(extras: list[tuple[np.ndarray, float]]) -> np.ndarray:
        columns = [_standardise(fund.lean_2024), _standardise(fund.lean_2020)]
        regions = sorted(set(fund.regions))
        onehot = np.zeros((len(fund.regions), len(regions)))
        for i, region in enumerate(fund.regions):
            onehot[i, regions.index(region)] = 1.0
        columns.append(onehot * corr_cfg.region_weight)
        for vector, weight in extras:
            columns.append(_standardise(vector) * weight)
        distances = distance_matrix(np.column_stack(columns))
        corr = (1 - corr_cfg.nugget) * np.exp(-distances / corr_cfg.length_scale)
        corr[np.diag_indices_from(corr)] += corr_cfg.nugget
        np.fill_diagonal(corr, 1.0)
        return corr

    def score(extras, resid=None, prs=None, obs=None) -> float:
        predicted = _demeaned(
            kernel(extras),
            resid if resid is not None else residuals,
            prs if prs is not None else pairs,
            index,
        )
        if predicted.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(
            predicted, obs if obs is not None else observed
        )[0, 1])

    specs = {
        "education raw": lambda w: [(education, w)],
        "education resid": lambda w: [(education_r, w)],
        "urban raw": lambda w: [(urban, w)],
        "urban resid": lambda w: [(urban_r, w)],
        "both resid": lambda w: [(education_r, w), (urban_r, w)],
    }
    weights = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)

    base = score([])
    print("=" * 74)
    print(f"  Agreement with the measured ranking. Baseline (political) {base:+.3f}")
    print("=" * 74)
    print("  weight " + "".join(f"{name:>17}" for name in specs))
    table = {}
    for weight in weights:
        row = {name: score(build(weight)) for name, build in specs.items()}
        table[weight] = row
        print(f"  {weight:6.2f} " + "".join(f"{row[n]:+17.3f}" for n in specs))

    best = max(
        ((row[n], n, w) for w, row in table.items() for n in specs),
        key=lambda t: t[0],
    )
    best_value, best_name, best_weight = best
    print()
    if best_value <= base:
        print("  Nothing beats the political baseline. No bootstrap needed: "
              "there is\n  no candidate improvement to test.")
        _print_district_note()
        return 0

    print(f"  best: {best_name} at weight {best_weight} -> {best_value:+.3f} "
          f"({best_value - base:+.3f})")
    print()
    print("=" * 74)
    print("  Bootstrap over whole cycles: is that gain real?")
    print("=" * 74)

    rng = np.random.default_rng(20261014)
    cycles = sorted(residuals)
    gaps, wins = [], 0
    for _ in range(args.trials):
        drawn = rng.choice(cycles, size=len(cycles), replace=True)
        resampled = {f"{c}-{i}": residuals[c] for i, c in enumerate(drawn)}
        emp = FC.empirical_correlations(resampled)
        prs = [(a, b) for a, b in emp if a in index and b in index]
        if len(prs) < 30:
            continue
        obs = np.array([emp[(a, b)][0] for a, b in prs])
        a_base = score([], resampled, prs, obs)
        a_new = score(specs[best_name](best_weight), resampled, prs, obs)
        if np.isnan(a_base) or np.isnan(a_new):
            continue
        gaps.append(a_new - a_base)
        wins += a_new > a_base

    gaps = np.array(gaps)
    low, high = np.percentile(gaps, 5), np.percentile(gaps, 95)
    print(f"  {len(gaps)} usable resamples")
    print(f"  the demographic covariate wins in {wins / len(gaps):.0%}")
    print(f"  gain: mean {gaps.mean():+.4f}, 90% interval {low:+.4f} to {high:+.4f}")
    if low > 0:
        print("  -> the interval excludes zero. This is a real improvement.")
    else:
        print("  -> the interval includes zero, so this is noise by the same "
              "standard\n     that rejected the region_weight change "
              "(fit_correlation.py finding 2).\n     Do not commit it.")
    _print_district_note()
    return 0


def _print_district_note() -> None:
    print()
    print("=" * 74)
    print("  And at district level, where the kernel matters more?")
    print("=" * 74)
    counts = district_pair_counts()
    print(f"  {'era':<12}{'district-cycles':>17}{'pairs >=4 cycles':>19}")
    for era, (district_cycles, recurring) in counts.items():
        print(f"  {era:<12}{district_cycles:>17}{recurring:>19}")
    print(
        "\n  House polling concentrates on whichever seats are competitive that\n"
        "  year, and those change, so no pair of districts recurs often enough to\n"
        "  estimate a correlation. Boundaries are redrawn every decade, so pairs\n"
        "  cannot be pooled across eras either. The House kernel is unconfirmable\n"
        "  at district level from this dataset -- not merely unconfirmed."
    )


if __name__ == "__main__":
    sys.exit(main())
