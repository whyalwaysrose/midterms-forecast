"""Do states whose polls miss together look alike to the kernel?

`election_day_error.correlation` decides how much state polling errors move
together, and that single choice does more to set the width of the seat
distribution than almost anything else: errors that are independent average out
across 35 races, errors that are perfectly correlated do not. Its `length_scale`
and `region_weight` were hand-set.

This measures the thing they are supposed to describe. For every cycle in the
vendored FiveThirtyEight file, each state gets a polling error -- late poll
average minus actual result. Removing the cycle mean strips out the national
error, which the model represents separately as `national_sd`, and leaves the
state-specific part the kernel is meant to govern. The average product of two
states' residual errors across cycles estimates their covariance.

Then the question is simply whether the kernel agrees: are the pairs it calls
politically close the pairs that actually erred together?

FINDINGS (2026-08-24), on 13 cycles and 135 state pairs:

1. **The covariates are validated.** Agreement between the kernel's ranking and
   the measured one is +0.17 at the committed parameters and positive across
   every setting tried. States the kernel calls politically similar really do
   miss together. That was previously an assumption.

2. **No parameter change is justified.** Agreement rises as `region_weight`
   falls -- monotonically, at every length scale -- which looks like a finding
   until you resample. Bootstrapping over whole cycles, region_weight 0.0 beats
   0.6 in only 71% of resamples with a 90% interval of -0.030 to +0.063. That is
   noise. The committed values stand.

3. **The comparison must be demeaned on both sides.** See
   :func:`demeaned_prediction`. Comparing raw kernel correlations against
   demeaned measurements makes the kernel look wildly over-correlated (+0.329
   against -0.064) and would invite shrinking it toward zero -- which would let
   errors cancel across 35 races and leave the seat distribution far too narrow.
   The artefact is in the measurement, not the model. Same shape as the trap
   where the backtest appears to ask for wider error scales.

Usage:
    python scripts/fit_correlation.py
"""

from __future__ import annotations

import csv
import gzip
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW_POLLS = REPO / "data" / "history" / "raw_polls.csv.gz"

#: Polls this close to the election, so the error is a polling miss rather than
#: opinion that moved afterwards -- the same window the error scales are fitted in.
MAX_DAYS = 21
#: A state-cycle needs this many polls before its "error" means anything.
MIN_POLLS = 3
#: A pair needs this many shared cycles before its covariance is worth reading.
MIN_SHARED_CYCLES = 4


def state_errors() -> dict[int, dict[str, float]]:
    """Polling error in points, by cycle then state, signed toward Democrats."""
    with gzip.open(RAW_POLLS, "rt", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("type_simple") == "Sen-G"]

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        try:
            days = float(row["time_to_election"])
            cycle = int(row["cycle"])
        except (TypeError, ValueError):
            continue
        if not 0 <= days <= MAX_DAYS:
            continue
        if {row["cand1_party"], row["cand2_party"]} != {"DEM", "REP"}:
            continue
        sign = 1.0 if row["cand1_party"] == "DEM" else -1.0
        try:
            error = sign * (float(row["margin_poll"]) - float(row["margin_actual"]))
        except (TypeError, ValueError):
            continue
        grouped[(cycle, row["location"].strip().upper())].append(error)

    by_cycle: dict[int, dict[str, float]] = defaultdict(dict)
    for (cycle, state), errors in grouped.items():
        if len(errors) >= MIN_POLLS:
            by_cycle[cycle][state] = statistics.mean(errors)
    return dict(by_cycle)


def residualise(by_cycle: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """Remove each cycle's mean error.

    That mean is the national miss, which the model already carries as its own
    perfectly-correlated term. Leaving it in would make every pair of states look
    strongly correlated and would double-count it.
    """
    out: dict[int, dict[str, float]] = {}
    for cycle, errors in by_cycle.items():
        if len(errors) < 4:
            continue
        national = statistics.mean(errors.values())
        out[cycle] = {s: e - national for s, e in errors.items()}
    return out


def empirical_correlations(residuals):
    """Correlation per state pair, plus how many cycles support it."""
    products: dict[tuple[str, str], list[float]] = defaultdict(list)
    squares: dict[str, list[float]] = defaultdict(list)

    for errors in residuals.values():
        for state, value in errors.items():
            squares[state].append(value**2)
        for a, b in combinations(sorted(errors), 2):
            products[(a, b)].append(errors[a] * errors[b])

    variance = {s: statistics.mean(v) for s, v in squares.items() if len(v) >= 3}
    correlations: dict[tuple[str, str], tuple[float, int]] = {}
    for (a, b), values in products.items():
        if len(values) < MIN_SHARED_CYCLES or a not in variance or b not in variance:
            continue
        denominator = float(np.sqrt(variance[a] * variance[b]))
        if denominator <= 0:
            continue
        correlations[(a, b)] = (
            float(np.clip(statistics.mean(values) / denominator, -1.0, 1.0)),
            len(values),
        )
    return correlations


def main() -> int:
    from midterms import fundamentals as F
    from midterms.config import load_all
    from midterms.model.correlation import covariate_matrix, distance_matrix

    residuals = residualise(state_errors())
    print(f"{len(residuals)} cycles with enough polled states")
    print(f"  cycles: {sorted(residuals)}")

    empirical = empirical_correlations(residuals)
    print(f"{len(empirical)} state pairs with >= {MIN_SHARED_CYCLES} shared cycles\n")
    if len(empirical) < 50:
        print("too few pairs to say anything")
        return 1

    races, cfg = load_all()
    fund = F.compute(races, cfg)
    state_of = {r.id: r.unit for r in races.races}
    index = {state_of[rid]: i for i, rid in enumerate(fund.race_ids)}

    usable = [(a, b) for a, b in empirical if a in index and b in index]
    print(f"{len(usable)} of those pairs are states racing in 2026\n")
    if len(usable) < 30:
        print("too few overlapping pairs to fit against")
        return 1

    def distances_for(length_scale: float, region_weight: float):
        from dataclasses import replace

        tuned = replace(
            cfg.election_day_error.correlation,
            length_scale=length_scale,
            region_weight=region_weight,
        )
        return distance_matrix(covariate_matrix(fund, tuned)), tuned

    observed = np.array([empirical[(a, b)][0] for a, b in usable])
    weights = np.array([empirical[(a, b)][1] for a, b in usable], dtype=float)

    def kernel_matrix(length_scale: float, region_weight: float) -> np.ndarray:
        dist, tuned = distances_for(length_scale, region_weight)
        corr = (1 - tuned.nugget) * np.exp(-dist / length_scale)
        corr[np.diag_indices_from(corr)] += tuned.nugget
        np.fill_diagonal(corr, 1.0)
        return corr

    def demeaned_prediction(corr: np.ndarray) -> np.ndarray:
        """What the kernel predicts AFTER the same cycle-demeaning is applied.

        This is the correction that makes the comparison honest. Subtracting a
        cycle's mean is a linear map M = I - 11'/n, so errors drawn from the
        kernel become M e with covariance M C M'. Demeaning removes the common
        component, which is why the measured correlations average near zero -- a
        set of n states demeaned together has average pairwise correlation about
        -1/(n-1) whatever the true structure is.

        Comparing raw kernel correlations against demeaned measurements would
        therefore make the kernel look far too correlated, and 'fixing' that by
        shrinking it would leave errors that cancel across 35 races and a seat
        distribution far too narrow. Same trap as the backtest appearing to want
        wider error scales: the artefact is in the measurement, not the model.
        """
        per_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
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
            for a, b in usable
        ])

    current = cfg.election_day_error.correlation
    print("=" * 72)
    print("  Does the kernel rank pairs the way history does?")
    print("=" * 72)

    corr = kernel_matrix(current.length_scale, current.region_weight)
    raw = np.array([corr[index[a], index[b]] for a, b in usable])
    predicted = demeaned_prediction(corr)
    print(f"  current config (length_scale {current.length_scale}, "
          f"region_weight {current.region_weight})")
    print(f"    agreement with measured ranking : "
          f"{float(np.corrcoef(predicted, observed)[0, 1]):+.3f}")
    print(f"    measured pair correlations      : mean {observed.mean():+.3f}, "
          f"sd {observed.std(ddof=1):.3f}")
    print(f"    kernel, after same demeaning    : mean {predicted.mean():+.3f}, "
          f"sd {predicted.std(ddof=1):.3f}")
    print(f"    kernel, raw (NOT comparable)    : mean {raw.mean():+.3f}")

    print()
    print("=" * 72)
    print("  Grid search, comparing like with like")
    print("=" * 72)
    print(f"  {'length':>7} {'region':>7} {'agreement':>10} {'mean pred':>10} "
          f"{'weighted SSE':>13}")

    best = None
    for length_scale in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        for region_weight in (0.0, 0.3, 0.6, 1.0):
            corr = kernel_matrix(length_scale, region_weight)
            pred = demeaned_prediction(corr)
            if pred.std() < 1e-9:
                continue
            agree = float(np.corrcoef(pred, observed)[0, 1])
            sse = float(np.sum(weights * (pred - observed) ** 2) / weights.sum())
            row = (sse, length_scale, region_weight, agree, float(pred.mean()))
            if best is None or sse < best[0]:
                best = row
            print(f"  {length_scale:7.2f} {region_weight:7.2f} {agree:+10.3f} "
                  f"{pred.mean():+10.3f} {sse:13.4f}")

    sse, length_scale, region_weight, agree, mean_pred = best
    print(f"\n  best by weighted SSE: length_scale {length_scale}, "
          f"region_weight {region_weight}  (agreement {agree:+.3f})")
    print(
        "\n  Read agreement first: it is the part the demeaning cannot distort.\n"
        "  A clearly positive value means the kernel's notion of 'politically\n"
        "  similar' does track which states actually missed together. SSE is a\n"
        "  weaker guide -- with 13 cycles the individual pair correlations are\n"
        "  very noisy, so prefer a parameter change that improves agreement over\n"
        "  one that only improves SSE."
    )

    # ------------------------------------------------------------- bootstrap
    #
    # Agreement rises monotonically as region_weight falls, at every length
    # scale, which is more persuasive than any single pair of numbers. But with
    # 135 pairs the standard error on a correlation near 0.2 is about 0.09, so
    # the gap between +0.173 and +0.215 is well inside noise on its own.
    # Resampling whole cycles -- the actual unit of independence here, since
    # every pair in a cycle shares that year's polling environment -- says how
    # often the ordering survives.
    print()
    print("=" * 72)
    print("  Bootstrap over cycles: is 'region_weight hurts' real?")
    print("=" * 72)

    rng = np.random.default_rng(20260824)
    cycles = sorted(residuals)
    wins = 0
    trials = 400
    gaps = []

    for _ in range(trials):
        drawn = rng.choice(cycles, size=len(cycles), replace=True)
        resampled = {}
        for i, cycle in enumerate(drawn):
            resampled[f"{cycle}-{i}"] = residuals[cycle]
        emp = empirical_correlations(resampled)
        pairs = [(a, b) for a, b in emp if a in index and b in index]
        if len(pairs) < 30:
            continue
        obs = np.array([emp[(a, b)][0] for a, b in pairs])

        # Raw kernel correlations here, not demeaned. Only the *difference*
        # between two region weights is being read, and demeaning shifts both by
        # nearly the same amount, so it cancels out of the comparison.
        scores = {}
        for region_weight in (0.0, 0.6):
            corr = kernel_matrix(current.length_scale, region_weight)
            pred = np.array([corr[index[a], index[b]] for a, b in pairs])
            scores[region_weight] = float(np.corrcoef(pred, obs)[0, 1])
        gaps.append(scores[0.0] - scores[0.6])
        if scores[0.0] > scores[0.6]:
            wins += 1

    if gaps:
        gaps_array = np.array(gaps)
        print(f"    region_weight 0.0 beats 0.6 in {wins}/{len(gaps)} resamples "
              f"({100 * wins / len(gaps):.0f}%)")
        print(f"    agreement gap: mean {gaps_array.mean():+.3f}, "
              f"90% interval {np.quantile(gaps_array, 0.05):+.3f} to "
              f"{np.quantile(gaps_array, 0.95):+.3f}")
        if wins / len(gaps) > 0.9:
            print("    -> consistent enough to act on")
        else:
            print("    -> NOT consistent; leave the parameter alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
