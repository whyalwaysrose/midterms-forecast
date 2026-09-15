"""Score the 2026 forecast against the 2026 result.

Everything else in this project is a proxy. ``backtest.py`` asks whether the
model predicts polls it has not seen; ``backtest_history.py`` asks whether a
simplified version of it would have called 2010-2022 Senate races at the right
rate. Both are necessary and neither is the thing. This module is the thing,
and it can only be run once.

WRITTEN BEFORE THE RESULT, ON PURPOSE
-------------------------------------
Committed 2026-09-15, seven weeks before the election. That timing is the point
rather than a convenience: a metric chosen after seeing the result is not a
measurement, it is a press release. What gets computed is fixed here, now,
while nobody knows the answer:

* **Per race** -- Brier score, log score, a reliability curve, and coverage of
  the 50% and 90% margin intervals. Scored against two baselines, because a Brier
  score on its own is unreadable: always saying 50%, and always calling the
  fundamentals favourite. A model that cannot beat those has learned nothing
  from the polls.

* **Per chamber** -- the probability assigned to the side that actually won,
  whether the true seat total fell inside the 90% interval, and where it fell
  in the predicted distribution. The last is the interesting one. A single
  election cannot tell you an interval was too wide, but it can tell you the
  result landed at the 3rd percentile, which is a start.

* **Against the other forecasters** -- the same Brier score on the same outcome
  for every chamber probability in ``data/rivals/forecasts.csv``. This is the
  only thing that settles September's disagreement, where 50+1 had the House at
  97%, DDHQ at 68% and this model at 71.9%.

WHAT ONE ELECTION CAN AND CANNOT SETTLE
---------------------------------------
Chamber-level, almost nothing. Two outcomes, one draw. If the Democrats take
the House, 50+1's 97% scores better than this model's 71.9%, and that is close
to meaningless -- the confident forecast wins whenever it is right, which is
what being confident means. Only a long run of elections separates
well-calibrated from lucky.

Race-level is different. 435 House districts and 35 Senate races are enough for
a reliability curve to say something real, and coverage of the margin intervals
is a direct test of the width that September's comparison put in question.

So the honest reading, fixed in advance: **the per-race calibration is
evidence, the chamber result is an anecdote.** If this model loses the chamber
comparison and wins the reliability curve, the reliability curve is the one
that matters, and this paragraph exists so that claim cannot be made to look
like special pleading after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Probabilities are clipped before taking a log, so a confident miss costs a
#: large finite number rather than infinity. At 1e-4 the worst possible per-race
#: log score is about 9.2.
LOG_CLIP = 1e-4

#: Reliability curve buckets. Ten is too many for 35 Senate races and about
#: right for 435 House districts; the curve reports the count in each bucket so
#: a bucket holding two races is visibly not evidence.
RELIABILITY_EDGES = (0.0, 0.05, 0.15, 0.25, 0.35, 0.45,
                     0.55, 0.65, 0.75, 0.85, 0.95, 1.0)


@dataclass(frozen=True)
class RaceResult:
    """What actually happened in one race."""

    race_id: str
    dem_won: bool
    #: Democratic margin in points, where known. Optional because a race can be
    #: called long before the count is final, and a called race with no margin
    #: still scores the win probability.
    dem_margin: float | None = None


@dataclass(frozen=True)
class ChamberResult:
    """What actually happened in one chamber."""

    chamber: str
    dem_seats: int


def brier(probabilities, outcomes) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_score(probabilities, outcomes) -> float:
    """Mean negative log likelihood. Lower is better, and punishes confidence."""
    p = np.clip(np.asarray(probabilities, dtype=float), LOG_CLIP, 1 - LOG_CLIP)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def skill(score: float, baseline: float) -> float:
    """Fractional improvement over a baseline. 0 is no better, 1 is perfect."""
    return float("nan") if baseline == 0 else 1.0 - score / baseline


def reliability(probabilities, outcomes, edges=RELIABILITY_EDGES) -> list[dict]:
    """Did races called at p% come true p% of the time?

    Returns one row per bucket with the count in it, so a bucket holding three
    races cannot be read as evidence of anything.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # The top bucket is closed so a probability of exactly 1 lands in it.
        inside = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if not inside.any():
            continue
        rows.append({
            "lo": lo, "hi": hi, "n": int(inside.sum()),
            "predicted": float(p[inside].mean()),
            "actual": float(y[inside].mean()),
        })
    return rows


def interval_coverage(results, forecast_races, levels=(0.5, 0.9)) -> list[dict]:
    """How often the true margin fell inside each predicted interval.

    A 90% interval should contain the truth 90% of the time. Under-coverage
    means the forecast was too confident; over-coverage means it was too timid,
    which is safer and still wrong.

    Only races whose final margin is known are scored, and the count is
    reported, because early in the count this will be a handful of races.
    """
    # Only the levels the payload actually carries. It publishes p05/p25/p50/
    # p75/p95, so 50% and 90% are available and an 80% interval is not -- asking
    # for one would silently score nothing.
    quantiles = {0.5: ("p25", "p75"), 0.9: ("p05", "p95")}
    rows = []
    for level in levels:
        lo_key, hi_key = quantiles[level]
        hits = total = 0
        for result in results:
            race = forecast_races.get(result.race_id)
            if race is None or result.dem_margin is None:
                continue
            margin = race.get("margin", {})
            if lo_key not in margin or hi_key not in margin:
                continue
            total += 1
            hits += margin[lo_key] <= result.dem_margin <= margin[hi_key]
        if total:
            rows.append({"level": level, "n": total, "covered": hits / total})
    return rows


def score_races(forecast: dict, results) -> dict:
    """Per-race scoring, with the two baselines that make a Brier readable."""
    races = {r["id"]: r for r in forecast.get("races", [])}
    scored = [r for r in results if r.race_id in races]
    if not scored:
        return {"n": 0}

    model = [races[r.race_id]["dem_win_prob"] for r in scored]
    outcomes = [float(r.dem_won) for r in scored]

    # Fundamentals: what the model believed before any poll, at a tied national
    # environment. Beating this is the whole claim of reading polls.
    prior = []
    for r in scored:
        margin = races[r.race_id].get("fundamentals_prior_margin")
        prior.append(0.5 if margin is None else float(margin > 0))

    coin = [0.5] * len(scored)
    out = {
        "n": len(scored),
        "brier": brier(model, outcomes),
        "log_score": log_score(model, outcomes),
        "brier_coin": brier(coin, outcomes),
        "brier_fundamentals": brier(prior, outcomes),
        "reliability": reliability(model, outcomes),
        "coverage": interval_coverage(scored, races),
        "called_right": int(sum((p > 0.5) == bool(y) for p, y in zip(model, outcomes, strict=True))),
    }
    out["skill_vs_coin"] = skill(out["brier"], out["brier_coin"])
    out["skill_vs_fundamentals"] = skill(out["brier"], out["brier_fundamentals"])

    margins = [(races[r.race_id]["margin"]["p50"], r.dem_margin)
               for r in scored if r.dem_margin is not None]
    if margins:
        errors = np.array([abs(pred - actual) for pred, actual in margins])
        out["margin_mae"] = float(errors.mean())
        out["margin_bias"] = float(np.mean([actual - pred for pred, actual in margins]))
        out["margin_n"] = len(margins)
    return out


def score_chamber(forecast: dict, result: ChamberResult) -> dict:
    """Chamber scoring, including where the truth fell in the distribution.

    The percentile is the informative part. A single election cannot show an
    interval was too wide, but a result at the 3rd percentile is a fact about
    the shape of the forecast rather than about which side won.
    """
    control = forecast["chamber_forecast"]
    dem_prob = control["dem_control_prob"]
    dem_won = result.dem_seats >= control["dem_seats_for_majority"]

    distribution = {int(k): v for k, v in control.get("seat_distribution", {}).items()}
    percentile = None
    if distribution:
        at_or_below = sum(v for k, v in distribution.items() if k <= result.dem_seats)
        percentile = float(at_or_below)

    seats = control["dem_seats"]
    return {
        "chamber": result.chamber,
        "dem_seats_actual": result.dem_seats,
        "dem_seats_median": seats["median"],
        "seat_error": result.dem_seats - seats["median"],
        "dem_control_prob": dem_prob,
        "dem_won": dem_won,
        "prob_assigned_to_winner": dem_prob if dem_won else 1.0 - dem_prob,
        "inside_90": seats["p05"] <= result.dem_seats <= seats["p95"],
        "inside_50": seats["p25"] <= result.dem_seats <= seats["p75"],
        "seat_percentile": percentile,
        "brier": brier([dem_prob], [float(dem_won)]),
    }


def score_rivals(rows, result: ChamberResult, dem_won: bool) -> list[dict]:
    """The same Brier score, on the same outcome, for every other forecaster.

    Uses each forecaster's most recent row for the chamber, so a registry that
    accumulates over months scores their final published number rather than
    whichever row happens to be first.
    """
    latest: dict[str, dict] = {}
    for row in rows:
        if row["chamber"] != result.chamber:
            continue
        key = row["forecaster"]
        if key not in latest or row["as_of"] > latest[key]["as_of"]:
            latest[key] = row

    out = []
    for name, row in sorted(latest.items()):
        prob = float(row["dem_control_prob"])
        out.append({
            "forecaster": name,
            "as_of": row["as_of"],
            "dem_control_prob": prob,
            "brier": brier([prob], [float(dem_won)]),
        })
    return out


def simulate_outcome(forecast: dict, seed: int = 0):
    """Draw one plausible election from the forecast, for dry runs.

    This exists so the scoring pipeline can be exercised end to end before there
    is a result to score, and so that what election night will print is visible
    in advance rather than being discovered on the night.

    **It cannot validate the model.** Scoring a forecast against draws from its
    own distribution shows perfect calibration by construction. It validates the
    code, and nothing else.
    """
    rng = np.random.default_rng(seed)
    races, outcomes = forecast.get("races", []), []
    for race in races:
        margin = race["margin"]
        # Spread implied by the published interval, so a safe seat stays safe
        # and a toss-up is a coin flip.
        sd = (margin["p95"] - margin["p05"]) / (2 * 1.6449)
        drawn = rng.normal(margin["p50"], max(sd, 1e-6))
        outcomes.append(RaceResult(race["id"], bool(drawn > 0), round(float(drawn), 2)))

    control = forecast["chamber_forecast"]
    dem_seats = control["seats_not_up"].get("D", 0) + sum(o.dem_won for o in outcomes)
    chamber = ChamberResult(forecast.get("chamber", "senate"), int(dem_seats))
    return outcomes, chamber


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _load_rivals(chamber: str) -> list[dict]:
    import csv

    from . import paths

    registry = paths.REPO_ROOT / "data" / "rivals" / "forecasts.csv"
    if not registry.exists():
        return []
    with registry.open(encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["chamber"] == chamber]


def run_score(chamber: str = "senate", dry_run: int | None = None) -> int:
    """Score the published forecast against the result, if there is one yet.

    Returns 0 whether or not a result exists. An election that has not happened
    is the normal state of the world for most of this command's life, and a
    scheduled job that fails every day until November is a job whose failures
    stop being read.
    """
    import json

    from . import paths
    from .data import results as results_source

    name = "forecast.json" if chamber == "senate" else "forecast_house.json"
    path = paths.SITE_DATA_DIR / name
    if not path.exists():
        print(f"No published {chamber} forecast at {path}.")
        return 0
    forecast = json.loads(path.read_text(encoding="utf-8"))

    if dry_run is not None:
        race_results, chamber_result = simulate_outcome(forecast, seed=dry_run)
        provenance = (
            f"DRY RUN (seed {dry_run}) -- one election drawn from this forecast's own\n"
            "  distribution. This exercises the scoring code end to end and shows what\n"
            "  election night will print. It cannot validate the model: scoring a\n"
            "  forecast against its own draws is calibrated by construction."
        )
    else:
        seats = results_source.fetch_chamber_seats(chamber)
        if seats is None:
            print(f"No {chamber} result published yet. Nothing to score.")
            return 0
        # Independents are reported separately and never folded in; see
        # data/results.py. Scoring uses the party count, and the caucus
        # alternative is printed beside it.
        race_results = []
        chamber_result = ChamberResult(chamber, seats.dem)
        provenance = (
            f"  Result: D {seats.dem}, R {seats.rep}, I {seats.ind} "
            f"(via infobox `{seats.field}`)\n  Source: {seats.source}"
        )

    print("=" * 74)
    print(f"  SCORING THE {chamber.upper()} FORECAST OF {forecast['run_date']}")
    print("=" * 74)
    print(provenance)
    print()

    chamber_score = score_chamber(forecast, chamber_result)
    print("  Chamber")
    print(f"    actual {chamber_score['dem_seats_actual']} seats against a median of "
          f"{chamber_score['dem_seats_median']} "
          f"({chamber_score['seat_error']:+d})")
    print(f"    inside the 90% interval: {chamber_score['inside_90']}   "
          f"inside the 50%: {chamber_score['inside_50']}")
    if chamber_score["seat_percentile"] is not None:
        print(f"    the result sits at the {chamber_score['seat_percentile']:.1%} "
              f"point of the predicted distribution")
    print(f"    probability given to the side that won: "
          f"{chamber_score['prob_assigned_to_winner']:.1%}   "
          f"Brier {chamber_score['brier']:.4f}")

    rivals = score_rivals(_load_rivals(chamber), chamber_result, chamber_score["dem_won"])
    if rivals:
        print()
        print("  Against the other forecasters (their last published number):")
        for row in sorted(rivals, key=lambda r: r["brier"]):
            print(f"    {row['forecaster']:<16}{row['as_of']}  "
                  f"P(D)={row['dem_control_prob']:.1%}  Brier {row['brier']:.4f}")
        print("    Lower is better. One election barely separates these -- a confident")
        print("    forecast wins whenever it is right, which is what confident means.")

    race_score = score_races(forecast, race_results)
    if race_score["n"]:
        print()
        print(f"  Races ({race_score['n']} scored, "
              f"{race_score['called_right']} called correctly)")
        print(f"    Brier {race_score['brier']:.4f}   "
              f"vs coin {race_score['brier_coin']:.4f} "
              f"(skill {race_score['skill_vs_coin']:+.1%})   "
              f"vs fundamentals {race_score['brier_fundamentals']:.4f} "
              f"(skill {race_score['skill_vs_fundamentals']:+.1%})")
        print(f"    log score {race_score['log_score']:.4f}")
        if "margin_mae" in race_score:
            print(f"    margin error {race_score['margin_mae']:.2f} pts mean absolute, "
                  f"bias {race_score['margin_bias']:+.2f}")
        for row in race_score["coverage"]:
            print(f"    {row['level']:.0%} interval covered "
                  f"{row['covered']:.1%} of {row['n']} races")
        print()
        print("    Reliability (predicted vs actual, by bucket):")
        for row in race_score["reliability"]:
            print(f"      {row['lo']:.0%}-{row['hi']:.0%}  n={row['n']:<4} "
                  f"predicted {row['predicted']:.1%}  actual {row['actual']:.1%}")
    else:
        print()
        print("  No per-race results available, so only the chamber is scored.")
        print("  Per-race scoring is where the evidence is; see score.py.")

    print("=" * 74)
    return 0
