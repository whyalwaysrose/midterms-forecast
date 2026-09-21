"""Score each day's new polls against the forecast that had not seen them.

METHODOLOGY 8 asks whether the model predicts polls it was never shown, and
says plainly what the next step is: *accumulate held-out polls across daily runs
and revisit ``design_effect`` and ``nu`` once the sample supports it.* That never
happened. ``backtest.py`` answers the question by refitting on a holdout window,
which costs a second full sample, so it was run once by hand in August on 33
polls and not since. Twenty-six days of evidence went by uncollected.

The free version is this. Every day brings polls that yesterday's fit had never
seen, and yesterday's forecast is archived under ``outputs/runs/``. Scoring
today's new polls against yesterday's posterior is genuinely out of sample,
costs no sampling at all, and grows by a few polls a day without anyone
remembering to do anything.

HOW A POLL IS PREDICTED
-----------------------
The archived payload carries the latent trajectory -- the poll-based state, with
no election-day error in it, which is right because a poll measures opinion now
rather than the result in November. Everything standing between that state and a
number a pollster publishes comes from the ``predictive`` block of the same
payload: the house effect for that pollster, the population effect for likely
versus registered voters, the partisan-sponsor lean, and the excess noise a
larger sample does not buy away.

    centre = latent + house + population + partisan x sign
    spread = latent^2 + house^2 + sampling(n, design effect) + excess^2

then Student-t with the archived run's own ``nu``, including its variance
correction. Using the *archived* likelihood settings rather than today's config
matters: both ``design_effect`` and ``nu`` have been retuned during this cycle,
and reading the current values would grade an old forecast against a likelihood
it never had.

WHAT THE NUMBERS MEAN
---------------------
The PIT value is where the observed poll fell in that predictive distribution.
For a well-calibrated model these are uniform on [0, 1]. Piled up in the middle
means the intervals are too wide; piled at the ends means too narrow. Interval
coverage says the same thing more bluntly, and is what the config parameters
would actually be retuned against.

WHAT THIS IS NOT
----------------
Not a test of whether the forecast is right about the election. Polls are the
thing being predicted here, and polls can be collectively biased in a way this
can never detect -- that is what ``election_day_error`` represents and only a
result can check.

Nor is it fully independent day to day. A pollster that publishes ten polls in a
week contributes ten rows that share a house effect, and consecutive days of the
same race share most of a latent state. So the effective sample is smaller than
the row count, and the row count is reported next to every figure to keep that
visible.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from . import paths
from .config import load_all
from .data.polls import NATIONAL_RACE_ID, build_poll_table
from .data.roster import Roster
from .data.votehub import latest_snapshot, load_snapshot
from .model.design import POOLED_POLLSTER, _sampling_variance

log = logging.getLogger(__name__)

CREDIBLE_LEVELS = (0.5, 0.8, 0.9)

#: Half-width of a 90% interval in standard deviations, for turning the
#: published p05/p95 band back into a spread.
Z90 = 1.6449

FILENAME = {"senate": "forecast.json", "house": "forecast_house.json"}


def _record_path(chamber: str) -> Path:
    return paths.REPO_ROOT / "data" / "calibration" / f"forward_{chamber}.csv"


FIELDS = ("scored_on", "forecast_run", "chamber", "race_id", "poll_id",
          "pollster", "field_date", "observed_logit", "predicted_logit",
          "predictive_sd", "pit")


@dataclass(frozen=True)
class ScoredPoll:
    race_id: str
    poll_id: str
    pollster: str
    field_date: date
    observed: float
    centre: float
    sd: float
    pit: float


def margin_to_logit(margin_pts: float) -> float:
    """Undo ``logit_to_margin``. The trajectory is published in points."""
    share = (margin_pts / 100.0 + 1.0) / 2.0
    share = min(max(share, 1e-6), 1 - 1e-6)
    return math.log(share / (1.0 - share))


def _latent_at(trajectory: list[dict], when: date) -> tuple[float, float] | None:
    """Latent centre and spread on a given day, in logits.

    The published trajectory is thinned to roughly weekly, so a poll fielded
    between two points is interpolated. That is safe here and would not be for a
    jumpier quantity: the latent state is a smooth random walk by construction.
    """
    if not trajectory:
        return None
    points = [(date.fromisoformat(p["date"]), p) for p in trajectory]
    points.sort(key=lambda item: item[0])

    if when <= points[0][0]:
        chosen = points[0][1]
    elif when >= points[-1][0]:
        chosen = points[-1][1]
    else:
        after = next(i for i, (d, _) in enumerate(points) if d >= when)
        before_date, before = points[after - 1]
        after_date, after_point = points[after]
        span = (after_date - before_date).days or 1
        weight = (when - before_date).days / span
        chosen = {
            key: before[key] * (1 - weight) + after_point[key] * weight
            for key in ("p05", "p50", "p95")
        }

    centre = margin_to_logit(chosen["p50"])
    spread = (margin_to_logit(chosen["p95"]) - margin_to_logit(chosen["p05"])) / (2 * Z90)
    return centre, max(spread, 1e-6)


def _seen_poll_ids(payload: dict) -> set[str]:
    seen: set[str] = set()
    for race in payload.get("races", []):
        seen.update(race.get("all_poll_ids", []))
    seen.update(payload.get("predictive", {}).get("national_poll_ids", []))
    return seen


def previous_run(chamber: str, before: date) -> dict | None:
    """The most recent archived run from before ``before``, if any."""
    import json

    runs = paths.OUTPUTS_DIR / "runs"
    candidates = []
    for directory in runs.glob("*/"):
        payload = directory / FILENAME[chamber]
        if not payload.exists():
            continue
        try:
            when = date.fromisoformat(directory.name)
        except ValueError:
            continue
        if when < before:
            candidates.append((when, payload))
    if not candidates:
        return None
    _, path = max(candidates, key=lambda item: item[0])
    return json.loads(path.read_text(encoding="utf-8"))


def score_against(payload: dict, polls) -> list[ScoredPoll]:
    """Score polls the archived run had never seen."""
    predictive = payload.get("predictive")
    if not predictive:
        # Two very different situations share this branch, and only one of them
        # is fine. An archive from before the block existed has no latent tail
        # either -- that is history, and refusing it is correct. An archive that
        # HAS the latent tail but no block is a bug: the two were introduced in
        # the same change, so one without the other means the block failed to
        # assemble and was swallowed by its own safety wrapper.
        #
        # That is not hypothetical. From 2026-09-16 every run did exactly this,
        # the wrapper caught it as designed, every workflow reported success, and
        # five days of calibration evidence went uncollected before anyone
        # looked. So the second case now raises a GitHub error annotation: it
        # still cannot stop a forecast, but it can no longer be quiet.
        has_tail = any(r.get("latent") for r in payload.get("races", []))
        if has_tail:
            message = (
                f"archived run {payload.get('run_date')} has a latent tail but no "
                "predictive block -- the block failed to assemble in that run. "
                "Check its log for 'could not assemble the predictive block'."
            )
            log.error(message)
            print(f"::error title=Forward calibration broken::{message}")
        else:
            log.warning("archived run %s predates the predictive block; "
                        "cannot score it", payload.get("run_date"))
        return []

    # The archived payload carries a short `latent` tail; the live one carries
    # the full `trajectory`. Accept either, so this works against an archive and
    # against site/data alike.
    def series(block: dict) -> list:
        return block.get("latent") or block.get("trajectory") or []

    trajectories = {r["id"]: series(r) for r in payload.get("races", [])}
    trajectories[NATIONAL_RACE_ID] = series(payload.get("national", {}))

    seen = _seen_poll_ids(payload)
    house = predictive["house_effect"]
    population = predictive["population_effect"]
    partisan = predictive["partisan_effect"]["mean"]
    excess = predictive["sigma_excess"]["mean"]
    nu = predictive["student_t_nu"]
    design_effect = predictive["design_effect"]

    from scipy import stats

    out: list[ScoredPoll] = []
    for poll in polls:
        if poll.poll_id in seen:
            continue
        latent = _latent_at(trajectories.get(poll.race_id, []), poll.field_date)
        if latent is None:
            continue
        latent_centre, latent_sd = latent

        # A pollster the archived run never saw has no house effect of its own
        # and must not borrow one. The pooled bucket is what the model itself
        # uses for exactly this case.
        entry = house.get(poll.pollster) or house.get(POOLED_POLLSTER)
        house_mean, house_sd = (entry["mean"], entry["sd"]) if entry else (0.0, 0.0)

        pop = population.get(poll.population, {"mean": 0.0, "sd": 0.0})
        sampling_var = _sampling_variance(poll, design_effect)

        centre = latent_centre + house_mean + pop["mean"] + partisan * poll.partisan_sign
        variance = latent_sd**2 + house_sd**2 + pop["sd"] ** 2 + sampling_var + excess**2
        sd = math.sqrt(variance)

        scale = sd
        if predictive.get("match_student_t_variance"):
            scale = sd / math.sqrt(nu / (nu - 2.0))

        observed = math.log(
            min(max(poll.two_party_dem, 1e-6), 1 - 1e-6)
            / (1 - min(max(poll.two_party_dem, 1e-6), 1 - 1e-6))
        )
        pit = float(stats.t.cdf((observed - centre) / scale, df=nu))
        out.append(ScoredPoll(poll.race_id, poll.poll_id, poll.pollster,
                              poll.field_date, observed, centre, sd, pit))
    return out


def append(chamber: str, scored: list[ScoredPoll], run_date: str, today: date) -> int:
    """Add rows, skipping any poll already recorded. Returns rows written."""
    path = _record_path(chamber)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            existing = {row["poll_id"] for row in csv.DictReader(fh)}

    fresh = [s for s in scored if s.poll_id not in existing]
    if not fresh:
        return 0

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for s in fresh:
            writer.writerow({
                "scored_on": today.isoformat(), "forecast_run": run_date,
                "chamber": chamber, "race_id": s.race_id, "poll_id": s.poll_id,
                "pollster": s.pollster, "field_date": s.field_date.isoformat(),
                "observed_logit": round(s.observed, 5),
                "predicted_logit": round(s.centre, 5),
                "predictive_sd": round(s.sd, 5), "pit": round(s.pit, 5),
            })
    return len(fresh)


def accumulated(chamber: str) -> dict:
    """Everything recorded so far, summarised."""
    path = _record_path(chamber)
    if not path.exists():
        return {"n": 0}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"n": 0}

    pit = np.array([float(r["pit"]) for r in rows])
    coverage = []
    for level in CREDIBLE_LEVELS:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        coverage.append({"level": level,
                         "covered": float(np.mean((pit >= lo) & (pit <= hi)))})

    # Kolmogorov-Smirnov against uniform: how far the PIT distribution is from
    # what a calibrated model would produce.
    ordered = np.sort(pit)
    n = len(ordered)
    grid = (np.arange(1, n + 1)) / n
    ks = float(np.max(np.abs(ordered - grid)))

    return {
        "n": n,
        "polls": len({r["poll_id"] for r in rows}),
        "races": len({r["race_id"] for r in rows}),
        "pollsters": len({r["pollster"] for r in rows}),
        "days": len({r["scored_on"] for r in rows}),
        "first": min(r["scored_on"] for r in rows),
        "last": max(r["scored_on"] for r in rows),
        "coverage": coverage,
        "ks": ks,
        "pit_mean": float(pit.mean()),
    }


def run_forward(chamber: str = "senate", today: date | None = None) -> int:
    """Score today's new polls against the last archived run, and accumulate.

    Returns 0 even when there is nothing to score. A daily step that fails on
    the ordinary case -- a quiet day with no new polls -- is a step whose
    failures stop being read.
    """
    today = today or date.today()
    payload = previous_run(chamber, before=today)
    if payload is None:
        print(f"No archived {chamber} run before {today}; nothing to score against.")
        return 0

    races, cfg = load_all(chamber=chamber)
    snapshot = latest_snapshot()
    if snapshot is None:
        print("No poll snapshot available.")
        return 0
    raw = load_snapshot(snapshot)
    table = build_poll_table(raw, races, cfg, Roster.load(), as_of=today)

    scored = score_against(payload, table.polls)
    written = append(chamber, scored, payload["run_date"], today)

    print("=" * 70)
    print(f"  FORWARD CALIBRATION -- {chamber.upper()}")
    print("=" * 70)
    print(f"  Scored against the run of {payload['run_date']}, which had not seen "
          f"{len(scored)} of today's polls")
    print(f"  New rows recorded: {written}")

    stats = accumulated(chamber)
    if stats["n"]:
        print()
        print(f"  Accumulated: {stats['n']} polls across {stats['races']} races "
              f"and {stats['pollsters']} pollsters, over {stats['days']} days "
              f"({stats['first']} to {stats['last']})")
        for row in stats["coverage"]:
            target = row["level"]
            got = row["covered"]
            verdict = ("about right" if abs(got - target) < 0.05
                       else "too wide" if got > target else "TOO NARROW")
            print(f"    {target:.0%} interval contained {got:.1%}   {verdict}")
        print(f"    PIT mean {stats['pit_mean']:.3f} (0.5 is centred), "
              f"KS {stats['ks']:.3f}")
        print()
        print("  Polls from one pollster share a house effect and consecutive days")
        print("  share a latent state, so the effective sample is smaller than the")
        print("  row count. Treat this as a trend, not a p-value.")
    print("=" * 70)
    return 0
