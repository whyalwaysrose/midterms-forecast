"""Does the data reaching the model look like the data that should reach it?

Every other check in this project tests the model *given its inputs*: the
backtest, the forward calibration, the convergence report, the unit tests. None
of them tests whether the inputs arrived. That asymmetry is not theoretical --
it is where this project's real defects have lived, and all of them shared one
signature: **the pipeline produced a plausible artefact that was quietly wrong,
and "fewer records" is indistinguishable from "legitimately fewer records".**

The record, each of which ran green:

* the House candidate roster was generated once at 40 polled districts and
  never rebuilt. By October the feed had 66 and the forecast used 38. Seven
  weeks, every workflow successful, the audit step printing the missing names
  into a log nobody reads;
* `midterms fetch` requested three poll types and omitted `us-representative`,
  so the one command whose purpose is to populate the snapshot could not
  populate it for the House. It exited 0;
* `forward.py` built its poll table with `Roster.load()` and no poll type, both
  of which default to the Senate, so three weeks of House calibration scored
  only generic-ballot polls. Four rows, all `__national__`;
* a `{{party stripe}}` capitalisation dropped 27 districts from the 2024 result
  parse while 2022 validated perfectly;
* OR-03's presidential lean was 30 points wrong, live in a published forecast,
  found only by comparing two independent sources.

So these checks are written backwards from that list: each one is an invariant
that a real failure violated. A check nobody can point at a past bug for is a
check that will be tuned away the first time it is inconvenient.

Thresholds are deliberately loose. The purpose is to catch a pipeline that has
stopped working, not to police ordinary variation -- a noisy check gets
disabled, and a disabled check is worse than none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from . import paths

#: Districts can legitimately be unusable: California's top-two produces
#: D-vs-D and R-vs-R general elections, and a member-vs-member race after
#: redistricting is a real contest the two-party model cannot read. Those are
#: correct refusals. What is not legitimate is most of the map being unusable
#: because the roster cannot resolve names.
#:
#: Calibrated against both observed states rather than picked: with the stale
#: roster 27 of 65 districts with poll data were unusable (41%); with it rebuilt
#: the figure is 5 of 66 (8%), and those five are genuine same-party matchups.
#: 20% sits clear of both.
MAX_BLOCKED_FRACTION = 0.20

#: A chamber losing a quarter of its polled races overnight is not polls ageing
#: out -- `max_age_days` is 400, so nothing ages out before this election. It is
#: a pipeline fault.
MAX_COVERAGE_DROP = 0.25

#: Beyond this the roster is stale enough that newly polled districts are being
#: silently rejected. Matches the workflow's own freshness job.
MAX_ROSTER_AGE_DAYS = 7


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    #: What this invariant is standing guard over, named so a future reader can
    #: decide whether it still earns its place.
    guards: str
    value: Any = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "ok": self.ok, "detail": self.detail,
            "guards": self.guards, "value": self.value,
        }


@dataclass
class Report:
    chamber: str
    run_date: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "chamber": self.chamber,
            "run_date": self.run_date,
            "ok": self.ok,
            "metrics": self.metrics,
            "checks": [c.as_dict() for c in self.checks],
        }


def report_path(chamber: str) -> Path:
    return paths.OUTPUTS_DIR / "health" / f"{chamber}.json"


def previous_report(chamber: str) -> dict | None:
    path = report_path(chamber)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _roster_built(chamber: str) -> date | None:
    """The build stamp the House roster generator writes into its header."""
    from .data.roster import ROSTER_FILE

    path = paths.CONFIG_DIR / ROSTER_FILE[chamber]
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if line.startswith("# built:"):
            try:
                return date.fromisoformat(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def build(chamber: str, raw: dict, table, races, today: date | None = None) -> Report:
    """Assemble the report for one chamber from the data the run actually used.

    Takes the already-built poll table rather than rebuilding one, so what is
    checked is what the forecast consumed. A health check that fetches its own
    copy of the data is checking a different pipeline from the one that ran.
    """
    from .data.polls import ALL_POLL_TYPES, NATIONAL_POLL_TYPES, RACE_POLL_TYPE

    today = today or date.today()
    poll_type = RACE_POLL_TYPE[chamber]
    report = Report(chamber=chamber, run_date=today.isoformat())

    # --- the feeds this run needed arrived --------------------------------
    #
    # This chamber's race poll type plus the shared national ones -- which is
    # exactly what `cmd_run` requests. NOT every poll type: `fetch_and_snapshot`
    # merges into the snapshot on disk but *returns only what was asked for*, so
    # a Senate run's `raw` legitimately has no `us-representative`.
    #
    # The first version of this check asserted all four and failed three nights
    # running, each chamber reporting the other's feed missing while its own
    # coverage was perfect. It was written against `load_snapshot(...)`, which
    # holds everything, instead of against the dict `cmd_run` actually passes --
    # a check for "the inputs are not what you think" that was itself wrong
    # about its input. Whole-snapshot completeness is `cmd_fetch`'s job and it
    # already fails there; it cannot be this one's, because the Senate runs
    # before the House and the snapshot is genuinely half-built in between.
    required = (poll_type, *NATIONAL_POLL_TYPES)
    missing = [t for t in required if not raw.get(t)]
    report.checks.append(Check(
        name="required_feeds_present",
        ok=not missing,
        detail=(f"all of {', '.join(required)} present" if not missing
                else f"this run received no {', '.join(missing)}"),
        guards="a run proceeding on a feed that returned nothing, which reads "
               "downstream as a quiet day rather than a failed fetch",
        value={t: len(raw.get(t, ())) for t in ALL_POLL_TYPES},
    ))

    # --- this chamber's own polls reached the model -----------------------
    race_polls = table.race_polls
    report.checks.append(Check(
        name="race_polls_used",
        ok=bool(race_polls),
        detail=f"{len(race_polls)} race polls from {poll_type}",
        guards="a chamber-blind default feeding the Senate's polls to the House",
        value=len(race_polls),
    ))

    # --- the roster can still read the feed -------------------------------
    #
    # The headline number. A district whose every poll is rejected because no
    # candidate name resolves is a district the model is not using and could be.
    # A district rejected with its names resolved is a genuine same-party race.
    # Only the first kind is a fault, and separating them is the whole check.
    used = {p.race_id for p in race_polls}
    blocked = set(table.unknown_candidates) - used
    with_data = used | blocked
    fraction = len(blocked) / len(with_data) if with_data else 0.0
    report.checks.append(Check(
        name="roster_reads_the_feed",
        ok=fraction <= MAX_BLOCKED_FRACTION,
        detail=(
            f"{len(blocked)} of {len(with_data)} races with poll data are unusable "
            f"because no candidate name resolves ({fraction:.0%}; "
            f"limit {MAX_BLOCKED_FRACTION:.0%})"
        ),
        guards="the House roster going stale for seven weeks at 41% blocked",
        value={"blocked": len(blocked), "with_data": len(with_data),
               "fraction": round(fraction, 4),
               "races": sorted(blocked)},
    ))

    # --- coverage did not collapse overnight ------------------------------
    previous = previous_report(chamber)
    before = (previous or {}).get("metrics", {}).get("races_with_polls")
    if before:
        drop = (before - len(used)) / before
        report.checks.append(Check(
            name="coverage_did_not_collapse",
            ok=drop <= MAX_COVERAGE_DROP,
            detail=(f"races with polls {before} -> {len(used)} "
                    f"({-drop:+.0%}; limit {-MAX_COVERAGE_DROP:.0%})"),
            guards="any new fault that silently drops races, the shared signature "
                   "of every defect in this file's docstring",
            value={"before": before, "after": len(used)},
        ))

    # --- the roster is current --------------------------------------------
    built = _roster_built(chamber)
    if built is not None:
        age = (today - built).days
        report.checks.append(Check(
            name="roster_is_fresh",
            ok=age <= MAX_ROSTER_AGE_DAYS,
            detail=f"roster built {built} ({age} days ago; "
                   f"limit {MAX_ROSTER_AGE_DAYS})",
            guards="the rebuild step failing quietly and the roster re-staling",
            value={"built": built.isoformat(), "age_days": age},
        ))

    # --- every race has a usable lean -------------------------------------
    #
    # OR-03's presidential lean was 30 points wrong and live for weeks. A range
    # check cannot catch 30 points of error inside a plausible band, and this
    # does not pretend to; it catches the absent and the absurd, which is the
    # part that can be caught without a second source.
    bad_lean = [
        race.id for race in races.races
        if not (0.0 < race.pres_2024_dem_two_party < 1.0)
        or not (0.0 < race.pres_2020_dem_two_party < 1.0)
    ]
    report.checks.append(Check(
        name="every_race_has_a_lean",
        ok=not bad_lean,
        detail=("all leans present and in range" if not bad_lean
                else f"{len(bad_lean)} races with an unusable lean: {bad_lean[:5]}"),
        guards="a district with no measured lean reaching the model as a 0 or a 1",
        value=len(bad_lean),
    ))

    report.metrics = {
        "races_total": len(races.races),
        "races_with_polls": len(used),
        "races_blocked_by_roster": len(blocked),
        "race_polls": len(race_polls),
        "national_polls": len(table.national),
        "feed_records": {t: len(raw.get(t, ())) for t in ALL_POLL_TYPES},
        "rejections": dict(table.rejections),
        "unresolved_name_races": len(table.unknown_candidates),
    }
    return report


def write(report: Report) -> Path:
    path = report_path(report.chamber)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=1), encoding="utf-8")
    return path


def render(report: Report) -> str:
    lines = [
        "=" * 70,
        f"  PIPELINE HEALTH -- {report.chamber.upper()}  ({report.run_date})",
        "=" * 70,
    ]
    for check in report.checks:
        lines.append(f"  {'PASS' if check.ok else 'FAIL'}  {check.name}")
        lines.append(f"        {check.detail}")
        if not check.ok:
            lines.append(f"        guards against: {check.guards}")
    m = report.metrics
    lines.append("")
    lines.append(f"  {m['races_with_polls']} of {m['races_total']} races have polls; "
                 f"{m['race_polls']} race polls, {m['national_polls']} national")
    lines.append("=" * 70)
    return "\n".join(lines)
