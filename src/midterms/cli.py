"""Command-line entry points.

    midterms run              fetch polls, fit, simulate, write outputs
    midterms run --offline    same, but reuse the latest raw snapshot
    midterms fetch            fetch and snapshot polls only
    midterms audit-roster     report candidate names the roster cannot classify
    midterms audit-pollsters  report pollster names that look like duplicates
    midterms check-keys       report which optional API keys are configured
    midterms fetch-markets    snapshot prediction-market odds for the dashboard
    midterms backtest         calibration check against held-out polls

``run`` is what CI invokes daily.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import date

# PyTensor prints a loud warning when no C++ compiler is present. We sample via
# nutpie (Numba), so the C backend is genuinely not needed; silence it before
# PyTensor is imported anywhere.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

log = logging.getLogger("midterms")

#: Candidate roster per chamber. The Senate's is hand-written and reviewed; the
#: House's is generated, and covers only the districts that have been polled --
#: names matter for reading a poll, and an unpolled district has none to read.
ROSTER_FILE = {
    "senate": "candidates_senate_2026.yaml",
    "house": "candidates_house_2026.yaml",
}


def _use_utf8_output() -> None:
    """Stop a Windows console killing the run at the last step.

    The commentary is generated prose and contains real punctuation -- arrows,
    en dashes, non-ASCII candidate names like Ben Ray Lujan. A Windows console
    defaults to cp1252, so printing it raised UnicodeEncodeError *after* the
    model had sampled and written the forecast, losing the changelog and
    exiting through a traceback. CI never saw it because the runners are Linux.

    Reconfiguring rather than sanitising the text: the strings are correct, it
    is only this one console that cannot render them, and replacement affects
    nothing that is written to a file.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _configure_logging(verbose: bool) -> None:
    _use_utf8_output()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("arviz").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data.votehub import VoteHubClient

    client = VoteHubClient()
    raw = client.fetch_and_snapshot(["us-senator", "generic-ballot", "approval"])
    for poll_type, records in raw.items():
        log.info("%s: %d polls", poll_type, len(records))
    return 0


def cmd_audit_roster(args: argparse.Namespace) -> int:
    """Report poll answers the roster cannot classify.

    Run this after a primary: new nominees appear in the feed as unrecognised
    names, their polls are skipped, and the race silently reverts to its
    fundamentals prior until the roster is updated.
    """
    from . import paths
    from .config import load_all
    from .data.polls import RACE_POLL_TYPE
    from .data.roster import Roster
    from .data.votehub import (
        VoteHubClient,
        is_primary_subject,
        latest_snapshot,
        load_snapshot,
    )

    chamber = args.chamber
    poll_type = RACE_POLL_TYPE[chamber]
    races, _cfg = load_all(chamber=chamber)
    roster = Roster.load(paths.CONFIG_DIR / ROSTER_FILE[chamber])

    snapshot = latest_snapshot()
    if snapshot is None or not args.offline:
        raw = VoteHubClient().fetch_and_snapshot([poll_type, "generic-ballot", "approval"])
    else:
        raw = load_snapshot(snapshot)

    subject_to_race = {r.subject_for(races.cycle): r.id for r in races.races}
    unknown: dict[str, set[str]] = {}

    for record in raw.get(poll_type, []):
        subject = str(record.get("subject", ""))
        if is_primary_subject(subject):
            continue
        race_id = subject_to_race.get(subject)
        if race_id is None:
            unknown.setdefault(f"<unmapped subject: {subject}>", set())
            continue
        names = [str(a.get("choice", "")) for a in record.get("answers") or []]
        missing = roster.unknown_names(race_id, names)
        if missing:
            unknown.setdefault(race_id, set()).update(missing)

    if not unknown:
        print("Roster is complete: every candidate in the current feed is classified.")
        return 0

    print("Unclassified candidate names (their polls are being skipped):\n")
    for race_id in sorted(unknown):
        for name in sorted(unknown[race_id]):
            print(f"  {race_id}: {name}")
    print(
        f"\nAdd each to config/{ROSTER_FILE[chamber]} under the correct party, "
        "or under `other` for minor-party candidates."
    )
    if chamber == "house":
        print(
            "\nFor the House, an unclassified name is often benign. Pollsters "
            "test candidates months before they file with the FEC, and a "
            "primary field of four Democrats is not a general-election matchup: "
            "the `no single D-vs-R matchup` guard rejects those on its own, "
            "which is the correct outcome rather than a gap to be filled."
        )
    return 1


def cmd_audit_pollsters(args: argparse.Namespace) -> int:
    """Propose pollster names that look like one firm filing two ways.

    Case and punctuation variants merge on their own. Anything beyond that is a
    judgement -- "Marist University" and "Marist College" are the same institute,
    but a rule general enough to know that would also merge firms that merely
    share a word. So this proposes and a person decides, exactly as
    ``audit-roster`` does for candidates.
    """
    from .config import load_all
    from .data.polls import build_poll_table
    from .data.pollsters import ALIASES, find_probable_duplicates
    from .data.roster import Roster
    from .data.votehub import VoteHubClient, latest_snapshot, load_snapshot

    races, cfg = load_all()
    snapshot = latest_snapshot()
    if snapshot is None or not args.offline:
        raw = VoteHubClient().fetch_and_snapshot(["us-senator", "generic-ballot", "approval"])
    else:
        raw = load_snapshot(snapshot)

    table = build_poll_table(raw, races, cfg, Roster.load())
    counts = Counter(p.pollster for p in table.polls)
    proposals = find_probable_duplicates(counts)

    print(f"{len(counts)} distinct pollsters after merging; "
          f"{len(ALIASES)} aliases in force.")
    if not proposals:
        print("No further names look like duplicates.")
        return 0

    print("\nThese look like one pollster filing under several names:\n")
    for variants in proposals.values():
        for name in variants:
            print(f"  {counts.get(name, 0):3d}  {name}")
        print()
    print(
        "If they are the same firm, add an entry to ALIASES in "
        "src/midterms/data/pollsters.py mapping each variant's normalised form "
        "to the canonical name. If they are genuinely different, leave them."
    )
    return 1


def cmd_check_keys(args: argparse.Namespace) -> int:
    """Report which optional API keys are configured and whether they work."""
    from .keys import status

    results = status(check_network=not args.offline)
    print("Optional API keys (the forecast runs without both):\n")
    for entry in results:
        if not entry.present:
            mark = "  -  "
        elif entry.working is None:
            mark = "  ?  "
        else:
            mark = "  OK " if entry.working else " FAIL"
        print(f"{mark} {entry.name:16s} {entry.detail}")

    missing = [e.name for e in results if not e.present]
    broken = [e.name for e in results if e.present and e.working is False]
    if missing:
        print(
            "\nTo add a key, put it in .env in the repo root (gitignored):\n"
            + "\n".join(f"    {name}=your-key-here" for name in missing)
        )
    if broken:
        print("\nA key is set but not working; check for stray whitespace or quotes.")
        return 1
    return 0


def cmd_fetch_markets(args: argparse.Namespace) -> int:
    """Snapshot the prediction-market odds shown alongside the forecast.

    Separate from `run` on purpose. Polymarket is blocked in some jurisdictions
    -- France's regulator null-routes every polymarket.com domain -- so this is
    run where it is reachable and the result committed, and readers get the
    numbers from our own origin. It also means a market outage can never stop a
    forecast from publishing.
    """
    from .data import markets

    events = markets.fetch()
    if not events:
        log.error("no markets fetched; leaving the previous snapshot in place")
        return 1

    # Time series for the outcomes worth charting. Separate from the snapshot
    # fetch because it is one request per outcome and a failure here should
    # still leave today's prices usable.
    markets.attach_history(events)

    path = markets.write_snapshot(events)
    for event in events.values():
        print(f"\n{event.title}  (${event.volume:,.0f} traded)")
        for outcome in event.outcomes:
            if outcome.probability >= 0.005:
                print(f"   {outcome.label[:52]:52s} {outcome.probability * 100:5.1f}%")
    print(f"\nSnapshot -> {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from . import fundamentals as F
    from . import outputs, paths
    from .commentary import generate as generate_commentary
    from .commentary import previous_payload, write_commentary
    from .config import load_all
    from .data import Roster, build_poll_table
    from .data.polls import RACE_POLL_TYPE
    from .data.votehub import VoteHubClient, latest_snapshot, load_snapshot
    from .model.design import build_model_data
    from .model.hierarchical import build_model, convergence_report, sample
    from .model.simulate import simulate_chamber

    paths.ensure_dirs()
    run_date = date.fromisoformat(args.date) if args.date else date.today()
    chamber = args.chamber

    # Cheap and deterministic, so just regenerate them: this keeps both pictures
    # in step with the config if a race is ever added, a district's lean is
    # recomputed, or a projection parameter changes.
    from . import cartogram
    from .geo import write_state_paths

    write_state_paths()

    races, cfg = load_all(chamber=chamber)
    roster = Roster.load(paths.CONFIG_DIR / ROSTER_FILE[chamber])
    log.info("loaded %d races for %s %d", len(races.races), races.chamber, races.cycle)

    if chamber == "house":
        cartogram.write(
            paths.SITE_DATA_DIR / "us-districts.json", races, paths.SITE_DATA_DIR
        )

    # --- data ------------------------------------------------------------
    poll_type = RACE_POLL_TYPE[chamber]
    if args.offline:
        snapshot = latest_snapshot()
        if snapshot is None:
            log.error("--offline requested but no raw snapshot exists; run `midterms fetch`")
            return 2
        log.info("using cached snapshot %s", snapshot.name)
        raw = load_snapshot(snapshot)
        # Say what is actually wrong. Older snapshots pre-date the merge fix in
        # `fetch_and_snapshot` and may hold only the chamber that wrote them
        # last, and the symptom without this is "no usable race polls" a hundred
        # lines later -- which reads as a data problem rather than a missing
        # poll type.
        if not raw.get(poll_type):
            log.error(
                "%s holds no %s polls (it has: %s). It was written before both "
                "chambers shared a snapshot, or by a run of the other chamber. "
                "Re-run without --offline to fetch them.",
                snapshot.name, poll_type, ", ".join(sorted(raw)) or "nothing",
            )
            return 2
    else:
        raw = VoteHubClient().fetch_and_snapshot(
            [poll_type, "generic-ballot", "approval"], run_date
        )

    table = build_poll_table(
        raw, races, cfg, roster,
        as_of=run_date, race_poll_type=poll_type,
    )
    if table.unknown_candidates:
        log.warning(
            "unclassified candidates in %d races; run "
            "`midterms audit-roster --chamber %s`",
            len(table.unknown_candidates), chamber,
        )
    if not table.race_polls and chamber == "senate":
        log.error("no usable race polls; aborting")
        return 3
    if not table.race_polls:
        # Not fatal for the House. Nine districts in ten are never polled at
        # all, so a cycle with no district polling anywhere is a thin forecast
        # rather than a broken one -- the generic ballot and the district leans
        # still carry it. Aborting would publish nothing when we can publish
        # something honest and clearly labelled.
        log.warning("no district polls; the House forecast rests on fundamentals alone")

    # --- model -----------------------------------------------------------
    fund = F.compute(races, cfg)
    data = build_model_data(table, races, cfg, fund)
    log.info(
        "design: %d races x %d grid points, %d race polls, %d national polls, "
        "%d house effects",
        data.n_races, data.n_grid, data.n_race_polls, data.n_national_polls,
        len(data.pollster_names),
    )

    model = build_model(data, cfg)
    log.info("sampling (%d draws x %d chains)...", cfg.sampler.draws, cfg.sampler.chains)
    idata = sample(model, cfg, progressbar=not args.quiet_sampler)

    diagnostics = convergence_report(idata)
    log.info("convergence: %s", diagnostics)
    if diagnostics["max_r_hat"] > 1.05 or diagnostics["divergences"] > 0:
        log.warning(
            "SAMPLING QUALITY WARNING: max_r_hat=%.3f divergences=%d — "
            "treat this run's numbers with caution",
            diagnostics["max_r_hat"], diagnostics["divergences"],
        )

    simulation = simulate_chamber(idata, races, cfg, fund)

    # --- outputs ---------------------------------------------------------
    run = outputs.ForecastRun(
        run_date=run_date,
        races=races,
        cfg=cfg,
        table=table,
        fundamentals=fund,
        simulation=simulation,
        idata=idata,
        diagnostics=diagnostics,
        roster=roster,
    )

    previous = previous_payload(run_date, chamber=chamber)
    payload = run.to_dict()

    forecast_path = outputs.write_forecast(run)
    outputs.append_history(run)

    commentary = generate_commentary(payload, previous)
    _feed_path, changelog_path = write_commentary(commentary, chamber=chamber)

    # --- report ----------------------------------------------------------
    # Named `summary`, not `chamber`. This line used to rebind `chamber` from
    # the chamber name to this dict, which nothing caught because the only later
    # use of the name was a hardcoded path -- itself the bug below.
    summary = payload["chamber_forecast"]
    print()
    print("=" * 72)
    print(f"  {races.chamber.upper()} {races.cycle} — forecast for {run_date}")
    print("=" * 72)
    print(f"  P(Democratic control): {summary['dem_control_prob']:.1%}")
    print(f"  P(Republican control): {summary['rep_control_prob']:.1%}")
    print(
        f"  Democratic seats: {summary['dem_seats']['median']} "
        f"(90% interval {summary['dem_seats']['p05']}–{summary['dem_seats']['p95']})"
    )
    print()
    print("  " + commentary.headline)
    print()
    # Both paths reported as returned rather than reconstructed. The changelog
    # line was hardcoded to CHANGELOG.md and so told a House run it had written
    # the Senate's file. It had not -- the write was correct -- but a run that
    # misreports where it put something is a run you cannot check.
    print(f"  forecast  -> {forecast_path}")
    print(f"  changelog -> {changelog_path}")
    print("=" * 72)
    return 0


def cmd_backtest_history(args: argparse.Namespace) -> int:
    from .backtest_history import run_historical_backtest

    return run_historical_backtest()


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import run_calibration

    return run_calibration(
        days_window=(args.min_days, args.max_days),
        min_cycle=args.min_cycle,
        chamber=args.chamber,
    )


def cmd_build_map(args: argparse.Namespace) -> int:
    from .geo import write_state_paths

    path = write_state_paths()
    print(f"Projected state paths written to {path}")
    return 0


def cmd_stamp_assets(args: argparse.Namespace) -> int:
    """Pin script and stylesheet URLs to their contents before deploying."""
    from pathlib import Path

    from .assets import stamp
    from .paths import SITE_DIR

    site = Path(args.site) if args.site else SITE_DIR
    try:
        applied = stamp(site)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    print(f"Stamped {len(applied)} assets in {site / 'index.html'}")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .bundle import build

    output = Path(args.output) if args.output else None
    try:
        path = build(output=output)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    print(f"Standalone dashboard written to {path}")
    print("Open it directly in a browser, or send the single file to someone.")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest import run_backtest

    return run_backtest(holdout_days=args.holdout_days, verbose=True)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="midterms", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="full pipeline: fetch, fit, simulate, write")
    run.add_argument("--offline", action="store_true", help="reuse the latest raw snapshot")
    run.add_argument("--date", help="run as of this date (YYYY-MM-DD); defaults to today")
    run.add_argument(
        "--chamber", choices=("senate", "house"), default="senate",
        help="which chamber to forecast (default: senate)",
    )
    run.add_argument(
        "--quiet-sampler", action="store_true", help="suppress the sampler progress bar"
    )
    run.set_defaults(func=cmd_run)

    fetch = sub.add_parser("fetch", help="fetch and snapshot polls only")
    fetch.set_defaults(func=cmd_fetch)

    audit = sub.add_parser("audit-roster", help="find candidate names the roster misses")
    audit.add_argument("--offline", action="store_true")
    audit.add_argument(
        "--chamber", choices=("senate", "house"), default="senate",
        help="which chamber's roster to audit (default: senate)",
    )
    audit.set_defaults(func=cmd_audit_roster)

    fetch_markets = sub.add_parser(
        "fetch-markets", help="snapshot prediction-market odds for the dashboard"
    )
    fetch_markets.set_defaults(func=cmd_fetch_markets)

    check_keys = sub.add_parser(
        "check-keys", help="report which optional API keys are set and working"
    )
    check_keys.add_argument(
        "--offline", action="store_true", help="report presence without calling the APIs"
    )
    check_keys.set_defaults(func=cmd_check_keys)

    audit_pollsters = sub.add_parser(
        "audit-pollsters", help="find pollster names that look like duplicates"
    )
    audit_pollsters.add_argument("--offline", action="store_true")
    audit_pollsters.set_defaults(func=cmd_audit_pollsters)

    hist = sub.add_parser(
        "backtest-history",
        help="score win probabilities against elections that actually happened",
    )
    hist.set_defaults(func=cmd_backtest_history)

    calibrate = sub.add_parser(
        "calibrate", help="fit error scales from historical polling"
    )
    calibrate.add_argument("--min-days", type=int, default=45)
    calibrate.add_argument("--max-days", type=int, default=120)
    calibrate.add_argument("--min-cycle", type=int, default=2010)
    calibrate.add_argument(
        "--chamber", choices=("senate", "house"), default="senate",
        help="which chamber's historical polling to fit (default: senate)",
    )
    calibrate.set_defaults(func=cmd_calibrate)

    build_map = sub.add_parser(
        "build-map", help="project the vendored TopoJSON into SVG paths for the dashboard"
    )
    build_map.set_defaults(func=cmd_build_map)

    bundle = sub.add_parser(
        "bundle", help="build a single self-contained HTML dashboard file"
    )
    bundle.add_argument("-o", "--output", help="output path (default: outputs/dashboard.html)")
    bundle.set_defaults(func=cmd_bundle)

    stamp_assets = sub.add_parser(
        "stamp-assets",
        help="add a content hash to the dashboard's script and stylesheet URLs",
    )
    stamp_assets.add_argument("--site", help="site directory (default: site/)")
    stamp_assets.set_defaults(func=cmd_stamp_assets)

    backtest = sub.add_parser("backtest", help="calibration check against held-out polls")
    backtest.add_argument("--holdout-days", type=int, default=30)
    backtest.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
