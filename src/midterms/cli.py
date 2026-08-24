"""Command-line entry points.

    midterms run              fetch polls, fit, simulate, write outputs
    midterms run --offline    same, but reuse the latest raw snapshot
    midterms fetch            fetch and snapshot polls only
    midterms audit-roster     report candidate names the roster cannot classify
    midterms audit-pollsters  report pollster names that look like duplicates
    midterms check-keys       report which optional API keys are configured
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


def _configure_logging(verbose: bool) -> None:
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
    from .config import load_all
    from .data.roster import Roster
    from .data.votehub import (
        VoteHubClient,
        is_primary_subject,
        latest_snapshot,
        load_snapshot,
    )

    races, _cfg = load_all()
    roster = Roster.load()

    snapshot = latest_snapshot()
    if snapshot is None or not args.offline:
        raw = VoteHubClient().fetch_and_snapshot(["us-senator", "generic-ballot", "approval"])
    else:
        raw = load_snapshot(snapshot)

    subject_to_race = {r.subject_for(races.cycle): r.id for r in races.races}
    unknown: dict[str, set[str]] = {}

    for record in raw.get("us-senator", []):
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
        "\nAdd each to config/candidates_senate_2026.yaml under the correct party, "
        "or under `other` for minor-party candidates."
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


def cmd_run(args: argparse.Namespace) -> int:
    from . import fundamentals as F
    from . import outputs, paths
    from .commentary import generate as generate_commentary
    from .commentary import previous_payload, write_commentary
    from .config import load_all
    from .data import Roster, build_poll_table
    from .data.votehub import VoteHubClient, latest_snapshot, load_snapshot
    from .model.design import build_model_data
    from .model.hierarchical import build_model, convergence_report, sample
    from .model.simulate import simulate_chamber

    paths.ensure_dirs()
    run_date = date.fromisoformat(args.date) if args.date else date.today()

    # Cheap and deterministic, so just regenerate it: it keeps the map in step
    # with the config if a race is ever added or a projection parameter changes.
    from .geo import write_state_paths

    write_state_paths()

    races, cfg = load_all()
    roster = Roster.load()
    log.info("loaded %d races for %s %d", len(races.races), races.chamber, races.cycle)

    # --- data ------------------------------------------------------------
    if args.offline:
        snapshot = latest_snapshot()
        if snapshot is None:
            log.error("--offline requested but no raw snapshot exists; run `midterms fetch`")
            return 2
        log.info("using cached snapshot %s", snapshot.name)
        raw = load_snapshot(snapshot)
    else:
        raw = VoteHubClient().fetch_and_snapshot(["us-senator", "generic-ballot", "approval"], run_date)

    table = build_poll_table(raw, races, cfg, roster, as_of=run_date)
    if table.unknown_candidates:
        log.warning(
            "unclassified candidates in %d races; run `midterms audit-roster`",
            len(table.unknown_candidates),
        )
    if not table.race_polls:
        log.error("no usable race polls; aborting")
        return 3

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

    previous = previous_payload(run_date)
    payload = run.to_dict()

    forecast_path = outputs.write_forecast(run)
    outputs.append_history(run)

    commentary = generate_commentary(payload, previous)
    write_commentary(commentary)

    # --- report ----------------------------------------------------------
    chamber = payload["chamber_forecast"]
    print()
    print("=" * 72)
    print(f"  {races.chamber.upper()} {races.cycle} — forecast for {run_date}")
    print("=" * 72)
    print(f"  P(Democratic control): {chamber['dem_control_prob']:.1%}")
    print(f"  P(Republican control): {chamber['rep_control_prob']:.1%}")
    print(
        f"  Democratic seats: {chamber['dem_seats']['median']} "
        f"(90% interval {chamber['dem_seats']['p05']}–{chamber['dem_seats']['p95']})"
    )
    print()
    print("  " + commentary.headline)
    print()
    print(f"  forecast  -> {forecast_path}")
    print(f"  changelog -> {paths.REPO_ROOT / 'CHANGELOG.md'}")
    print("=" * 72)
    return 0


def cmd_backtest_history(args: argparse.Namespace) -> int:
    from .backtest_history import run_historical_backtest

    return run_historical_backtest()


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import run_calibration

    return run_calibration(
        days_window=(args.min_days, args.max_days), min_cycle=args.min_cycle
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
        "--quiet-sampler", action="store_true", help="suppress the sampler progress bar"
    )
    run.set_defaults(func=cmd_run)

    fetch = sub.add_parser("fetch", help="fetch and snapshot polls only")
    fetch.set_defaults(func=cmd_fetch)

    audit = sub.add_parser("audit-roster", help="find candidate names the roster misses")
    audit.add_argument("--offline", action="store_true")
    audit.set_defaults(func=cmd_audit_roster)

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
