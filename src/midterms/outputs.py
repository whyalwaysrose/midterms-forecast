"""Serialise a forecast run to the JSON the dashboard reads.

Two files matter:

``forecast.json``
    The complete current picture: chamber probabilities, the seat distribution,
    every race, latent trajectories, diagnostics, and the polls behind it all.

``history.json``
    One compact record appended per run. This is what makes day-over-day
    commentary possible: it is the memory of what we believed yesterday.

Both are plain JSON with a ``schema_version``, so the front end can fail loudly
rather than silently rendering a stale shape.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .config import ModelConfig, RaceSet
from .data.polls import PLACEHOLDER_NAMES, PollTable
from .data.roster import Roster
from .data.votehub import ATTRIBUTION
from .fundamentals import Fundamentals, logit_to_margin
from .model.simulate import SimulationResult, national_environment_summary

log = logging.getLogger(__name__)

SCHEMA_VERSION = 5


def model_fingerprint() -> str:
    """A short hash of everything that decides the numbers.

    Day-over-day commentary compares two runs and attributes what moved to the
    polls. That is only honest while the model is held fixed. Change a prior,
    or the simulation, and every race shifts at once — which the differ would
    otherwise report as though the polls had done it.

    Hashing the config plus the two modules that define the arithmetic means a
    model change is detected automatically, with nothing to remember to bump.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in (
        paths.MODEL_CONFIG,
        paths.REPO_ROOT / "src" / "midterms" / "model" / "hierarchical.py",
        paths.REPO_ROOT / "src" / "midterms" / "model" / "simulate.py",
        paths.REPO_ROOT / "src" / "midterms" / "fundamentals.py",
    ):
        try:
            content = path.read_bytes()
        except OSError:  # pragma: no cover - only if the tree is incomplete
            digest.update(b"<missing>")
            continue
        # Normalise line endings before hashing. Git checks the same files out
        # with CRLF on Windows and LF on Linux, so the raw bytes differ between
        # a local run and CI even though the model is byte-identical in the
        # repository. That produced exactly the false alarm this fingerprint
        # exists to prevent: the first CI run announced "the model changed in
        # this run" when nothing had changed but the checkout.
        digest.update(content.replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:12]

#: Quantiles reported for every margin, in a fixed order.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
QUANTILE_KEYS = ("p05", "p25", "p50", "p75", "p95")


def _round(value: Any, digits: int = 4) -> Any:
    """Round floats for compact, diff-friendly JSON."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def _trajectory(idata, race_index: int | None, grid_dates, thin: int = 1) -> list[dict]:
    """Median and 90% band of the latent margin over time.

    ``race_index=None`` returns the national environment instead of a race.
    Note this is the *poll-based* latent state: election-day error is applied
    only to the final forecast, not to history, because history is something we
    are estimating rather than predicting.
    """
    if race_index is None:
        draws = idata.posterior["eta"].to_numpy().reshape(-1, len(grid_dates))
    else:
        draws = (
            idata.posterior["theta"]
            .isel(race=race_index)
            .to_numpy()
            .reshape(-1, len(grid_dates))
        )

    margins = logit_to_margin(draws)
    lo, med, hi = np.quantile(margins, (0.05, 0.5, 0.95), axis=0)

    return [
        {
            "date": grid_dates[i].isoformat(),
            "p05": _round(lo[i], 2),
            "p50": _round(med[i], 2),
            "p95": _round(hi[i], 2),
        }
        for i in range(0, len(grid_dates), thin)
    ]


def _candidates(table: PollTable, race_id: str, roster: Roster) -> dict[str, str | None]:
    """Who is actually running, per side.

    Taken from the polls themselves where possible — the most recent name each
    side has been polled under is the best available evidence of who the
    nominee is, and it updates itself after a primary without anyone editing a
    file. Races with no polling fall back to the roster, which is why the
    roster lists nominees for unpolled races too.
    """
    polls = sorted(table.for_race(race_id), key=lambda p: p.field_date, reverse=True)

    def most_recent(attribute: str) -> str | None:
        for poll in polls:
            name = getattr(poll, attribute)
            # Generic placeholders ("Dem", "Rep") name no one.
            if name and name.lower() not in PLACEHOLDER_NAMES:
                return name
        return None

    dem = most_recent("dem_candidate")
    rep = most_recent("rep_candidate")

    if dem is None or rep is None:
        entry = roster.by_race.get(race_id, {})
        listed: dict[str, list[str]] = {"D": [], "R": []}
        for key, side in entry.items():
            if side in listed:
                listed[side].append(key)
        # by_race stores case-folded keys, so recover the display spelling.
        display = roster.display_names.get(race_id, {})

        def sole(side: str) -> str | None:
            """The one name on this side, or nothing.

            The roster carries whole primary fields -- ten of the 2026 races
            have several candidates on a side -- and it is not ordered by who
            won. With more than one listed there is no evidence here about
            which is the nominee, so name nobody; the page then says
            "Democratic candidate", which is true, rather than a real person
            who is not on the ballot.
            """
            if len(listed[side]) != 1:
                return None
            key = listed[side][0]
            return display.get(key, key.title())

        if dem is None:
            dem = sole("D")
        if rep is None:
            rep = sole("R")

    return {"dem": dem, "rep": rep}


def _polls_for_display(table: PollTable, race_id: str, limit: int = 25) -> list[dict]:
    polls = sorted(table.for_race(race_id), key=lambda p: p.field_date, reverse=True)
    return [
        {
            "id": p.poll_id,
            "pollster": p.pollster,
            "date": p.field_date.isoformat(),
            "start_date": p.start_date.isoformat(),
            "end_date": p.end_date.isoformat(),
            "sample_size": p.sample_size,
            "population": p.population,
            "dem_pct": _round(p.dem_pct, 1),
            "rep_pct": _round(p.rep_pct, 1),
            "margin": _round(p.margin, 1),
            "partisan_sign": p.partisan_sign,
            "sponsors": list(p.sponsors),
            "url": p.url,
            "dem_candidate": p.dem_candidate,
            "rep_candidate": p.rep_candidate,
        }
        for p in polls[:limit]
    ]


@dataclass
class ForecastRun:
    """Everything one run produced, ready to serialise."""

    run_date: date
    races: RaceSet
    cfg: ModelConfig
    table: PollTable
    fundamentals: Fundamentals
    simulation: SimulationResult
    idata: Any
    diagnostics: dict[str, float]
    roster: Roster

    def to_dict(self) -> dict:
        races = self.races
        sim = self.simulation
        grid_dates = [
            date.fromisoformat(d) for d in self.idata.posterior.coords["grid"].values
        ]

        margin_q = sim.margin_quantiles(QUANTILES)
        tipping = sim.tipping_point_probs()
        poll_counts = self.table.counts_by_race()
        prior_margins = logit_to_margin(self.fundamentals.prior_mean)

        race_records = []
        for i, race in enumerate(races.races):
            race_polls = self.table.for_race(race.id)
            latest = max((p.field_date for p in race_polls), default=None)

            notes: list[str] = []
            if race.id in self.roster.independent_notes:
                notes.append(self.roster.independent_notes[race.id])
            if not race_polls:
                notes.append(
                    "No qualifying general-election polls; this forecast is carried "
                    "by the fundamentals prior and the national environment."
                )

            race_records.append(
                {
                    "id": race.id,
                    "unit": race.unit,
                    "name": race.name,
                    "special": race.special,
                    "incumbent_party": race.incumbent_party,
                    "incumbent_status": race.incumbent_status,
                    "dem_win_prob": _round(sim.dem_win_prob[i]),
                    "rep_win_prob": _round(1.0 - sim.dem_win_prob[i]),
                    "margin": {
                        key: _round(margin_q[j][i], 2)
                        for j, key in enumerate(QUANTILE_KEYS)
                    },
                    "fundamentals_prior_margin": _round(prior_margins[i], 2),
                    "tipping_point_prob": _round(tipping.get(race.id, 0.0)),
                    "candidates": _candidates(self.table, race.id, self.roster),
                    "poll_count": int(poll_counts.get(race.id, 0)),
                    "latest_poll_date": latest.isoformat() if latest else None,
                    "trajectory": _trajectory(self.idata, i, grid_dates),
                    "polls": _polls_for_display(self.table, race.id),
                    # Every poll id in the window, not just the displayed ones.
                    # `polls` is capped for display, so diffing on it alone
                    # would misreport a back-filled older poll (VoteHub does
                    # add polls fielded months earlier) as brand new.
                    "all_poll_ids": [p.poll_id for p in race_polls],
                    "notes": notes,
                }
            )

        # Order the dashboard by competitiveness: closest races first.
        race_records.sort(key=lambda r: abs(r["dem_win_prob"] - 0.5))

        seat_dist = sim.seat_distribution(races.control.total_seats)

        return {
            "schema_version": SCHEMA_VERSION,
            "model_fingerprint": model_fingerprint(),
            "run_date": self.run_date.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cycle": races.cycle,
            "chamber": races.chamber,
            "election_date": races.election_date.isoformat(),
            "days_to_election": (races.election_date - self.run_date).days,
            "data_source": {
                "name": "VoteHub Polling API",
                "url": "https://votehub.com/polls/api/",
                "license": "CC BY 4.0",
                "attribution": ATTRIBUTION,
            },
            "national": {
                "generic_ballot": {
                    k: _round(v, 2)
                    for k, v in national_environment_summary(self.idata).items()
                },
                "trajectory": _trajectory(self.idata, None, grid_dates),
            },
            "chamber_forecast": {
                "dem_control_prob": _round(sim.dem_control_prob),
                "rep_control_prob": _round(sim.rep_control_prob),
                "exact_tie_prob": _round(sim.tie_prob),
                "dem_seats": {
                    "mean": _round(float(np.mean(sim.dem_seats)), 2),
                    "median": int(np.median(sim.dem_seats)),
                    "p05": int(np.quantile(sim.dem_seats, 0.05)),
                    "p25": int(np.quantile(sim.dem_seats, 0.25)),
                    "p75": int(np.quantile(sim.dem_seats, 0.75)),
                    "p95": int(np.quantile(sim.dem_seats, 0.95)),
                },
                "seat_distribution": {str(k): _round(v) for k, v in seat_dist.items()},
                "seats_not_up": dict(races.control.seats_not_up),
                "dem_seats_for_majority": races.control.dem_seats_for_majority,
                "tiebreaker_party": races.control.tiebreaker_party,
                "n_simulations": int(sim.n_sims),
            },
            "races": race_records,
            "methodology": _methodology_block(self.cfg),
            "diagnostics": {k: _round(v, 4) for k, v in self.diagnostics.items()},
            "poll_summary": {
                "n_race_polls": len(self.table.race_polls),
                "n_national_polls": len(self.table.national),
                "n_pollsters": len({p.pollster for p in self.table.polls}),
                "rejections": dict(self.table.rejections),
                "unknown_candidates": {
                    k: sorted(v) for k, v in self.table.unknown_candidates.items()
                },
            },
        }

    def history_record(self) -> dict:
        """The compact per-run record appended to ``history.json``."""
        sim = self.simulation
        return {
            "run_date": self.run_date.isoformat(),
            "dem_control_prob": _round(sim.dem_control_prob),
            "dem_seats_mean": _round(float(np.mean(sim.dem_seats)), 2),
            "dem_seats_median": int(np.median(sim.dem_seats)),
            "generic_ballot_dem_margin": _round(
                national_environment_summary(self.idata)["dem_margin_median"], 2
            ),
            "n_race_polls": len(self.table.race_polls),
            "races": {
                race.id: {
                    "dem_win_prob": _round(sim.dem_win_prob[i]),
                    "margin_p50": _round(sim.margin_quantiles((0.5,))[0][i], 2),
                    "poll_count": int(self.table.counts_by_race().get(race.id, 0)),
                }
                for i, race in enumerate(self.races.races)
            },
        }


def _methodology_block(cfg: ModelConfig) -> dict:
    """Calibration and validation figures, computed rather than written down.

    The dashboard explains how the uncertainty was arrived at. Hard-coding those
    numbers into the page would let them drift silently the moment anyone
    re-fits the scales — the page would keep describing a model that no longer
    exists. Recomputing them each run costs a couple of seconds and makes the
    explanation structurally unable to lie.
    """
    from .calibration import POINTS_PER_LOGIT, estimate_error_components

    block: dict[str, Any] = {
        "election_day_error": {
            "national_pts": _round(cfg.election_day_error.national_sd * POINTS_PER_LOGIT, 2),
            "state_pts": _round(cfg.election_day_error.state_sd * POINTS_PER_LOGIT, 2),
            "total_pts": _round(
                float(
                    np.hypot(
                        cfg.election_day_error.national_sd,
                        cfg.election_day_error.state_sd,
                    )
                )
                * POINTS_PER_LOGIT,
                2,
            ),
        },
        "design_effect": _round(cfg.polls.design_effect, 2),
        "student_t_nu": cfg.polls.student_t_nu,
    }

    try:
        fitted = estimate_error_components()
        block["calibration"] = {
            "source": "FiveThirtyEight pollster-ratings archive (CC BY 4.0)",
            "n_polls": fitted.n_polls,
            "n_races": fitted.n_races,
            "n_cycles": fitted.n_cycles,
            "window_days": list(fitted.window),
            "min_cycle": fitted.min_cycle,
            "fitted_national_pts": _round(fitted.national_sd, 2),
            "fitted_state_pts": _round(fitted.state_sd, 2),
            "fitted_poll_pts": _round(fitted.poll_sd, 2),
            "fitted_design_effect": _round(fitted.design_effect, 2),
        }
    except Exception as exc:  # pragma: no cover - only if history is missing
        log.warning("calibration figures unavailable for the dashboard: %s", exc)

    try:
        from .backtest_history import build_race_table, score

        races = build_race_table()
        result = score(
            cfg.election_day_error.national_sd * POINTS_PER_LOGIT,
            cfg.election_day_error.state_sd * POINTS_PER_LOGIT,
            "current",
            races,
        )
        block["backtest"] = {
            "n_races": result.n_races,
            "n_cycles": result.n_cycles,
            "brier": _round(result.brier, 4),
            "skill_vs_naive": _round(result.brier_skill_vs_naive, 3),
            "coverage": {str(k): _round(v, 3) for k, v in result.coverage.items()},
            "reliability": [
                {
                    "bin": row["bin"],
                    "n": int(row["n"]),
                    "predicted": _round(row["mean_pred"], 3),
                    "actual": _round(row["actual"], 3),
                }
                for _, row in result.reliability.iterrows()
            ],
        }
    except Exception as exc:  # pragma: no cover
        log.warning("backtest figures unavailable for the dashboard: %s", exc)

    return block


def slim_payload(payload: dict) -> dict:
    """Strip a forecast down to what day-over-day diffing actually needs.

    The full payload is ~380 KB, dominated by latent trajectories and poll
    display records. Archiving that every day would add tens of megabytes to
    the repository over a cycle for no benefit, because the commentary
    generator reads the *previous* run only for:

      - chamber-level and national headline numbers,
      - each race's win probability and median margin,
      - the set of poll IDs already seen.

    Details of any genuinely new poll come from the *current* payload, not the
    archived one, so IDs alone are sufficient here.
    """
    return {
        "schema_version": payload["schema_version"],
        "model_fingerprint": payload.get("model_fingerprint"),
        "run_date": payload["run_date"],
        "generated_at": payload["generated_at"],
        "chamber_forecast": payload["chamber_forecast"],
        "national": {"generic_ballot": payload["national"]["generic_ballot"]},
        "poll_summary": payload["poll_summary"],
        "diagnostics": payload["diagnostics"],
        "races": [
            {
                "id": race["id"],
                "name": race["name"],
                "dem_win_prob": race["dem_win_prob"],
                "margin": race["margin"],
                "poll_count": race["poll_count"],
                "all_poll_ids": race.get("all_poll_ids", []),
            }
            for race in payload.get("races", [])
        ],
    }


def write_forecast(run: ForecastRun, site_data_dir: Path | None = None) -> Path:
    """Write the full ``forecast.json`` and archive a slim dated copy."""
    site_data_dir = site_data_dir or paths.SITE_DATA_DIR
    site_data_dir.mkdir(parents=True, exist_ok=True)

    payload = run.to_dict()
    target = site_data_dir / "forecast.json"
    target.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    archive_dir = paths.run_dir(run.run_date.isoformat())
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "forecast.json").write_text(
        json.dumps(slim_payload(payload), indent=1), encoding="utf-8"
    )
    return target


def append_history(run: ForecastRun, site_data_dir: Path | None = None) -> Path:
    """Append this run to ``history.json``, replacing any record for the same date.

    Replacing rather than appending on a repeat date keeps the series
    idempotent, so re-running the pipeline twice in one day does not create two
    conflicting entries for that day.
    """
    site_data_dir = site_data_dir or paths.SITE_DATA_DIR
    site_data_dir.mkdir(parents=True, exist_ok=True)
    target = site_data_dir / "history.json"

    records: list[dict] = []
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            records = loaded.get("runs", []) if isinstance(loaded, dict) else list(loaded)
        except json.JSONDecodeError:
            records = []

    record = run.history_record()
    records = [r for r in records if r.get("run_date") != record["run_date"]]
    records.append(record)
    records.sort(key=lambda r: r["run_date"])

    target.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "runs": records}, indent=1),
        encoding="utf-8",
    )
    return target


def load_history(site_data_dir: Path | None = None) -> list[dict]:
    """Read the history series, oldest first."""
    site_data_dir = site_data_dir or paths.SITE_DATA_DIR
    target = site_data_dir / "history.json"
    if not target.exists():
        return []
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    records = loaded.get("runs", []) if isinstance(loaded, dict) else list(loaded)
    return sorted(records, key=lambda r: r.get("run_date", ""))
