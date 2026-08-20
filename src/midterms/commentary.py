"""Day-over-day commentary: what moved, and what moved it.

This diffs the current run against the previous archived run and writes a
human-readable note. The important discipline here is **attribution**: it is
easy to report that Georgia moved 2 points and useless to leave it there. Every
race-level change is paired with the polls that arrived since the last run, so
a reader can see whether a move was driven by new data, and by whose data.

Where nothing new arrived for a race that nonetheless moved, that is stated
explicitly too — such moves come from the national environment or from
correlated movement in similar states, and saying so is more honest than
implying a poll caused it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from . import paths

#: A race must move at least this much to be worth mentioning.
MIN_PROB_CHANGE = 0.01
MIN_MARGIN_CHANGE = 0.3

#: A margin shift only counts as news in a race that could plausibly flip.
#:
#: Safe seats drift by a few tenths of a point every run as the national
#: environment moves, and in a race sitting at 0.6% that drift is arithmetic,
#: not information. Without this guard the "races that moved" list fills up with
#: Kentucky and Tennessee while the reader is looking for Georgia and Michigan.
#: A large probability change is still always reported, whatever the race.
COMPETITIVE_BAND = (0.02, 0.98)


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _fmt_signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def _fmt_margin(margin: float) -> str:
    """A margin as a party-leading string, e.g. ``D+3.2``."""
    if margin >= 0:
        return f"D+{margin:.1f}"
    return f"R+{abs(margin):.1f}"


def _pp(value: float) -> str:
    """Percentage-point change with an explicit sign."""
    return f"{100.0 * value:+.1f} pts"


def _pp_abs(value: float) -> str:
    """Magnitude in percentage points, unsigned.

    Used where a direction word ("up"/"down") already carries the sign, so that
    we never emit "down +0.7 pts".
    """
    return f"{100.0 * abs(value):.1f} pts"


@dataclass
class RaceChange:
    race_id: str
    name: str
    prob_before: float
    prob_after: float
    margin_before: float
    margin_after: float
    new_polls: list[dict] = field(default_factory=list)

    @property
    def prob_delta(self) -> float:
        return self.prob_after - self.prob_before

    @property
    def margin_delta(self) -> float:
        return self.margin_after - self.margin_before

    @property
    def is_competitive(self) -> bool:
        """Could this race plausibly flip, before or after the move?"""
        low, high = COMPETITIVE_BAND
        return any(low <= p <= high for p in (self.prob_before, self.prob_after))

    @property
    def is_notable(self) -> bool:
        if abs(self.prob_delta) >= MIN_PROB_CHANGE:
            return True
        return abs(self.margin_delta) >= MIN_MARGIN_CHANGE and self.is_competitive

    def describe(self) -> str:
        head = (
            f"**{self.name}** — Democratic win probability "
            f"{_fmt_pct(self.prob_before)} → {_fmt_pct(self.prob_after)} "
            f"({_pp(self.prob_delta)}); projected margin "
            f"{_fmt_margin(self.margin_before)} → {_fmt_margin(self.margin_after)} "
            f"({_fmt_signed(self.margin_delta)})."
        )
        if not self.new_polls:
            return (
                head
                + " No new polls in this race; the move comes from the national"
                " environment and from correlated movement in similar states."
            )

        bullets = []
        for poll in self.new_polls:
            sponsor = ""
            if poll.get("partisan_sign"):
                side = "D" if poll["partisan_sign"] > 0 else "R"
                sponsor = f", {side}-aligned sponsor"
            bullets.append(
                f"    - {poll['pollster']} ({poll['date']}, n={poll['sample_size']}, "
                f"{poll['population'].upper()}{sponsor}): {_fmt_margin(poll['margin'])}"
            )
        label = "new poll" if len(self.new_polls) == 1 else "new polls"
        return head + f" Driven by {len(self.new_polls)} {label}:\n" + "\n".join(bullets)


@dataclass
class Commentary:
    run_date: str
    previous_run_date: str | None
    headline: str
    body: list[str]
    race_changes: list[RaceChange]
    new_poll_count: int

    def to_markdown(self) -> str:
        lines = [f"## {self.run_date}", "", self.headline, ""]
        lines.extend(self.body)
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict:
        return {
            "run_date": self.run_date,
            "previous_run_date": self.previous_run_date,
            "headline": self.headline,
            "body": self.body,
            "new_poll_count": self.new_poll_count,
            "race_changes": [
                {
                    "race_id": c.race_id,
                    "name": c.name,
                    "prob_before": c.prob_before,
                    "prob_after": c.prob_after,
                    "prob_delta": c.prob_delta,
                    "margin_before": c.margin_before,
                    "margin_after": c.margin_after,
                    "margin_delta": c.margin_delta,
                    "new_poll_count": len(c.new_polls),
                    "new_polls": c.new_polls,
                }
                for c in self.race_changes
            ],
        }


def _poll_ids(payload: dict) -> set[str]:
    """Every poll id a payload knows about.

    Prefers ``all_poll_ids`` (complete) and falls back to the capped ``polls``
    display list only for payloads written before that field existed.
    """
    ids: set[str] = set()
    for race in payload.get("races", []):
        if race.get("all_poll_ids"):
            ids.update(race["all_poll_ids"])
        else:
            ids.update(poll["id"] for poll in race.get("polls", []))
    return ids


def _first_run_commentary(current: dict) -> Commentary:
    chamber = current["chamber_forecast"]
    national = current["national"]["generic_ballot"]
    summary = current["poll_summary"]

    headline = (
        f"First run of the model. Democrats have a "
        f"{_fmt_pct(chamber['dem_control_prob'])} chance of Senate control, with a "
        f"projected {chamber['dem_seats']['median']} seats "
        f"(90% interval {chamber['dem_seats']['p05']}–{chamber['dem_seats']['p95']})."
    )
    body = [
        f"Fitted to {summary['n_race_polls']} state-level polls and "
        f"{summary['n_national_polls']} generic-ballot polls from "
        f"{summary['n_pollsters']} pollsters.",
        "",
        f"The national environment is estimated at "
        f"{_fmt_margin(national['dem_margin_median'])} on the generic ballot "
        f"(90% interval {_fmt_margin(national['dem_margin_p05'])} to "
        f"{_fmt_margin(national['dem_margin_p95'])}).",
        "",
        "Closest races:",
    ]
    for race in current["races"][:6]:
        body.append(
            f"  - **{race['name']}**: {_fmt_pct(race['dem_win_prob'])} Democratic, "
            f"projected {_fmt_margin(race['margin']['p50'])} "
            f"({race['poll_count']} poll{'s' if race['poll_count'] != 1 else ''})."
        )
    body.append("")
    body.append("Day-over-day commentary begins with the next run.")

    return Commentary(
        run_date=current["run_date"],
        previous_run_date=None,
        headline=headline,
        body=body,
        race_changes=[],
        new_poll_count=summary["n_race_polls"],
    )


def generate(current: dict, previous: dict | None) -> Commentary:
    """Build the commentary for ``current`` relative to ``previous``."""
    if previous is None:
        return _first_run_commentary(current)

    previous_ids = _poll_ids(previous)
    prev_races = {r["id"]: r for r in previous.get("races", [])}

    changes: list[RaceChange] = []
    total_new_polls = 0

    for race in current.get("races", []):
        before = prev_races.get(race["id"])
        if before is None:
            continue

        new_polls = [p for p in race.get("polls", []) if p["id"] not in previous_ids]
        total_new_polls += len(new_polls)

        changes.append(
            RaceChange(
                race_id=race["id"],
                name=race["name"],
                prob_before=before["dem_win_prob"],
                prob_after=race["dem_win_prob"],
                margin_before=before["margin"]["p50"],
                margin_after=race["margin"]["p50"],
                new_polls=sorted(new_polls, key=lambda p: p["date"], reverse=True),
            )
        )

    notable = [c for c in changes if c.is_notable]
    notable.sort(key=lambda c: -abs(c.prob_delta))

    cur_chamber = current["chamber_forecast"]
    prev_chamber = previous["chamber_forecast"]
    control_delta = cur_chamber["dem_control_prob"] - prev_chamber["dem_control_prob"]
    seats_delta = cur_chamber["dem_seats"]["mean"] - prev_chamber["dem_seats"]["mean"]

    if abs(control_delta) < 0.005 and total_new_polls == 0:
        headline = (
            f"No new polling since {previous['run_date']}. Democratic chances of "
            f"Senate control hold at {_fmt_pct(cur_chamber['dem_control_prob'])}."
        )
    else:
        poll_phrase = (
            f"{total_new_polls} new poll{'s' if total_new_polls != 1 else ''}"
            if total_new_polls
            else "no new polls"
        )
        if abs(control_delta) < 0.0005:
            movement = f"are unchanged at {_fmt_pct(cur_chamber['dem_control_prob'])}"
        else:
            direction = "up" if control_delta > 0 else "down"
            movement = (
                f"are {direction} {_pp_abs(control_delta)} to "
                f"{_fmt_pct(cur_chamber['dem_control_prob'])}"
            )
        headline = (
            f"Democratic chances of Senate control {movement} on {poll_phrase}. "
            f"Projected seats {prev_chamber['dem_seats']['mean']:.1f} → "
            f"{cur_chamber['dem_seats']['mean']:.1f} ({_fmt_signed(seats_delta)})."
        )

    body: list[str] = []

    cur_gb = current["national"]["generic_ballot"]["dem_margin_median"]
    prev_gb = previous["national"]["generic_ballot"]["dem_margin_median"]
    if abs(cur_gb - prev_gb) >= 0.1:
        body.append(
            f"**National environment.** The generic ballot moved "
            f"{_fmt_margin(prev_gb)} → {_fmt_margin(cur_gb)} "
            f"({_fmt_signed(cur_gb - prev_gb)}). Because every race is geared to the "
            f"national environment by its own elasticity, this shifts all 35 contests "
            f"together, not just the ones with new polls."
        )
        body.append("")

    if notable:
        body.append(f"**Races that moved** ({len(notable)} of {len(changes)}):")
        body.append("")
        for change in notable[:10]:
            body.append(f"  - {change.describe()}")
            body.append("")
    else:
        body.append("No individual race moved enough to be worth reporting.")
        body.append("")

    races_with_new_polls = [c for c in changes if c.new_polls and not c.is_notable]
    if races_with_new_polls:
        names = ", ".join(
            f"{c.name} ({len(c.new_polls)})" for c in races_with_new_polls[:8]
        )
        body.append(
            f"**New polling that did not move the needle.** {names}. "
            f"New polls close to the existing estimate confirm it rather than change it."
        )
        body.append("")

    return Commentary(
        run_date=current["run_date"],
        previous_run_date=previous["run_date"],
        headline=headline,
        body=body,
        race_changes=notable,
        new_poll_count=total_new_polls,
    )


def previous_payload(run_date: date, runs_dir: Path | None = None) -> dict | None:
    """The most recent archived forecast strictly before ``run_date``."""
    runs_dir = runs_dir or paths.RUNS_DIR
    if not runs_dir.exists():
        return None

    candidates = sorted(
        p for p in runs_dir.iterdir()
        if p.is_dir() and (p / "forecast.json").exists() and p.name < run_date.isoformat()
    )
    if not candidates:
        return None
    return json.loads((candidates[-1] / "forecast.json").read_text(encoding="utf-8"))


def write_commentary(
    commentary: Commentary,
    site_data_dir: Path | None = None,
    changelog_path: Path | None = None,
) -> tuple[Path, Path]:
    """Append the entry to ``commentary.json`` and prepend it to ``CHANGELOG.md``."""
    site_data_dir = site_data_dir or paths.SITE_DATA_DIR
    site_data_dir.mkdir(parents=True, exist_ok=True)
    changelog_path = changelog_path or paths.REPO_ROOT / "CHANGELOG.md"

    # --- structured feed for the dashboard --------------------------------
    feed_path = site_data_dir / "commentary.json"
    entries: list[dict[str, Any]] = []
    if feed_path.exists():
        try:
            loaded = json.loads(feed_path.read_text(encoding="utf-8"))
            entries = loaded.get("entries", []) if isinstance(loaded, dict) else list(loaded)
        except json.JSONDecodeError:
            entries = []

    entry = commentary.to_dict()
    entry["markdown"] = commentary.to_markdown()
    entries = [e for e in entries if e.get("run_date") != entry["run_date"]]
    entries.append(entry)
    entries.sort(key=lambda e: e["run_date"], reverse=True)

    feed_path.write_text(
        json.dumps({"schema_version": 2, "entries": entries}, indent=1), encoding="utf-8"
    )

    # --- human-readable changelog, newest first ---------------------------
    header = (
        "# Forecast changelog\n\n"
        "Automatically generated after each model run. Newest entries first.\n\n"
    )
    sections = [e["markdown"] for e in entries]
    changelog_path.write_text(header + "\n---\n\n".join(sections), encoding="utf-8")

    return feed_path, changelog_path
