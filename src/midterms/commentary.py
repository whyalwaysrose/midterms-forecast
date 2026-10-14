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
    #: Polls that are new to the *forecast* but were already published when the
    #: previous run was made: they were in the feed and being rejected, and
    #: something other than a pollster changed to let them in -- in practice a
    #: candidate name the roster could not resolve until the roster was rebuilt.
    #:
    #: They have to be told apart from genuinely new polling or the commentary
    #: reports a data-coverage fix as fresh movement in the race. When the House
    #: roster was first rebuilt, 30 such polls landed at once -- taking district
    #: coverage from 38 to 60 -- one of them fielded six weeks earlier. Every one
    #: would have been called new, under a headline crediting fresh polling for
    #: 229 of 435 races moving.
    newly_usable: list[dict] = field(default_factory=list)
    #: Set when the run that produced this change used a different model, so a
    #: move cannot be attributed to polling.
    model_changed: bool = False

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
        if not self.new_polls and not self.newly_usable:
            if self.model_changed:
                return head + " No new polls in this race; this is the model revision."
            return (
                head
                + " No new polls in this race; the move comes from the national"
                " environment and from correlated movement in similar states."
            )

        def _bullets(polls: list[dict]) -> list[str]:
            out = []
            for poll in polls:
                sponsor = ""
                if poll.get("partisan_sign"):
                    side = "D" if poll["partisan_sign"] > 0 else "R"
                    sponsor = f", {side}-aligned sponsor"
                out.append(
                    f"    - {poll['pollster']} ({poll['date']}, "
                    f"n={poll['sample_size']}, {poll['population'].upper()}"
                    f"{sponsor}): {_fmt_margin(poll['margin'])}"
                )
            return out

        parts = []
        if self.new_polls:
            label = "new poll" if len(self.new_polls) == 1 else "new polls"
            parts.append(
                f" Driven by {len(self.new_polls)} {label}:\n"
                + "\n".join(_bullets(self.new_polls))
            )
        if self.newly_usable:
            n = len(self.newly_usable)
            # Said plainly, because the alternative is a reader seeing a poll
            # fielded weeks ago offered as the reason a race moved today.
            was = "This poll was" if n == 1 else f"These {n} polls were"
            parts.append(
                f"\n{'  ' if self.new_polls else ''}  {was} already published at "
                "the last run and counted for the first time in this one, so this "
                "part of the move is a correction to our coverage rather than a "
                "change in the race:\n"
                + "\n".join(_bullets(self.newly_usable))
            )
        return head + "".join(parts)


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
                    "newly_usable_count": len(c.newly_usable),
                    "newly_usable": c.newly_usable,
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


def _chamber_name(payload: dict) -> str:
    """"Senate" or "House", for prose.

    Read from the payload rather than passed in, so the commentary can never
    describe one chamber's numbers using the other's name -- the wording and the
    figures come from the same object.
    """
    return "House" if payload.get("chamber") == "house" else "Senate"


def _first_run_commentary(current: dict) -> Commentary:
    chamber = current["chamber_forecast"]
    which = _chamber_name(current)
    national = current["national"]["generic_ballot"]
    summary = current["poll_summary"]

    headline = (
        f"First run of the model. Democrats have a "
        f"{_fmt_pct(chamber['dem_control_prob'])} chance of {which} control, with a "
        f"projected {chamber['dem_seats']['median']} seats "
        f"(90% interval {chamber['dem_seats']['p05']}–{chamber['dem_seats']['p95']})."
    )
    body = [
        f"Fitted to {summary['n_race_polls']} "
        f"{'district' if which == 'House' else 'state'}-level polls and "
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

    # Did the model itself change since the last run? If so, every race can
    # move at once for reasons that have nothing to do with polling, and
    # attributing that to "the national environment" — as this differ otherwise
    # would — is simply false.
    model_changed = (
        current.get("model_fingerprint") is not None
        and previous.get("model_fingerprint") is not None
        and current["model_fingerprint"] != previous["model_fingerprint"]
    )

    previous_ids = _poll_ids(previous)
    prev_races = {r["id"]: r for r in previous.get("races", [])}

    changes: list[RaceChange] = []
    total_new_polls = 0
    total_newly_usable = 0

    for race in current.get("races", []):
        before = prev_races.get(race["id"])
        if before is None:
            continue

        # A poll absent from the previous run is not necessarily new. It may
        # have been in the feed and rejected -- most often because the roster
        # could not resolve a candidate name -- and become usable without any
        # pollster doing anything. Conflating the two reports our own coverage
        # catching up as movement in the race.
        #
        # `created_at` is when the feed first carried the poll, and is the only
        # field that separates them. The field date cannot: a poll fielded on
        # the 10th and published on the 14th is genuinely new to us on the 14th,
        # and most new polling arrives with a lag of days, so a field-date rule
        # would mislabel the ordinary case rather than the rare one.
        unseen = [p for p in race.get("polls", []) if p["id"] not in previous_ids]
        fresh, recovered = [], []
        for poll in unseen:
            created = poll.get("created_at")
            # Both uncertain cases resolve to "new", because "we had this poll
            # and were not counting it" is the claim that needs evidence:
            #
            #  - no `created_at` at all: a payload written before the field
            #    existed, or a feed that does not supply it. Falling back to
            #    "new" is also exactly what every run before this change did.
            #  - created on the day of the previous run: the date has no time in
            #    it, so that run may well have read the feed before the poll
            #    landed. Hence strictly before, not on or before.
            was_available = bool(created) and created < previous["run_date"]
            (recovered if was_available else fresh).append(poll)
        total_new_polls += len(fresh)
        total_newly_usable += len(recovered)

        changes.append(
            RaceChange(
                race_id=race["id"],
                name=race["name"],
                prob_before=before["dem_win_prob"],
                prob_after=race["dem_win_prob"],
                margin_before=before["margin"]["p50"],
                margin_after=race["margin"]["p50"],
                new_polls=sorted(fresh, key=lambda p: p["date"], reverse=True),
                newly_usable=sorted(
                    recovered, key=lambda p: p["date"], reverse=True
                ),
                model_changed=model_changed,
            )
        )

    notable = [c for c in changes if c.is_notable]
    notable.sort(key=lambda c: -abs(c.prob_delta))

    cur_chamber = current["chamber_forecast"]
    prev_chamber = previous["chamber_forecast"]
    which = _chamber_name(current)
    control_delta = cur_chamber["dem_control_prob"] - prev_chamber["dem_control_prob"]
    seats_delta = cur_chamber["dem_seats"]["mean"] - prev_chamber["dem_seats"]["mean"]

    if abs(control_delta) < 0.005 and total_new_polls == 0 and not total_newly_usable:
        headline = (
            f"No new polling since {previous['run_date']}. Democratic chances of "
            f"{which} control hold at {_fmt_pct(cur_chamber['dem_control_prob'])}."
        )
    else:
        poll_phrase = (
            f"{total_new_polls} new poll{'s' if total_new_polls != 1 else ''}"
            if total_new_polls
            else "no new polls"
        )
        if total_newly_usable:
            # Named in the headline, not just per race: on the day the roster is
            # rebuilt this is the largest single cause of movement in the run,
            # and a reader who sees only "N new polls" is being told the wrong
            # thing about why the forecast changed.
            recovered_phrase = (
                f"{total_newly_usable} poll"
                f"{'s' if total_newly_usable != 1 else ''} that "
                f"{'were' if total_newly_usable != 1 else 'was'} already "
                "published but had not been counted until now"
            )
            poll_phrase = (
                recovered_phrase
                if not total_new_polls
                else f"{poll_phrase}, plus {recovered_phrase}"
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
            f"Democratic chances of {which} control {movement} on {poll_phrase}. "
            f"Projected seats {prev_chamber['dem_seats']['mean']:.1f} → "
            f"{cur_chamber['dem_seats']['mean']:.1f} ({_fmt_signed(seats_delta)})."
        )

    body: list[str] = []

    if model_changed:
        headline = (
            f"**The model changed in this run.** Democratic chances of {which} "
            f"control are {_fmt_pct(cur_chamber['dem_control_prob'])}, against "
            f"{_fmt_pct(prev_chamber['dem_control_prob'])} on "
            f"{previous['run_date']} — but the two runs were produced by "
            f"different models, so the difference is not a change in the race."
        )
        body.append(
            "Movement below reflects the model revision as well as any new "
            "polling, and the two cannot be separated from a single comparison. "
            "Day-over-day attribution resumes with the next run."
        )
        body.append("")

    cur_gb = current["national"]["generic_ballot"]["dem_margin_median"]
    prev_gb = previous["national"]["generic_ballot"]["dem_margin_median"]
    if abs(cur_gb - prev_gb) >= 0.1:
        body.append(
            f"**National environment.** The generic ballot moved "
            f"{_fmt_margin(prev_gb)} → {_fmt_margin(cur_gb)} "
            f"({_fmt_signed(cur_gb - prev_gb)}). Because every race is geared to the "
            f"national environment by its own elasticity, this shifts all "
            f"{len(current.get('races', [])) or 35} contests "
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


def previous_payload(
    run_date: date, runs_dir: Path | None = None, chamber: str = "senate"
) -> dict | None:
    """The most recent archived forecast for this chamber before ``run_date``.

    Per chamber, not per date: a dated run directory can hold a Senate archive
    and no House one, because the House was added to the pipeline part-way
    through the cycle. Scanning for the chamber's own file means the first House
    commentary correctly reports no previous run instead of diffing the House
    against the Senate.
    """
    runs_dir = runs_dir or paths.RUNS_DIR
    if not runs_dir.exists():
        return None

    name = paths.chamber_filename("forecast", chamber)
    candidates = sorted(
        p for p in runs_dir.iterdir()
        if p.is_dir() and (p / name).exists() and p.name < run_date.isoformat()
    )
    if not candidates:
        return None
    return json.loads((candidates[-1] / name).read_text(encoding="utf-8"))


#: Changelog per chamber. Two files rather than one interleaved by date,
#: because the chambers are refit by separate runs and a merged file would read
#: as a single narrative jumping between them mid-paragraph.
CHANGELOG_FILE = {
    "senate": "CHANGELOG.md",
    "house": "CHANGELOG_house.md",
}


def write_commentary(
    commentary: Commentary,
    site_data_dir: Path | None = None,
    changelog_path: Path | None = None,
    chamber: str = "senate",
) -> tuple[Path, Path]:
    """Append the entry to the chamber's feed and prepend it to its changelog."""
    site_data_dir = site_data_dir or paths.SITE_DATA_DIR
    site_data_dir.mkdir(parents=True, exist_ok=True)
    changelog_path = changelog_path or paths.REPO_ROOT / CHANGELOG_FILE[chamber]

    # --- structured feed for the dashboard --------------------------------
    feed_path = site_data_dir / paths.chamber_filename("commentary", chamber)
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
        f"# {chamber.capitalize()} forecast changelog\n\n"
        "Automatically generated after each model run. Newest entries first.\n\n"
    )
    sections = [e["markdown"] for e in entries]
    changelog_path.write_text(header + "\n---\n\n".join(sections), encoding="utf-8")

    return feed_path, changelog_path
