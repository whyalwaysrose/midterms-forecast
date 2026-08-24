"""Candidate name → party resolution.

The VoteHub feed identifies poll answers only by candidate name. This module
turns those names into party sides using ``config/candidates_senate_2026.yaml``.

The rule everywhere is *fail closed*: an unknown name resolves to ``other`` and
is excluded from the two-party share. It is never guessed at.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import paths

log = logging.getLogger(__name__)

# Party sides used downstream. "I" exists only so an independent can be mapped
# onto a side explicitly via `independent_counts_as`.
SIDE_D = "D"
SIDE_R = "R"
SIDE_OTHER = "other"


@dataclass
class Roster:
    """Per-race candidate → side lookup."""

    #: race id -> (normalised candidate name -> side)
    by_race: Mapping[str, Mapping[str, str]]
    #: normalised misspelling -> canonical name
    aliases: Mapping[str, str]
    #: generic placeholder ("Dem") -> side
    generic_labels: Mapping[str, str]
    #: race ids where an independent is counted on a major-party side
    independent_notes: Mapping[str, str] = field(default_factory=dict)
    #: race id -> {folded key: original spelling}, so the dashboard can show
    #: "Jon Ossoff" rather than the lookup key "jon ossoff".
    display_names: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @staticmethod
    def _norm(name: str) -> str:
        """Case- and whitespace-insensitive key for a candidate name."""
        return " ".join(str(name).split()).casefold()

    @classmethod
    def load(cls, path: Path | None = None) -> Roster:
        path = path or paths.CONFIG_DIR / "candidates_senate_2026.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        aliases = {cls._norm(k): v for k, v in (raw.get("aliases") or {}).items()}
        generic = {cls._norm(k): v for k, v in (raw.get("generic_labels") or {}).items()}

        by_race: dict[str, dict[str, str]] = defaultdict(dict)
        display: dict[str, dict[str, str]] = defaultdict(dict)
        independent_notes: dict[str, str] = {}

        for race_id, entry in (raw.get("races") or {}).items():
            entry = entry or {}
            independent_side = entry.get("independent_counts_as")

            for side_key, side in (("D", SIDE_D), ("R", SIDE_R), ("other", SIDE_OTHER)):
                for name in entry.get(side_key) or []:
                    by_race[race_id][cls._norm(name)] = side
                    display[race_id][cls._norm(name)] = name

            for name in entry.get("I") or []:
                if independent_side in (SIDE_D, SIDE_R):
                    by_race[race_id][cls._norm(name)] = independent_side
                    display[race_id][cls._norm(name)] = name
                    independent_notes[race_id] = (
                        f"{name} is an independent, counted on the "
                        f"{'Democratic' if independent_side == SIDE_D else 'Republican'} "
                        f"side for chamber-control arithmetic."
                    )
                else:
                    by_race[race_id][cls._norm(name)] = SIDE_OTHER

        return cls(
            by_race=dict(by_race),
            aliases=aliases,
            generic_labels=generic,
            independent_notes=independent_notes,
            display_names={k: dict(v) for k, v in display.items()},
        )

    def resolve(self, race_id: str, choice: str) -> str:
        """Side for one poll answer: ``"D"``, ``"R"`` or ``"other"``."""
        key = self._norm(choice)
        key = self._norm(self.aliases.get(key, key))

        if key in self.generic_labels:
            return self.generic_labels[key]

        return self.by_race.get(race_id, {}).get(key, SIDE_OTHER)

    def unknown_names(
        self, race_id: str, choices: list[str]
    ) -> list[str]:
        """Names in ``choices`` this roster does not classify at all.

        Used by the audit command; a name explicitly listed under ``other`` is
        classified, so it is not reported here.
        """
        known = self.by_race.get(race_id, {})
        unknown = []
        for choice in choices:
            key = self._norm(choice)
            key = self._norm(self.aliases.get(key, key))
            if key in self.generic_labels or key in known:
                continue
            unknown.append(choice)
        return unknown
