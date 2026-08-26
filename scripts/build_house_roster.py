"""Candidate → party roster for the polled House districts, from FEC filings.

VoteHub returns poll answers as bare names -- `{"choice": "French Hill"}` --
with no party, so the model cannot tell which side is which without being told.
The Senate roster states that by hand for 35 races. The House has 435, which is
not a hand-editable number.

It does not have to be. Only districts that are actually polled need a roster
at all: an unpolled district has no names to classify and falls back to its
fundamentals prior. That is 40 districts and 114 names in the current feed, and
every one of those people has filed with the FEC, which publishes name, party
and district as a public-domain bulk file.

So this generates the roster instead of asking anyone to type it, and refuses
to guess. A name that matches no filing, or matches filings from more than one
party, is written into the file as a comment rather than assigned -- exactly
the fail-closed behaviour the Senate roster has, where a wrong party assignment
would corrupt a district's whole polling history while an unclassified name
only loses that poll.

Usage:
    python scripts/build_house_roster.py --fec weball26.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "config" / "candidates_house_2026.yaml"
VOTEHUB = "https://api.votehub.com/polls?poll_type=us-representative&limit=2000"

# weball column positions (no header row).
CAND_ID, CAND_NAME, CAND_ICI, _PTY, PARTY = 0, 1, 2, 3, 4
CAND_STATE, CAND_DISTRICT = 18, 19

DEM = {"DEM", "DFL"}
REP = {"REP", "GOP"}

#: Names a pollster uses for "whoever the party nominates".
PLACEHOLDERS = {"dem", "rep", "democrat", "democratic", "republican", "gop"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "mr", "mrs", "ms", "dr"}


def normalise(name: str) -> str:
    """Lowercase, accent-stripped, letters and spaces only.

    Accents are decomposed rather than deleted: without this "Giménez" became
    "gimnez" while the FEC's "GIMENEZ" became "gimenez", and the two never met.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", stripped.strip().lower())


def parts(name: str) -> tuple[str, str]:
    """(surname, first name) from either "LAST, FIRST M" or "First Last".

    The comma is read *before* normalising. `normalise` strips punctuation, so
    normalising first turned "JONES, MARCUS" into "jones marcus" -- no comma
    left -- and the "First Last" branch then read Marcus as the surname. Every
    FEC name came out reversed and 113 of 114 lookups failed.

    Only the last word of a surname is kept, so "DE LA CRUZ" and
    "Monica De La Cruz" meet at "cruz". Within a single district that is
    specific enough, and a same-surname clash across parties is caught by the
    caller rather than resolved here.
    """
    raw = name.strip()
    if "," in raw:
        surname_raw, _, rest = raw.partition(",")
        surname_words = normalise(surname_raw).split()
        rest_words = [w for w in normalise(rest).split() if w not in SUFFIXES]
        surname = surname_words[-1] if surname_words else ""
        return surname, (rest_words[0] if rest_words else "")

    words = [w for w in normalise(raw).split() if w not in SUFFIXES]
    if not words:
        return normalise(raw), ""
    return words[-1], words[0]


def load_filings(path: Path) -> dict[str, list[dict]]:
    """House candidates by district code, from the FEC bulk file."""
    with zipfile.ZipFile(path) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", "replace")

    by_district: dict[str, list[dict]] = defaultdict(list)
    for line in text.splitlines():
        row = line.split("|")
        if len(row) <= CAND_DISTRICT or not row[CAND_ID].startswith("H"):
            continue
        state = row[CAND_STATE].strip().upper()
        raw_district = row[CAND_DISTRICT].strip()
        if not state or not raw_district.isdigit():
            continue
        # FEC writes an at-large seat as district 00; VoteHub polls call it
        # -01. The poll feed is what has to be joined against and its naming is
        # not negotiable, so -01 wins and at-large seats are numbered like any
        # other. Alaska's whole slate went unclassified on this.
        code = f"{state}-{max(1, int(raw_district)):02d}"
        party = row[PARTY].strip().upper()
        side = "D" if party in DEM else "R" if party in REP else "other"
        surname, first = parts(row[CAND_NAME])
        by_district[code].append({
            "name": row[CAND_NAME].strip(), "side": side, "party": party,
            "surname": surname, "first": first,
        })
    return by_district


def poll_names() -> dict[str, set[str]]:
    """Every candidate name the current feed uses, by district."""
    request = urllib.request.Request(
        VOTEHUB, headers={"User-Agent": "midterms-forecast"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode())
    polls = payload if isinstance(payload, list) else payload.get("polls", [])

    found: dict[str, set[str]] = defaultdict(set)
    for poll in polls:
        subject = str(poll.get("subject", ""))
        if not subject.startswith("2026"):
            continue
        district = subject.split()[-1].upper()
        for answer in poll.get("answers") or []:
            choice = str(answer.get("choice", "")).strip()
            if choice:
                found[district].add(choice)
    return found


def resolve(name: str, filings: list[dict]) -> tuple[str | None, str]:
    """Which side a name belongs to, and why. None means unresolved."""
    if normalise(name) in PLACEHOLDERS:
        return None, "generic placeholder, not a person"

    surname, first = parts(name)
    matches = [f for f in filings if f["surname"] == surname]
    if not matches:
        # Some FEC records have the name the wrong way round -- NY-01's
        # incumbent is filed as "NICK, LALOTA" rather than "LALOTA, NICK" --
        # so the parsed surname is really the first name. Try it the other way
        # before giving up, and only when the strict match found nothing.
        matches = [f for f in filings if f["first"] == surname and f["surname"] == first]
    if not matches:
        return None, "no FEC filing with that surname in this district"

    if len({f["side"] for f in matches}) > 1:
        # Same surname, different parties -- try the first name before giving up.
        narrowed = [f for f in matches if f["first"] == first]
        if len({f["side"] for f in narrowed}) == 1 and narrowed:
            matches = narrowed
        else:
            parties = sorted({f["party"] for f in matches})
            return None, f"surname matches filings from {', '.join(parties)}"

    side = matches[0]["side"]
    if side == "other":
        return "other", f"filed as {matches[0]['party']}"
    return side, f"FEC: {matches[0]['name']} ({matches[0]['party']})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec", type=Path, required=True, help="weball26.zip")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    filings = load_filings(args.fec)
    print(f"{sum(len(v) for v in filings.values())} House filings "
          f"across {len(filings)} districts")

    polled = poll_names()
    print(f"{len(polled)} districts polled, "
          f"{sum(len(v) for v in polled.values())} names to classify\n")

    lines = [
        "# =============================================================================",
        "# 2026 U.S. House — candidate → party roster",
        "# =============================================================================",
        "# GENERATED by scripts/build_house_roster.py from FEC bulk filings.",
        "#",
        "# Only districts that are actually polled appear here. An unpolled district",
        "# has no names to classify and falls back to its fundamentals prior, so a",
        "# roster entry for it would be dead weight -- 40 districts, not 435.",
        "#",
        "# Party comes from each candidate's own FEC filing, which is public domain,",
        "# rather than from anyone's judgement. Names that match no filing, or match",
        "# filings from more than one party, are listed as comments and left",
        "# unclassified: their polls are skipped. That is the same fail-closed rule",
        "# the Senate roster uses, and for the same reason -- a wrong party ruins a",
        "# district's whole polling history, an unclassified name only loses a poll.",
        "#",
        "# Regenerate rather than hand-edit, unless you are recording something the",
        "# FEC does not know.",
        "",
        "generic_labels:",
        "  Dem: D",
        "  Rep: R",
        "",
        "races:",
    ]

    resolved = unresolved = 0
    for district in sorted(polled):
        sides: dict[str, list[str]] = {"D": [], "R": [], "other": []}
        notes: list[str] = []
        for name in sorted(polled[district]):
            if normalise(name) in PLACEHOLDERS:
                continue  # handled once by generic_labels, not per district
            side, why = resolve(name, filings.get(district, []))
            if side is None:
                notes.append(f"    # UNCLASSIFIED {name!r} — {why}")
                unresolved += 1
            else:
                sides[side].append(name)
                resolved += 1

        lines.append(f"  house-2026-{district}:")
        for key in ("D", "R", "other"):
            if sides[key]:
                inner = ", ".join(f"{n}" for n in sides[key])
                lines.append(f"    {key}: [{inner}]")
        lines.extend(notes)

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    print(f"resolved {resolved} names, left {unresolved} unclassified")
    print(f"wrote {args.out}")
    if unresolved:
        print("\nUnclassified names are comments in the file; their polls are skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
