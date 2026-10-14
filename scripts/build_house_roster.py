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

The FEC file is fetched automatically and cached under data/raw, so this can
run unattended in the daily job. Pass --fec to use a local copy instead.

Usage:
    python scripts/build_house_roster.py
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
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "config" / "candidates_house_2026.yaml"
VOTEHUB = "https://api.votehub.com/polls?poll_type=us-representative&limit=2000"
FEC_BULK = "https://www.fec.gov/files/bulk-downloads/2026/weball26.zip"
FEC_CACHE = REPO / "data" / "raw" / "weball26.zip"

#: The bulk file names its one member for the cycle.
FEC_MEMBER = "weball26.txt"

# weball column positions (no header row).
CAND_ID, CAND_NAME, CAND_ICI, _PTY, PARTY = 0, 1, 2, 3, 4
CAND_STATE, CAND_DISTRICT = 18, 19

DEM = {"DEM", "DFL"}
REP = {"REP", "GOP"}

#: Names a pollster uses for "whoever the party nominates".
PLACEHOLDERS = {"dem", "rep", "democrat", "democratic", "republican", "gop"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "mr", "mrs", "ms", "dr"}

#: Misspellings a pollster has published, mapped to the name the FEC filed.
#:
#: The one thing in this file the FEC cannot tell us, and therefore the one
#: thing that has to live in the script rather than be hand-added to the output
#: -- a regeneration overwrites the roster wholesale, so a hand-edited alias
#: would survive exactly until the next daily run.
#:
#: The bar is a published typo, confirmed against the filing. Not a nickname, a
#: married name or a shortened first name: the matcher already handles those by
#: surname, and anything needing judgement belongs in the UNCLASSIFIED comments
#: where a person can see it.
ALIASES = {
    # PA-10, fielded 2026-04-23: Stelson filed with the FEC, Stetson was typed.
    "Janelle Stetson": "Janelle Stelson",
}


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


def fetch_filings(cache: Path = FEC_CACHE) -> Path:
    """Download the FEC bulk candidate file, or reuse today's cached copy.

    Cached by date, like the poll snapshots: filings change when someone files,
    which is daily during a cycle, so a stale copy is the whole bug this is
    meant to remove.

    What arrives is checked before it replaces the cache, because the FEC
    answers a wrong bulk path with an HTML page and a 200. Written through, that
    fails much later and much less legibly, inside `load_filings`, as "not a zip
    file". Size cannot tell the two apart -- the 2026 file is only ~190 KB -- so
    the test is whether the archive opens and holds the member it should.
    """
    if cache.exists():
        age = date.today() - date.fromtimestamp(cache.stat().st_mtime)
        if age.days == 0:
            print(f"using cached {cache.name} (fetched today)")
            return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".zip.part")
    request = urllib.request.Request(
        FEC_BULK, headers={"User-Agent": "midterms-forecast"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        tmp.write_bytes(response.read())

    size = tmp.stat().st_size
    try:
        with zipfile.ZipFile(tmp) as archive:
            members = archive.namelist()
            if FEC_MEMBER not in members:
                raise RuntimeError(
                    f"archive holds {members or 'nothing'}, not {FEC_MEMBER}"
                )
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"{bad} is corrupt")
    except (zipfile.BadZipFile, RuntimeError) as exc:
        tmp.unlink()
        raise RuntimeError(
            f"{FEC_BULK} did not return the bulk file ({size:,} bytes): {exc}. "
            "The FEC answers a wrong path with an HTML page and a 200, so "
            "this is checked here rather than surfacing as a parse error later."
        ) from None
    tmp.replace(cache)
    print(f"fetched {FEC_BULK} ({size:,} bytes) -> {cache}")
    return cache


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


#: A regeneration may legitimately shrink a little -- a pollster stops testing
#: a district, a name is corrected -- but not by much. Anything past this is a
#: failure upstream, not news: an empty VoteHub response, or an FEC file for the
#: wrong cycle, both of which would otherwise write a near-empty roster and cost
#: the House every district poll with no error anywhere.
SHRINK_FLOOR = 0.8


def existing_resolved(path: Path) -> int:
    """How many names the roster on disk classifies, for the shrink guard.

    Counted from the rendered file rather than by importing the loader, so the
    guard holds even if the roster schema and this script drift apart.
    """
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("D:", "R:", "other:")) and "[" in stripped:
            inner = stripped[stripped.index("[") + 1:stripped.rindex("]")]
            total += len([n for n in inner.split(",") if n.strip()])
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fec", type=Path,
        help="local weball26.zip; omitted, it is fetched and cached",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--allow-shrink", action="store_true",
        help="write even if far fewer names resolve than are on disk",
    )
    args = parser.parse_args()

    filings = load_filings(args.fec or fetch_filings())
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
        f"# built: {date.today().isoformat()}",
        "#",
        "# Only districts that are actually polled appear here. An unpolled district",
        "# has no names to classify and falls back to its fundamentals prior, so a",
        f"# roster entry for it would be dead weight -- {len(polled)} districts, not 435.",
        "#",
        "# Party comes from each candidate's own FEC filing, which is public domain,",
        "# rather than from anyone's judgement. Names that match no filing, or match",
        "# filings from more than one party, are listed as comments and left",
        "# unclassified: their polls are skipped. That is the same fail-closed rule",
        "# the Senate roster uses, and for the same reason -- a wrong party ruins a",
        "# district's whole polling history, an unclassified name only loses a poll.",
        "#",
        "# Regenerate rather than hand-edit. The one thing the FEC cannot tell us",
        "# is a pollster's misspelling, and those live in ALIASES in the script",
        "# above -- added to this file by hand they would last until tomorrow.",
        "",
        "aliases:",
        *(f'  "{wrong}": "{right}"' for wrong, right in sorted(ALIASES.items())),
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
            if name in ALIASES:
                # The loader redirects it to the filed spelling, which is
                # classified in its own right. Emitting an UNCLASSIFIED
                # comment here would report a name that is in fact handled.
                continue
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

    before = existing_resolved(args.out)
    if before and resolved < before * SHRINK_FLOOR and not args.allow_shrink:
        print(
            f"::error::refusing to write: the roster on disk classifies {before} "
            f"names and this run classified only {resolved}. Something upstream "
            "is wrong -- an empty poll feed, or an FEC file for the wrong cycle. "
            "Pass --allow-shrink if the drop is genuine.",
            file=sys.stderr,
        )
        return 1

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    print(f"resolved {resolved} names, left {unresolved} unclassified"
          + (f" (was {before} resolved)" if before else ""))
    print(f"wrote {args.out}")
    if unresolved:
        print("\nUnclassified names are comments in the file; their polls are skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
