"""Generate `config/races_house_2026.yaml` — all 435 districts.

The Senate config is hand-written and reviewed, which is the right call for 35
contests. 435 is not, so this builds it from two files that are already in the
repository or already public domain:

* `data/history/house_district_lean_2024.csv` — presidential lean per district,
  derived from CC0 precinct returns by `build_house_lean.py`.
* the FEC's `weball26.zip` — every 2026 House filer, with party and incumbency.

**Do not use the FEC's incumbency flag.** It was tried and it is wrong, in the
same way it is wrong for the Senate: `CAND_ICI = I` means "has held this office
at some point", not "holds it now". There are 555 House filings carrying the
flag for 435 seats, 108 districts have more than one, and 28 have incumbents of
both parties -- Kyrsten Sinema is still flagged in AZ-09, Lucille Roybal-Allard
in CA-40 having left in 2023, Rick Renzi in AZ-01 having left in 2009. It gave
199 D / 232 R against an actual 215 / 220.

Current membership comes instead from `unitedstates/congress-legislators`, the
canonical open dataset, which is **CC0** and lists every sitting member with
their state, district and party. Whether that member is running again is a
separate question, answered by whether they appear among the 2026 FEC filers --
the filing itself is reliable even though the incumbency flag on it is not.

**2020 has no usable equivalent.** Presidential results for 2020 are under the
pre-2022 district lines, so they describe different places. Rather than blend
across a redistricting, the 2020 field repeats the 2024 value, which makes the
blend in `fundamentals.py` a no-op whatever weights it is given. That is stated
in the generated file rather than left for someone to discover.

Usage:
    python scripts/build_house_races.py --fec weball26.zip
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEAN = REPO / "data" / "history" / "house_district_lean_2024.csv"
OUT = REPO / "config" / "races_house_2026.yaml"

CAND_ID, CAND_NAME, CAND_ICI, _PTY, PARTY = 0, 1, 2, 3, 4
CAND_STATE, CAND_DISTRICT = 18, 19

DEM = {"DEM", "DFL"}
REP = {"REP", "GOP"}

#: Region per state, matching the Senate config so the correlation kernel keeps
#: one vocabulary across chambers.
REGION = {
    "CT": "northeast", "ME": "northeast", "MA": "northeast", "NH": "northeast",
    "NJ": "northeast", "NY": "northeast", "PA": "northeast", "RI": "northeast",
    "VT": "northeast", "DE": "northeast", "MD": "northeast",
    "IL": "midwest", "IN": "midwest", "IA": "midwest", "KS": "midwest",
    "MI": "midwest", "MN": "midwest", "MO": "midwest", "NE": "midwest",
    "ND": "midwest", "OH": "midwest", "SD": "midwest", "WI": "midwest",
    "AL": "south", "AR": "south", "FL": "south", "GA": "south", "KY": "south",
    "LA": "south", "MS": "south", "NC": "south", "OK": "south", "SC": "south",
    "TN": "south", "TX": "south", "VA": "south", "WV": "south",
    "AK": "west", "AZ": "west", "CA": "west", "CO": "west", "HI": "west",
    "ID": "west", "MT": "west", "NV": "west", "NM": "west", "OR": "west",
    "UT": "west", "WA": "west", "WY": "west",
}

#: Seats per state for the 119th Congress, so a missing district is an error
#: rather than a quietly shorter chamber.
SEATS = {
    "AL": 7, "AK": 1, "AZ": 9, "AR": 4, "CA": 52, "CO": 8, "CT": 5, "DE": 1,
    "FL": 28, "GA": 14, "HI": 2, "ID": 2, "IL": 17, "IN": 9, "IA": 4, "KS": 4,
    "KY": 6, "LA": 6, "ME": 2, "MD": 8, "MA": 9, "MI": 13, "MN": 8, "MS": 4,
    "MO": 8, "MT": 2, "NE": 3, "NV": 4, "NH": 2, "NJ": 12, "NM": 3, "NY": 26,
    "NC": 14, "ND": 1, "OH": 15, "OK": 5, "OR": 6, "PA": 17, "RI": 2, "SC": 7,
    "SD": 1, "TN": 9, "TX": 38, "UT": 4, "VT": 1, "VA": 11, "WA": 10, "WV": 2,
    "WI": 8, "WY": 1,
}


def load_lean() -> dict[str, dict]:
    return {
        row["district"]: row
        for row in csv.DictReader(LEAN.open(encoding="utf-8"))
    }


LEGISLATORS = ("https://unitedstates.github.io/congress-legislators/"
               "legislators-current.csv")
LEGISLATORS_PAST = ("https://unitedstates.github.io/congress-legislators/"
                    "legislators-historical.csv")


def fetch_csv(url: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "midterms-forecast"})
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8", "replace")
    return list(csv.DictReader(text.splitlines()))


def last_holders() -> dict[str, list[tuple[str, str]]]:
    """District -> everyone who has held it, for seats nobody holds now.

    A vacant seat is absent from `legislators-current` entirely, which leaves
    no answer for `incumbent_party` -- and guessing one from the presidential
    lean is precisely the mistake documented below. The historical file records
    each former member's last district and party, which is the actual answer.

    The whole list is returned rather than just the most recent, because the CSV
    carries no term-end date and its row order, while chronological in practice,
    is not documented to be. The caller prints the list so the choice is visible
    in the build log instead of being taken on trust; where every holder of a
    seat is the same party, the ordering does not matter at all.
    """
    past: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in fetch_csv(LEGISLATORS_PAST):
        if row.get("type") != "rep":
            continue
        state = (row.get("state") or "").strip().upper()
        raw = (row.get("district") or "").strip()
        if state not in SEATS or not raw.lstrip("-").isdigit():
            continue
        party = (row.get("party") or "").strip()
        side = "D" if party.startswith("Democrat") else "R" if party.startswith("Republic") else None
        if side:
            code = f"{state}-{max(1, int(raw)):02d}"
            past[code].append(((row.get("last_name") or "").strip(), side))
    return past


def load_sitting_members() -> dict[str, tuple[str, str]]:
    """District -> (party, surname) of the member holding it now.

    From `unitedstates/congress-legislators`, which is CC0 and is what this
    dataset exists for. Fetched rather than vendored because membership changes
    with each resignation and special election, and a stale copy would be worse
    than an occasional network call the daily run does not depend on.
    """
    request = urllib.request.Request(
        LEGISLATORS, headers={"User-Agent": "midterms-forecast"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", "replace")

    members: dict[str, tuple[str, str]] = {}
    for row in csv.DictReader(text.splitlines()):
        if row.get("type") != "rep":
            continue
        state = (row.get("state") or "").strip().upper()
        raw = (row.get("district") or "").strip()
        if state not in SEATS or not raw.lstrip("-").isdigit():
            continue
        # At-large seats are district 0 here and -01 everywhere else.
        code = f"{state}-{max(1, int(raw)):02d}"
        party = (row.get("party") or "").strip()
        side = "D" if party.startswith("Democrat") else "R" if party.startswith("Republic") else None
        if side:
            members[code] = (side, (row.get("last_name") or "").strip().lower())
    return members


def filed_surnames(path: Path) -> dict[str, set[str]]:
    """District -> surnames of everyone who filed for 2026.

    Used only to tell a seat whose member is running again from an open one.
    The filings are reliable; it is the incumbency flag on them that is not.
    """
    with zipfile.ZipFile(path) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", "replace")

    filed: dict[str, set[str]] = defaultdict(set)
    for line in text.splitlines():
        row = line.split("|")
        if len(row) <= CAND_DISTRICT or not row[CAND_ID].startswith("H"):
            continue
        state = row[CAND_STATE].strip().upper()
        raw = row[CAND_DISTRICT].strip()
        if state not in SEATS or not raw.isdigit():
            continue
        surname = row[CAND_NAME].split(",")[0].strip().lower()
        filed[f"{state}-{max(1, int(raw)):02d}"].add(surname)
    return filed


def all_districts() -> list[str]:
    return [
        f"{state}-{n:02d}"
        for state, seats in sorted(SEATS.items())
        for n in range(1, seats + 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec", type=Path, required=True, help="weball26.zip")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    lean = load_lean()
    members = load_sitting_members()
    past = last_holders()
    filed = filed_surnames(args.fec)
    districts = all_districts()
    print(f"{len(districts)} districts expected, {len(lean)} with a measured lean, "
          f"{len(members)} with a sitting member")
    held = Counter(side for side, _ in members.values())
    print(f"  current chamber: D {held['D']}, R {held['R']}, "
          f"vacant/unmatched {len(districts) - len(members)}")

    # A district with no lean of its own falls back to its state's average, so
    # the chamber is never quietly short. Stated per district in the output.
    by_state: dict[str, list[float]] = defaultdict(list)
    for code, row in lean.items():
        by_state[code.split("-")[0]].append(float(row["pres_2024_dem_two_party"]))
    state_mean = {s: sum(v) / len(v) for s, v in by_state.items()}

    entries: list[str] = []
    counts: Counter = Counter()
    fallbacks: list[str] = []
    vacant: list[tuple[str, list]] = []
    unknown: list[str] = []
    counts_open = 0

    for code in districts:
        state = code.split("-")[0]
        row = lean.get(code)
        if row:
            share = float(row["pres_2024_dem_two_party"])
            holder_2024 = row["house_2024_holder"] or ""
            note = ""
        else:
            share = state_mean.get(state, 0.5)
            holder_2024 = ""
            note = "    # NO DISTRICT DATA: uncontested 2024 House race, so its\n" \
                   "    # presidential votes had nothing to join to. Using the\n" \
                   "    # state average as a placeholder."
            fallbacks.append(code)

        sitting = members.get(code)
        if sitting:
            party, surname = sitting
            # Running again if they are among this district's 2026 filers.
            status = "elected" if surname in filed.get(code, set()) else "open"
            counts_open += status == "open"
        elif holder_2024:
            # A vacant seat, or one whose member could not be matched. The 2024
            # result says who held it; the seat is open either way.
            party, status = holder_2024, "open"
            counts_open += 1
        elif past.get(code):
            # Vacant, and with no 2024 House result either -- so ask who held it
            # last. This is the branch FL-20 falls into, and the branch an
            # earlier version got badly wrong.
            #
            # That version guessed the party from `share`, which is exactly the
            # wrong thing to do here: a district reaches this branch only when
            # its House race was uncontested, which is also why it has no
            # measured lean -- so `share` is the state average, and the guess is
            # made from the very placeholder that is missing. Florida's average
            # is 43% Democratic, which is below half, so the script wrote down
            # "Republican-held" for one of the most Democratic districts in the
            # country. Two gaps compounding into a confident false statement.
            party, status = past[code][-1][1], "open"
            counts_open += 1
            vacant.append((code, past[code]))
        else:
            # No sitting member, no 2024 result, and nobody has ever held it
            # under this numbering -- a newly created district that has not yet
            # been filled. Nothing here can answer it, so fail rather than
            # invent: a wrong incumbent_party is worth about three points of
            # margin in the prior and is invisible once written down.
            unknown.append(code)
            party, status = "R", "open"  # placeholder; the run aborts below
            counts_open += 1
        counts[party] += 1

        entries.append(
            f"  - id: house-2026-{code}\n"
            f"    unit: {code}\n"
            f"    name: {code}\n"
            f"    special: false\n"
            f"    incumbent_party: {party}\n"
            f"    incumbent_status: {status}\n"
            f"    pres_2024_dem_two_party: {share:.4f}\n"
            f"    pres_2020_dem_two_party: {share:.4f}\n"
            f"    region: {REGION[state]}\n"
            + (note + "\n" if note else "")
        )

    header = f"""\
# =============================================================================
# 2026 U.S. House — race definitions
# =============================================================================
# GENERATED by scripts/build_house_races.py. Do not hand-edit: regenerate.
#
# The Senate equivalent is written and reviewed by hand, which is right for 35
# contests and not for 435. Everything here comes from two sources:
#
#   pres_2024_dem_two_party   data/history/house_district_lean_2024.csv, built
#                             from CC0 precinct returns (see its README)
#   incumbent_party/status    the FEC's 2026 filings
#
# WHY 2020 REPEATS 2024
# `pres_2020_dem_two_party` is deliberately the same number as 2024. Actual 2020
# results are under the PRE-2022 district lines, so they describe different
# places and blending them would mix two different maps. Repeating the 2024
# value makes the blend in fundamentals.py a no-op whatever weights it is given,
# which is the honest way to say "we only have one cycle here".
#
# WHY FEC INCUMBENCY IS USED HERE AND NOT FOR THE SENATE
# The same CAND_ICI flag is unreliable for the Senate, where it marks sitting
# members who are not on the ballot at all. Every House seat is contested every
# two years, so a 2026 filer flagged `I` really is the sitting member seeking
# re-election. A district with no such filer is an open seat.
# =============================================================================

cycle: 2026
chamber: house
election_date: 2026-11-03

# ---------------------------------------------------------------------------
# Chamber control arithmetic.
#
# Every seat is up, so nothing is held back: 218 of 435 is a majority for
# either party. A 217-217 tie leaves no majority at all and is not resolved by
# a tiebreaker the way the Senate's is -- the Speaker is elected by the chamber
# itself. `tiebreaker_party: none` records that rather than inventing one.
# ---------------------------------------------------------------------------
control:
  total_seats: 435
  seats_not_up:
    D: 0
    R: 0
  seats_up:
    D: {counts['D']}
    R: {counts['R']}
  dem_seats_for_majority: 218
  rep_seats_for_majority: 218
  tiebreaker_party: none

races:
"""

    # Checked before the file is written, not after. A config carrying a made-up
    # incumbent party is worse than no config: it runs, it produces numbers, and
    # the error is a three-point shift in one district's prior that nothing
    # downstream can see.
    if unknown:
        print(f"\nERROR: no holder can be determined for {unknown}. Nothing in "
              f"these sources answers it, and guessing from the presidential "
              f"lean is the mistake this branch exists to prevent. Add them by "
              f"hand, or find a source, before regenerating.")
        print("Nothing was written.")
        return 1

    args.out.write_text(header + "\n".join(entries), encoding="utf-8", newline="")
    print(f"\nseats by current holder: D {counts['D']}, R {counts['R']}")
    print(f"open seats (member not standing again, or seat vacant): "
          f"{counts_open}")
    if fallbacks:
        print(f"districts using a state-average lean: {fallbacks}")

    # Printed, not assumed. The historical file has no term-end date, so the
    # holder is taken as the last row for that district; showing every holder
    # lets a reviewer see whether the ordering mattered.
    for code, holders in vacant:
        names = ", ".join(f"{name} ({side})" for name, side in holders)
        print(f"vacant seat {code}: taking the last of [{names}]")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
