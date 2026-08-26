"""District-level presidential lean for all 435 seats, from precinct returns.

This is the input a House forecast needs and the thing that blocked it. Every
district-level source was either unlicensed (ElectIndex, Daily Kos) or an
approximation built by apportioning counties across district lines.

Precinct data solves it, which is how everyone else does this. MIT Election Data
and Science Lab publishes 2024 precinct returns on Harvard Dataverse under
CC0 1.0 -- a public-domain dedication. A presidential row carries no district,
because MEDSL's `district` field is the *office's* district and a presidential
race is statewide; but the same precinct also reports its US House race, and
that row does. Joining the two on (state, county, precinct) gives every
presidential vote a congressional district, with no shapefiles and no
apportionment.

Both files are streamed rather than loaded: the presidential one is 400 MB and
the House one 186 MB, and neither needs to be in memory at once.

The output is 435 rows, vendored, so nobody has to repeat the download.

Usage:
    python scripts/build_house_lean.py --work-dir <somewhere-with-600MB>
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "history" / "house_district_lean_2024.csv"

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"
PRESIDENT_FILE = 13730901  # PRESIDENT_precinct_general.csv, doi:10.7910/DVN/XDJYKC
HOUSE_FILE = 13731101      # HOUSE_precinct_general.csv,     doi:10.7910/DVN/USBYR4

#: States with a single at-large district. MEDSL records these as `STATEWIDE`
#: rather than `001`.
#:
#: They are numbered `-01`, not `-AL`. "AL" is the usual political shorthand,
#: but the poll feed calls Alaska's seat `2026 AK-01` and the race config keys
#: off the same string -- and a second naming convention meant six at-large
#: states silently fell back to a state average they already were.
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}

csv.field_size_limit(10_000_000)


def download(file_id: int, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  using cached {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...", flush=True)
    # Dataverse answers urllib's default User-Agent with 403.
    request = urllib.request.Request(
        f"{DATAVERSE}/{file_id}",
        headers={"User-Agent": "midterms-forecast (github.com/whyalwaysrose)"},
    )
    with urllib.request.urlopen(request, timeout=600) as response, \
            dest.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.0f} MB")
    return dest


def rows(path: Path):
    """Stream a MEDSL precinct file, whatever delimiter it uses."""
    with path.open(encoding="utf-8", newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        yield from csv.DictReader(fh, delimiter=delimiter)


def precinct_key(row: dict) -> tuple[str, str, str]:
    """What identifies a precinct across two files.

    State is part of it: county FIPS is unique nationally, but a precinct name
    like "PRECINCT 70" certainly is not, and a missing FIPS would otherwise let
    two states collide.

    Case and surrounding space are normalised because the two files disagree
    about them. Maine writes `UNION` in the House file and `Union` in the
    presidential one, which cost 115 of its 512 precincts and lost ME-01
    entirely. Two precincts in one county differing only in case would be a
    collision, but no state does that -- and a state that did would show up as
    a district whose vote total is implausibly large.
    """
    return (
        row.get("state_po", "").strip().upper(),
        row.get("county_fips", "").strip(),
        row.get("precinct", "").strip().upper(),
    )


def district_code(state: str, district: str) -> str | None:
    """`AZ-06` from a state and MEDSL's zero-padded district field."""
    district = (district or "").strip().upper()
    if not district:
        return None
    if district in {"STATEWIDE", "AT-LARGE"}:
        # Only a genuine at-large state has one district. Rhode Island files
        # some rows as STATEWIDE despite having two, and mapping those to RI-AL
        # invents a district that does not exist.
        return f"{state}-01" if state in AT_LARGE else None
    if state in AT_LARGE:
        return f"{state}-01"
    if not district.isdigit():
        return None
    return f"{state}-{int(district):02d}"


def parse_votes(value: object) -> int | None:
    """Vote count, whatever shape the state filed it in.

    Arizona and Indiana write `316.0`, most states write `316`, and small
    jurisdictions write `*` where a count is suppressed to avoid identifying a
    voter. `int("316.0")` raises, which silently dropped both states -- 18
    districts and 5.7 million votes -- because the failure looked exactly like
    a precinct with no House race.
    """
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def party_side(row: dict) -> str | None:
    """"DEMOCRAT" or "REPUBLICAN" for a row, or None.

    `party_simplified` is the right field and works for 49 states. Nebraska
    labels every presidential row NONPARTISAN -- its ballot does not print party
    for president -- which silently excluded the entire state and all three of
    its districts. Its `party_detailed` is correct, so that is the fallback.

    Matched on the leading token because fusion states write
    "DEMOCRAT / WORKING FAMILIES"; those states set `party_simplified` properly
    anyway, so this only has to be right, not clever.
    """
    simple = (row.get("party_simplified") or "").strip().upper()
    if simple in ("DEMOCRAT", "REPUBLICAN"):
        return simple
    detailed = (row.get("party_detailed") or "").strip().upper()
    for party in ("DEMOCRAT", "REPUBLICAN"):
        if detailed.startswith(party):
            return party
    return None


def build_district_map(house_path: Path) -> tuple[dict, dict]:
    """How each precinct's votes divide between districts, from its House race.

    A share per district rather than a single assignment. Two reasons, and the
    first is not obvious from the data dictionary:

    Ohio files **every district for every precinct**, nearly all with zero
    votes -- 8,878 precincts each carrying rows for all 15 districts. Treating
    "appears under a district" as "belongs to it" made every Ohio precinct look
    split fifteen ways, and dropping ambiguous precincts then discarded the
    whole state: 5.7 million votes, and Ohio missing from the output entirely.
    Weighting by votes cast makes the empty rows contribute nothing on their own.

    Genuinely split precincts then fall out for free. A precinct straddling two
    districts casts House votes in both, so its presidential votes divide in the
    same proportion -- which is better than dropping it and much better than
    giving all of them to whichever district happens to be larger.
    """
    votes_by_district: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    winners: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows(house_path):
        if row.get("office") != "US HOUSE":
            continue
        code = district_code(row.get("state_po", ""), row.get("district", ""))
        if code is None:
            continue
        votes = parse_votes(row.get("votes"))
        if votes is None:
            continue
        if votes > 0:
            votes_by_district[precinct_key(row)][code] += votes
        # The 2024 winner's party is who holds the seat now, which the
        # fundamentals prior needs.
        party = party_side(row)
        if party:
            winners[code][party] += votes

    shares: dict[tuple[str, str, str], dict[str, float]] = {}
    split = 0
    for key, districts in votes_by_district.items():
        total = sum(districts.values())
        if total <= 0:
            continue
        if len(districts) > 1:
            split += 1
        shares[key] = {code: n / total for code, n in districts.items()}

    print(f"  {len(shares):,} precincts with House votes, "
          f"{split:,} spanning more than one district")
    return shares, winners


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="scratch space for the two source files (~600 MB)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    print("Sources (both CC0 1.0, MIT Election Data and Science Lab):")
    house = download(HOUSE_FILE, args.work_dir / "HOUSE_precinct_general.csv")
    president = download(PRESIDENT_FILE, args.work_dir / "PRESIDENT_precinct_general.csv")

    print("\nMapping precincts to districts via their House race:")
    shares, house_votes = build_district_map(house)

    print("\nAggregating presidential votes into those districts:")
    tally: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    placed = lost = 0
    by_state_lost: dict[str, int] = defaultdict(int)

    for row in rows(president):
        if row.get("office") != "US PRESIDENT":
            continue
        party = party_side(row)
        if party is None:
            continue
        votes = parse_votes(row.get("votes"))
        if votes is None:
            continue
        precinct_shares = shares.get(precinct_key(row))
        if not precinct_shares:
            lost += votes
            by_state_lost[row.get("state_po", "??")] += votes
            continue
        placed += votes
        for code, share in precinct_shares.items():
            tally[code][party] += votes * share

    total = placed + lost
    print(f"  placed {placed:,} of {total:,} two-party presidential votes "
          f"({100 * placed / total:.2f}%)")

    print("\nStates losing the most votes (precincts with no usable House race):")
    for state, missing in sorted(by_state_lost.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {state}  {missing:>10,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "district", "pres_2024_dem_two_party", "pres_2024_dem_votes",
            "pres_2024_rep_votes", "house_2024_holder",
        ])
        for code in sorted(tally):
            dem, rep = tally[code]["DEMOCRAT"], tally[code]["REPUBLICAN"]
            if dem + rep == 0:
                continue
            seat = house_votes.get(code, {})
            holder = ""
            if seat:
                holder = "D" if seat.get("DEMOCRAT", 0) >= seat.get("REPUBLICAN", 0) else "R"
            # Vote counts are fractional after proportional allocation;
            # rounded for the file, but the share is computed before rounding.
            writer.writerow([
                code, f"{dem / (dem + rep):.6f}", round(dem), round(rep), holder,
            ])

    print(f"\nwrote {args.out} ({len(tally)} districts)")
    if len(tally) < 400:
        print("  WARNING: fewer than 400 districts. Some state's precinct ids "
              "probably differ between the two files, which places nothing "
              "rather than erroring.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
