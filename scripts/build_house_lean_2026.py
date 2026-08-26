"""District presidential lean under the districts that will actually be on the
2026 ballot.

`build_house_lean.py` derives this from CC0 precinct returns and is the method
this project would prefer. It cannot be used for 2026, and the reason is not a
bug: it works by joining a precinct's presidential row to that precinct's own
**2024 US House row**, and ten states redrew their congressional boundaries
mid-decade. There was no 2024 election under the new lines, so there is no House
row to join to. The technique is sound and the data no longer exists.

    Texas +5 R,  California +5 D,  Florida +4 R,  Ohio +2 R,  Utah +1 D,
    North Carolina, Missouri, Tennessee, Louisiana, Alabama +1 R each

That is 181 of 435 seats and a net of roughly ten Republican ones. A forecast
built on the old lines is not slightly stale; for those states it is scoring
districts that will not appear on any ballot.

THE SOURCE, AND ITS LICENCE
---------------------------
The Downballot (formerly Daily Kos Elections) publishes 2024 presidential
results for every district under the 2026 maps, which is exactly the table
needed. It is free to read and has **no stated licence** -- the same position as
the prediction-market data: no restriction found, which is not the same as
permission confirmed. That is recorded here rather than glossed, and attribution
is given wherever the numbers appear.

Using it is a deliberate trade. The alternative was a GIS overlay of official
state boundary files onto precinct shapes, which is the dependency the
precinct-join method exists to avoid, and which would take far longer to get
right than the eight weeks left before the election.

IT IS ALSO BETTER THAN OURS WHERE BOTH EXIST
--------------------------------------------
The two sources are compared on the 254 districts whose boundaries did not
change, where they should agree exactly. Thirty-five of forty states agree to
within half a point on average -- which validates both the CC0 join and this
table. Five do not, and in every case the error is ours:

    OR   mean 7.00 pts off, max 29.89   OR-03 is Portland. We had it at 43.9%
                                        Democratic; it is 73.8%.
    WA   mean 2.63 pts off, max 12.03   WA-07 is Seattle.
    NY, IN, NJ   around 1 pt

Oregon and Washington vote entirely by mail and report in aggregated batches
rather than by polling place, so the precinct join had less to work with than it
appeared to. That defect sat undetected in a published forecast until these two
sources were put side by side, which is the argument for keeping this check in
the script rather than doing it once by hand.

So this file supersedes the CC0 one for all 435 districts, not only the redrawn
181. The CC0 file stays in the repository as the cross-check that found the
problem, and as the record of a method that needs no licence at all.

WHY 2020 IS STILL NOT USED
--------------------------
The source carries 2020 results under the new lines for the 254 unchanged
districts and not for the 181 redrawn ones. Using it where available would give
58% of the House a 0.75/0.25 blend and the rest a pure 2024 lean -- two groups
of districts treated differently inside one seat distribution, which is a worse
error than ignoring one cycle consistently. So `pres_2020` repeats `pres_2024`
for every district, exactly as before, and the blend in `fundamentals.py` stays
a no-op for this chamber.

Usage:
    python scripts/build_house_lean_2026.py
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CC0_LEAN = REPO / "data" / "history" / "house_district_lean_2024.csv"
OUT = REPO / "data" / "history" / "house_district_lean_2026.csv"

SHEET_ID = "1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0"
SHEET_GID = "1491069057"
SOURCE_URL = (
    "https://www.the-downballot.com/p/the-downballot-releases-presidential"
)

#: States that redrew congressional boundaries for 2026, with the published net
#: seat effect. Used only to report and to sanity-check, never to adjust a
#: number -- the leans themselves carry the change.
REDRAWN = {
    "TX": "+5 R", "CA": "+5 D", "FL": "+4 R", "OH": "+2 R", "UT": "+1 D",
    "NC": "+1 R", "MO": "+1 R", "TN": "+1 R", "LA": "+1 R", "AL": "+1 R",
}

#: How far two sources may differ on an unchanged district before it is worth
#: a human looking. Below this is rounding and precinct-allocation noise.
DISAGREEMENT_PTS = 2.0


def district_code(raw: str) -> str | None:
    """`AK-AL` and `TX-07` to the `AK-01` / `TX-07` form the config uses."""
    raw = raw.strip().upper()
    if "-" not in raw:
        return None
    state, _, number = raw.partition("-")
    if len(state) != 2 or not state.isalpha():
        return None
    if number == "AL":
        return f"{state}-01"
    if not number.isdigit():
        return None
    return f"{state}-{int(number):02d}"


def parse_votes(value: str) -> int | None:
    try:
        return int(value.replace(",", "").strip())
    except (AttributeError, ValueError):
        return None


def fetch_sheet() -> list[list[str]]:
    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
        f"/export?format=csv&gid={SHEET_GID}"
    )
    print(f"  fetching {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "midterms-forecast"})
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8", "replace")
    return list(csv.reader(text.splitlines()))


def read_downballot(rows: list[list[str]]) -> dict[str, dict]:
    """District -> Harris/Trump 2024 votes, from the published sheet.

    Columns are located by position because the sheet's header spans three rows
    with merged cells -- there is no single header line to key off. Rows that do
    not parse are skipped and counted, so a layout change shows up as a district
    count that is not 435 rather than as silence.
    """
    out: dict[str, dict] = {}
    for row in rows:
        if not row:
            continue
        code = district_code(row[0])
        if code is None or len(row) < 7:
            continue
        dem, rep = parse_votes(row[4]), parse_votes(row[5])
        if dem is None or rep is None or dem + rep <= 0:
            continue
        out[code] = {
            "dem": dem,
            "rep": rep,
            "share": dem / (dem + rep),
            "incumbent": row[1].strip(),
            "party": row[2].strip().strip("()"),
        }
    return out


def cross_check(new: dict[str, dict]) -> None:
    """Compare against the CC0 file on districts whose lines did not change.

    Kept in the script rather than done once by hand, because it is what found
    the Oregon and Washington errors described above -- and because the next
    person to regenerate this file deserves the same warning if a new source
    disagrees with a method that needs no licence.
    """
    if not CC0_LEAN.exists():
        print("  no CC0 file to cross-check against; skipping")
        return

    old = {
        row["district"]: float(row["pres_2024_dem_two_party"])
        for row in csv.DictReader(CC0_LEAN.open(encoding="utf-8"))
    }
    by_state: dict[str, list[float]] = defaultdict(list)
    flagged: list[tuple[str, float, float]] = []
    for code, entry in new.items():
        if code[:2] in REDRAWN or code not in old:
            continue
        gap = (entry["share"] - old[code]) * 100
        by_state[code[:2]].append(abs(gap))
        if abs(gap) >= DISAGREEMENT_PTS:
            flagged.append((code, old[code], entry["share"]))

    agree = sum(1 for v in by_state.values() if sum(v) / len(v) < 0.5)
    print(f"\nCross-check against the CC0 precinct join, on the "
          f"{sum(len(v) for v in by_state.values())} districts whose lines did not change:")
    print(f"  {agree} of {len(by_state)} states agree to within 0.5 points on average")
    if flagged:
        print(f"  {len(flagged)} districts differ by {DISAGREEMENT_PTS}+ points:")
        for code, was, now in sorted(flagged, key=lambda f: -abs(f[2] - f[1]))[:8]:
            print(f"     {code}  CC0 join {100 * was:5.1f}%  this source {100 * now:5.1f}%"
                  f"   ({100 * (now - was):+5.1f})")
        print("  Where these have been checked, the error was in the CC0 join:"
              " Oregon and Washington vote by mail and do not report by precinct.")


def main() -> int:
    print("Building 2026-line district presidential lean.")
    print(f"  source: The Downballot, {SOURCE_URL}")
    print("  licence: none stated. Free to read; no restriction found, which is")
    print("           not the same as permission confirmed. Attributed in the")
    print("           output file and in docs/DATA_SOURCES.md.\n")

    new = read_downballot(fetch_sheet())
    print(f"  parsed {len(new)} districts")
    if len(new) != 435:
        print(f"  ERROR: expected 435 districts, got {len(new)}. The sheet's "
              f"layout has probably changed; check the column positions in "
              f"read_downballot() before trusting anything here.")
        return 1

    cross_check(new)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "district", "pres_2024_dem_two_party", "pres_2024_dem_votes",
            "pres_2024_rep_votes", "redrawn_for_2026", "incumbent", "incumbent_party",
        ])
        for code in sorted(new):
            entry = new[code]
            writer.writerow([
                code, f"{entry['share']:.6f}", entry["dem"], entry["rep"],
                "yes" if code[:2] in REDRAWN else "no",
                entry["incumbent"], entry["party"],
            ])

    held = Counter(entry["party"] for entry in new.values())
    print(f"\n  incumbents by party under the new lines: {dict(sorted(held.items()))}")

    # Anything that is not D or R needs a human decision before it can enter
    # chamber arithmetic, the same way Nebraska's independent does in the
    # Senate. Printed rather than quietly coerced to one side.
    odd = sorted(c for c, e in new.items() if e["party"] not in ("D", "R"))
    if odd:
        print("  not D or R, so needing a decision for chamber arithmetic:")
        for code in odd:
            print(f"     {code}  {new[code]['incumbent']}  ({new[code]['party']})")

    redrawn = sum(1 for c in new if c[:2] in REDRAWN)
    dem_lean = sum(1 for e in new.values() if e["share"] > 0.5)
    total_dem = sum(e["dem"] for e in new.values())
    total_rep = sum(e["rep"] for e in new.values())
    print(f"\nwrote {OUT}")
    print(f"  435 districts, {redrawn} of them in the {len(REDRAWN)} redrawn states")
    print(f"  districts Harris carried: {dem_lean}")
    print(f"  implied national two-party Democratic share: "
          f"{100 * total_dem / (total_dem + total_rep):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
