# District-level presidential lean, 2024

`house_district_lean_2024.csv` — the 2024 presidential two-party result in each
congressional district, which is the baseline a House forecast rests on. All 435
districts, 13 KB.

- **Built by:** `scripts/build_house_lean.py`
- **Sources:** MIT Election Data and Science Lab precinct returns, both
  **CC0 1.0** (public domain dedication):
  - `doi:10.7910/DVN/XDJYKC` — U.S. President Precinct-Level Returns 2024
  - `doi:10.7910/DVN/USBYR4` — U.S. House Precinct-Level Returns 2024
- **Why derived rather than downloaded:** every ready-made district-level source
  is either unlicensed (ElectIndex, Daily Kos) or an approximation that
  apportions counties across district lines. Precinct data is what the
  professionals actually build from.

## How a presidential vote gets a district

A presidential row carries no district — MEDSL's `district` field is the
*office's* district, and a presidential race is statewide. But the same precinct
also reports its **US House** race, and that row does. Joining the two on
`(state, county_fips, precinct)` gives every presidential vote a district, with
no shapefiles, no spatial join and no county apportionment.

Precincts are weighted by the House votes cast in each district rather than
assigned to one, which handles a precinct genuinely straddling two districts by
splitting its presidential votes in the same proportion.

## What each state made us learn

Every one of these silently discarded data rather than erroring, which is why
the script reports coverage per state.

| State | Problem | Effect if unhandled |
| --- | --- | --- |
| Ohio | Files **every district for every precinct**, nearly all with zero votes | All 8,878 precincts looked split 15 ways; the whole state dropped, 5.7M votes |
| Arizona, Indiana | Vote counts written as floats, `316.0` | `int()` raised on every row; 18 districts lost |
| Nebraska | Every presidential row labelled `NONPARTISAN` — its ballot does not print party for president | Entire state excluded; `party_detailed` is correct and is the fallback |
| Maine | `UNION` in the House file, `Union` in the presidential one | 115 of 512 precincts unmatched; ME-01 lost |
| Rhode Island | Files some rows as `STATEWIDE` despite having two districts | Invented an `RI-AL` district that does not exist |

## Known limitations

**~~FL-20 and OK-03 are missing.~~ Fixed — recovered by subtraction.** Florida
and Oklahoma declare an unopposed House candidate elected without putting them
on the ballot, so those two races have no precinct rows at all and their
presidential votes had nothing to join to.

But those same precincts are exactly the ones that fail to join. Where a state
is missing exactly one district, every unplaced presidential vote in it belongs
to that district, so the answer is arithmetic rather than inference:

| District | Recovered votes | vs state average | Democratic share |
|---|---|---|---|
| FL-20 | 297,094 | 0.76x | **70.0%** |
| OK-03 | 523,352 | 0.82x | **26.1%** |

The earlier suggestion in this file — fill the gap from the state's own lean —
was **actively harmful** and was followed. Florida's average is 43% Democratic,
so FL-20, one of the most Democratic districts in the country, was recorded at
43%; a downstream script then read that number, saw it was below half, and wrote
the seat down as Republican-held.

The recovery is guarded rather than trusted. Unplaced votes also accumulate from
ordinary join failures, and attributing those to a real district would corrupt a
number rather than supply a missing one — so a recovered total must fall between
0.35x and 2.0x the state's average district before it is accepted, and anything
rejected is printed rather than silently dropped.

**The implied national two-party Democratic share is 49.16%, against an actual
figure near 48.3%.** The 1% of votes that still cannot be placed are
concentrated in precincts whose House race was uncontested, which is more common
in safe seats. Compute a district's *lean* against the national figure derived
from this same file rather than an external one, and the bias largely cancels —
which is what `fundamentals.py` does, subtracting the national logit from each
unit's.

**`house_2024_holder` gives 209 D / 223 R against an actual 215 / 220, and three
blanks. Do not use it as the incumbency source.** It is derived from precinct
House votes, so uncontested and partly-reported races skew it.

**And do not reach for the FEC's incumbency flag instead** — an earlier version
of this file recommended exactly that, reasoning that because every House seat
is contested every two years, a 2026 filer marked `I` must be the sitting
member. That reasoning is wrong. `CAND_ICI = I` means "has held this office at
some point": there are 555 flagged filings for 435 seats, 108 districts carry
more than one, and 28 carry incumbents of *both* parties. It gives 199 D / 232 R.

Use `unitedstates/congress-legislators` (CC0), which is what
`build_house_races.py` now does — `legislators-current` for sitting members and
`legislators-historical` for the five vacant seats. That gives **215 D / 220 R**,
the actual chamber.
