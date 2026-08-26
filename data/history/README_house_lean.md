# District-level presidential lean, 2024

`house_district_lean_2024.csv` — the 2024 presidential two-party result in each
congressional district, which is the baseline a House forecast rests on. 433 of
435 districts, 13 KB.

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

**FL-20 and OK-03 are missing.** Both had House races with no recorded
opposition votes, so there is nothing to join their presidential votes to. Both
are safe seats, but the gap is real and should be filled from the state's own
lean rather than left as a hole.

**The implied national two-party Democratic share is 49.20%, against an actual
figure near 48.3%.** The 1% of votes that could not be placed are concentrated
in precincts whose House race was uncontested, which is more common in safe
seats. Compute a district's *lean* against the national figure derived from this
same file rather than an external one, and the bias largely cancels — which is
what `fundamentals.py` does, subtracting the national logit from each unit's.

**`house_2024_holder` gives 209 D / 223 R against an actual 215 / 220.** It is
derived from precinct House votes, so uncontested and partly-reported races skew
it. Prefer the FEC's incumbency flag: for the House every seat is up every two
years, so a 2026 filer marked `I` really is the sitting member seeking
re-election — unlike the Senate, where the same flag marks members who are not
on the ballot at all.
