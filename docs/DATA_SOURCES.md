# Data sources

## The problem

FiveThirtyEight was shut down by ABC News in March 2025, and its public polling database
went with it. By 2026 the `fivethirtyeight.com` domain had been redirected and the
archives were largely gone. For roughly a decade that repository had been the default
free, machine-readable, permissively-licensed source of U.S. polling data, and a great
deal of open-source election modelling was built on it.

What replaced it is fragmented:

| Source | Machine-readable? | License | Verdict |
|---|---|---|---|
| **VoteHub API** | Yes — clean REST/JSON | **CC BY 4.0** | **Chosen.** |
| New York Times poll tracker | No public API | Proprietary | Inherited 538's poll-tracking role, but there is no licensed feed. Some third-party projects scrape it; that is not something to build a public, redistributed model on. |
| Silver Bulletin | Partial (per-page CSV downloads) | Proprietary, subscription | Excellent data and methodology, but not licensed for redistribution and largely paywalled. |
| RealClearPolitics / 270toWin / Race to the WH | HTML only | Proprietary | Scrapeable in practice, but terms do not clearly permit redistribution. |
| Roper Center iPoll | Yes | Paid academic license | High quality, but paid and not redistributable. |
| Wikipedia polling tables | Semi-structured wikitables | CC BY-SA | A genuine fallback: permissive and comprehensive. But parsing is brittle and the tables are inconsistently formatted across races. Held in reserve. |

## What was chosen: VoteHub

**`https://api.votehub.com`** — documented at <https://votehub.com/polls/api/>.

- **No API key.** No authentication, no rate-limit headers observed, no repository secret
  to manage.
- **Licensed CC BY 4.0.** The API documentation states this explicitly. That is a genuine
  open license permitting redistribution and derivative works with attribution — which is
  exactly what a public forecast that republishes derived numbers requires.
- **Live and current.** Verified at build time: 346 Senate polls and 533 generic-ballot
  polls, with state polling current to within three days.
- **538-shaped schema.** Fields map almost one-to-one onto the old FiveThirtyEight poll
  format — pollster, sample size, population screen, field dates, sponsors, partisan flag.

### Endpoints used

```
GET /polls?poll_type=us-senator          all Senate polls
GET /polls?poll_type=generic-ballot      national generic congressional ballot
GET /subjects                            available subjects per poll type
GET /pollsters                           pollster names
```

Subjects are named `"<cycle> <State>"` for general elections (`"2026 Georgia"`) and get a
trailing party name for primaries (`"2026 Texas Democratic"`). Primaries are filtered out
at ingestion.

### Attribution

Required by the license and provided in three places: the dashboard footer, the
`data_source` block of every `forecast.json`, and the `ATTRIBUTION` constant in
`src/midterms/data/votehub.py`.

> Poll data from VoteHub (https://votehub.com), licensed CC BY 4.0.

## The one real gap: no party labels

The single significant shortcoming is that poll answers carry **candidate names only**:

```json
{"choice": "Jon Ossoff", "pct": 47.7}
```

There is no party field on an answer, and no `/candidates` endpoint. Reducing a poll to a
two-party Democratic share therefore requires an external name → party mapping, which
lives in `config/candidates_senate_2026.yaml`.

That file is maintained **fail-closed**: an unrecognised name resolves to `other`, is
excluded from the two-party denominator, and the poll is used only if exactly one D-side
and one R-side candidate can be identified. A wrong party assignment would silently
corrupt a race's entire polling history; discarding an unknown name merely loses
information. `midterms audit-roster` reports anything unclassified.

Real data-quality issues the roster absorbs, all observed in the live feed:

- Misspellings — `"Dan Sullvian"`, `"Thom Thillis"` — handled by an alias table.
- Generic placeholders — `"Dem"`, `"Rep"` — used when a pollster tests an unnamed nominee.
- Third-party and minor candidates, excluded from the two-party share.
- Independents. Dan Osborn in Nebraska needs an explicit `independent_counts_as` decision;
  see the README's known-issues section.

## The House needed a second source, and it had to be free

District presidential lean is the input every serious House model needs, and it is the
one thing VoteHub does not carry. A congressional district is not an administrative unit
anyone reports election results for, so the number has to be constructed.

**The two usual routes were both rejected.** Commercial datasets (Daily Kos Elections,
and similar) publish exactly this table but are licensed, which breaks the project's rule
that everything is free and redistributable. A GIS overlay of precinct shapefiles onto
district boundaries is the standard open method, but it means shipping geometry
dependencies and a large boundary dataset for one number per district.

**What was used instead:** the MIT Election Data and Science Lab's 2024 precinct returns,
released **CC0** (public domain, no attribution required, though it is given below
anyway). The join needs no geometry at all: within a single precinct, the presidential
rows and the US House rows describe the same voters, and the House row already names the
district. Votes in precincts split across districts are allocated proportionally.

| | |
|---|---|
| **Source** | MIT Election Data and Science Lab, 2024 precinct-level returns |
| **Host** | Harvard Dataverse |
| **License** | **CC0 1.0** — public domain dedication |
| **Coverage achieved** | **All 435 districts.** 98.97% of two-party presidential votes join directly; the two uncontested-race districts are recovered by subtraction |
| **Built by** | `scripts/build_house_lean.py` → `data/history/house_district_lean_2024.csv` |

Two supporting sources, both also unrestricted:

| Source | Used for | License |
|---|---|---|
| [`unitedstates/congress-legislators`](https://github.com/unitedstates/congress-legislators) | Who currently holds each seat | **CC0** |
| FEC bulk candidate file (`weball26.zip`) | Whether that member is running again | US Government work, public domain |

**A correction worth recording, because it was asserted the wrong way round first.** An
earlier version of `build_house_races.py` used the FEC's own incumbency flag (`CAND_ICI`)
and its header explicitly argued that the flag, though unreliable for the Senate, was
trustworthy for the House because every seat is contested every two years. That reasoning
was wrong. `CAND_ICI = I` means "has held this office at some point", not "holds it now":
there are **555 flagged filings for 435 seats**, 108 districts carry more than one, and 28
carry incumbents of *both* parties — Kyrsten Sinema still flagged in AZ-09, Rick Renzi in
AZ-01 having left in 2009. It produced 199 D / 232 R against an actual 215 / 220.

Current membership now comes from `congress-legislators` — `legislators-current` for
sitting members and `legislators-historical` for the five vacant seats, which are absent
from the current file entirely. That gives **215 D / 220 R**, the actual chamber.

Known limitations are documented in `data/history/README_house_lean.md`, including the
five state-specific data quirks the builder handles, the guard on the subtraction, and the
fact that the file's implied national Democratic share (49.16%) differs from the actual
(48.3%) — so lean must be computed against the file's own national figure, not an
external one, which is what `fundamentals.py` does.

## Reproducibility

Every fetch writes a gzipped, dated snapshot to `data/raw/votehub-<date>.json.gz` **before
any parsing**. Re-running against an old snapshot reproduces that day's forecast exactly.
This matters for the commentary generator: it lets a day-over-day change be attributed to
*the polls* rather than to an unnoticed change in the feed.

Snapshots are gitignored — they are re-fetchable, and only derived artefacts belong in the
repository.

## If VoteHub goes away

The ingestion layer is deliberately thin and isolated in `src/midterms/data/votehub.py`.
Swapping sources means producing records with these fields:

```
id, pollster, subject, start_date, end_date, sample_size,
population ("lv"|"rv"|"a"), partisan (null|"DEM"|"REP"), sponsors[], url,
answers[{choice, pct}]
```

Everything downstream — normalisation, the model, the dashboard — is unchanged. The most
likely replacement is a Wikipedia wikitable parser, which is more work but carries a
permissive CC BY-SA license.
