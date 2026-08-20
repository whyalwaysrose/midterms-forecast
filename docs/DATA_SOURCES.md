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
