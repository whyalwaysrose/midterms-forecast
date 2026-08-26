# 2026 U.S. Midterms — Bayesian Forecast

A hierarchical Bayesian dynamic model of the 2026 U.S. Senate elections, refit daily
against new polling, publishing to a static dashboard with automatically generated
day-over-day commentary.

- **Model:** PyMC. Latent random-walk state per race, a shared national environment,
  pollster house effects, screen and partisan-sponsor adjustments, and a correlated
  election-day error that produces a full chamber seat distribution.
- **Data:** [VoteHub Polling API](https://votehub.com/polls/api/) — free, no API key,
  **CC BY 4.0**.
- **Output:** static JSON + a dependency-free dashboard — including an interactive
  Albers USA map of all 35 races — deployed to GitHub Pages by a daily GitHub Actions
  cron.

**Live:** <https://whyalwaysrose.github.io/midterms-forecast/> — refit and redeployed
daily at 11:00 UTC by GitHub Actions.

---

## Setup — already done for this deployment

The repository, Pages and the daily schedule are live. This section is kept as the
record of what was configured, and as the recipe for standing up a fresh copy.

Everything else runs itself; these steps need a GitHub account.

### 1. Create the GitHub repository

```bash
cd C:\Users\rsnjo\Projects\midterms-forecast
git add -A
git commit -m "Initial commit: Bayesian 2026 Senate forecast"
gh repo create midterms-forecast --public --source=. --push
```

No `gh` CLI? Create an empty **public** repo on github.com, then:

```bash
git remote add origin https://github.com/<your-username>/midterms-forecast.git
git branch -M main
git push -u origin main
```

> The repo must be **public** for GitHub Pages on a free account. If you need it
> private, Pages requires a paid plan.

### 2. Enable GitHub Pages

Repo → **Settings** → **Pages** → under *Build and deployment*, set
**Source = GitHub Actions**. Do not pick "Deploy from a branch".

### 3. Allow Actions to push commits

Repo → **Settings** → **Actions** → **General** → *Workflow permissions* →
select **Read and write permissions** → Save.

This is required. The daily job commits the regenerated forecast, and the next day's
commentary is produced by diffing against that commit. Without write access, every run
would think it was the first and no commentary would ever be generated.

### 4. Run it once by hand

Repo → **Actions** → **Daily forecast** → **Run workflow**.

The first run takes roughly 5–10 minutes (dependency install plus sampling). When it
finishes, the dashboard is live at:

```
https://<your-username>.github.io/midterms-forecast/
```

That is the link to share. Confirm the run's **Commit regenerated data** step actually
committed — if workflow permissions are wrong it succeeds while doing nothing, and the
failure only shows up as a missing changelog the following day.

### 5. Nothing to configure for data access

**No API key is needed.** The VoteHub API is free and unauthenticated, so there are no
repository secrets to add. If you later swap in a paid data source, add its key under
**Settings → Secrets and variables → Actions** and read it via `env:` in
`.github/workflows/daily-forecast.yml`.

### Ongoing maintenance — one recurring task

After a primary, new nominees appear in the feed as names the model does not recognise.
Their polls are **skipped** until you classify them, and the race quietly falls back to
its fundamentals prior. The daily workflow runs an audit step that surfaces this in the
log; you can also run it locally:

```bash
midterms audit-roster
```

Add any reported names to `config/candidates_senate_2026.yaml` under the correct party
(or under `other` for minor-party candidates) and commit.

---

## SETUP — two optional API keys (only you can request these)

**The forecast runs fine without both.** Nothing is broken; these unlock two
improvements that are currently blocked. Each is free, instant, and needs a
signup in your name, which is why they are not already done.

### 1. Census API key — demographic covariates

The election-day correlation kernel decides how much state polling errors move
together, which matters a lot: errors that are independent average out across 35
races, errors that move together do not. It currently uses political covariates
(presidential lean, region). `scripts/fit_correlation.py` confirms those really do
track which states miss together, so this is an improvement rather than a repair —
education and urbanicity are the covariates the 2016 and 2020 misses were about.

Request at **<https://api.census.gov/data/key_signup.html>** — it asks for an
organisation name and an email address, and emails the key.

### 2. FEC API key — fundraising without lookahead

Fundraising is measured and currently *not* used, for a specific reason. Against
171 historical races the effect is entirely incumbency (t=+0.27 once incumbency is
included), which the fundamentals prior already has. Against the fundamentals
baseline it looks strong — but the FEC bulk files are end-of-cycle, so that fit
uses money raised *after* the date a live forecast would have been made. Testing
it honestly needs cumulative totals as of June 30 of each election year, which
means per-report data for around 350 committees. The demo key allows **10
requests an hour**, which is why this needs a real one.

Request at **<https://api.data.gov/signup/>** — the same key works across all
api.data.gov services, FEC included.

### 3. Give the keys to the project

Put them in a `.env` file in the repo root. It is gitignored, so they stay off
GitHub:

```
CENSUS_API_KEY=your-census-key
FEC_API_KEY=your-fec-key
```

Then confirm they work — this calls each API once and prints only a redacted
fragment of the key, so the output is safe to paste anywhere:

```bash
midterms check-keys
```

For the GitHub Actions run, add the same two as repository secrets under
**Settings → Secrets and variables → Actions**. The daily forecast does not need
them today; nothing will break if they are absent.

## Quick start (local)

Requires Python 3.11+ (3.12 recommended). Verified on Windows 11 with Python 3.12.10.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install -e . --no-deps

midterms run                     # fetch polls, fit, simulate, write outputs
python -m http.server 8765 --directory site
# open http://localhost:8765
```

The dashboard **must** be served over HTTP — opening `site/index.html` as a `file://`
URL will fail because browsers block `fetch` from the filesystem.

### Commands

| Command | What it does |
|---|---|
| `midterms run` | Full pipeline: fetch → fit → simulate → write JSON + commentary |
| `midterms run --offline` | Same, reusing the last raw snapshot (no network) |
| `midterms run --date 2026-08-01` | Reproduce a run as of a past date (ignores later polls) |
| `midterms fetch` | Fetch and snapshot polls only |
| `midterms audit-roster` | Report candidate names the roster cannot classify |
| `midterms backtest --holdout-days 30` | Calibration check against held-out polls |
| `midterms bundle` | Build a single self-contained `outputs/dashboard.html` |
| `midterms build-map` | Re-project the vendored state boundaries (`run` does this too) |
| `midterms calibrate` | Fit the error scales from historical polling |
| `midterms backtest-history` | Score win probabilities against elections that happened |

`midterms bundle` inlines the CSS, JS and data into one portable HTML file. Useful for
sending someone a snapshot, or for viewing the dashboard without running a server —
browsers block `fetch` from `file://`, so the multi-file version needs one.

A run takes about 45 seconds of sampling on a modern laptop.

### Windows note

PyTensor warns that `g++` is not available. This is expected and harmless: sampling uses
**nutpie** (a Numba-compiled NUTS implementation), which needs no C++ toolchain and is
faster than the C backend anyway. The CLI silences the warning by setting
`PYTENSOR_FLAGS=cxx=`.

---

## Architecture

```
config/
  races_senate_2026.yaml      all 35 contests + chamber-control arithmetic
  candidates_senate_2026.yaml candidate → party roster (see below)
  races_house_2026.yaml       all 435 districts — GENERATED, do not hand-edit
  candidates_house_2026.yaml  roster for the polled districts — GENERATED
  model.yaml                  every prior scale and hyperparameter

src/midterms/
  config.py                   typed, validated config loading
  data/
    votehub.py                API client + dated raw snapshots
    roster.py                 candidate name → party resolution
    polls.py                  raw records → tidy two-party poll table
  fundamentals.py             structural prior per race
  model/
    design.py                 poll table → model index arrays + time grid
    hierarchical.py           the PyMC model
    correlation.py            state-similarity kernel for election-day error
    simulate.py               posterior → seat distribution, tipping points
  geo.py                      Albers USA projection → SVG paths for the map
  cartogram.py                435 equal squares, relaxed apart from centroids
  calibration.py              fit error scales from historical polling
  backtest_history.py         score against elections that actually happened
  outputs.py                  forecast.json / history.json, per chamber
  commentary.py               day-over-day diff → changelog
  backtest.py                 calibration against held-out polls
  cli.py                      entry points

site/                         the dashboard (plain HTML/CSS/JS, no build step)
  js/charts.js                seat / history / trajectory charts + hover layer
  js/map.js                   the interactive state map (Senate)
  js/districts.js             the district cartogram (House)
data/geo/                     vendored state boundaries (us-atlas, ISC)
data/history/                 vendored historical poll errors (538, CC BY 4.0)
outputs/runs/<date>/          slim archived runs — the memory commentary diffs against
.github/workflows/            daily refresh + CI
```

**Nothing above `model/` knows which chamber it is fitting.** Races come from config and
the model is written over a generic `race × time` grid, so the House was added by
supplying `config/races_house_2026.yaml` in the same schema with its own control
arithmetic — not by rewriting the model. That claim has now been tested rather than
merely asserted: the 435-race fit converges cleanly (§3.1 of the methodology).

Both chambers are refit by the same daily job and share one page, switched by a tab.
Run either directly:

```bash
midterms run --chamber senate    # ~60s
midterms run --chamber house     # ~700s
```

The Senate keeps the unsuffixed output filenames (`forecast.json`, `history.json`) it has
always had; the House is suffixed (`forecast_house.json`). The asymmetry is deliberate —
the published site, every archived run, and any saved link already point at the
unsuffixed names.

The **scripts that generate the House configs** are run by hand, not daily, because their
inputs change once a cycle rather than once a day:

```
scripts/build_house_lean.py     precinct returns → district presidential lean
scripts/build_house_races.py    lean + current membership → the 435-race config
scripts/build_house_roster.py   FEC filings → candidate roster for polled districts
```

---

## The model in brief

Everything is on the logit of the two-party Democratic share.

```
θ_{r,t} = α_r + λ_r · η_t + ε_{r,t}
```

| Term | Meaning |
|---|---|
| `η_t` | national environment, a Gaussian random walk fed by generic-ballot polls |
| `α_r` | race baseline under a *tied* national environment, from the fundamentals prior |
| `λ_r` | elasticity — how hard this race swings with the nation, pooled around 1 |
| `ε_{r,t}` | race-specific random walk (candidate news, spending, scandals) |

Polls observe that state through a biased instrument:

```
logit(yᵢ) ~ StudentT(ν, θ_{rᵢ,tᵢ} + houseᵢ + screenᵢ + ρ·partisanᵢ, √(vᵢ + σ²_excess))
```

Three deliberate choices keep it identified and well-behaved:

1. **House effects sum to zero.** Otherwise you could add a constant to every house
   effect and subtract it from the latent state for an identical likelihood, and the
   sampler would wander that ridge forever.
2. **Every random walk is non-centred** (cumulative sums of standard normals, scaled by a
   separately sampled SD). Centred walks produce funnel geometry and sample badly.
3. **State leans are measured relative to the nation**, so the national swing enters
   exactly once — through `η` — instead of being double-counted inside `α`.

Then, at simulation time, a **correlated election-day error** is added: a national
component common to every race, plus a state component whose correlation decays with
political distance between states. This is the piece that makes the seat distribution
honest — 35 independent errors would cancel out and produce absurd confidence.

Full detail, including why each prior scale is what it is: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Known data-quality items

Recorded openly rather than buried, because they affect how much to trust specific numbers.

- **Oklahoma and South Carolina** are marked as open seats. Sources conflict on whether
  the elected incumbents are on the 2026 ballot. Both are safe-R on fundamentals, so the
  chamber-level impact is negligible, but the per-race incumbency term may be wrong.
- **Nebraska.** Dan Osborn runs as an **independent** and has said he would not
  automatically caucus with either party. Chamber arithmetic forces a choice, so he is
  counted on the Democratic side. Read Nebraska — and its contribution to the seat
  distribution — with that caveat. It is flagged in the dashboard.
- **Montana** has no confirmed Democratic nominee in the roster, so its polls are skipped
  and it is carried by fundamentals alone.
- **Unpolled races.** Fourteen of the 35 races have no qualifying general-election polls.
  Their intervals are wide by construction, but they rest on the fundamentals prior, which
  knows nothing about candidate quality.
- **Generic-ballot recency.** At the time of writing, generic-ballot polls in the feed run
  several weeks behind state polls. The national environment's uncertainty widens
  accordingly, which is the correct behaviour, but it is worth knowing.
- **`sigma_excess` estimates near zero.** This is not a bug. The design effect (1.5) and
  the Student-t(4) likelihood already give each poll roughly 7 points of margin error, so
  there is no unexplained variance left to attribute. See the methodology document.

---

## Calibration against history

The two numbers that set how confident the headline is allowed to be —
`election_day_error.national_sd` and `.state_sd` — used to be asserted from the
literature. They are now **fitted** from FiveThirtyEight's archived pollster-ratings
dataset (20,466 polls paired with actual results, CC BY 4.0, vendored under
`data/history/`).

`midterms calibrate` decomposes historical Senate poll error into three levels that
behave completely differently in a seat forecast:

| Component | Was assumed | History says |
|---|---|---|
| National (cycle-wide, perfectly correlated) | 3.8 pts | **3.0 pts** |
| State-specific | 4.2 pts | **5.7 pts** |
| Total race-level | 5.7 pts | **6.4 pts** |
| Per-poll design effect | 1.50 | **1.18** |

`midterms backtest-history` then scores the result against 167 real races. The fitted
scales beat the asserted ones at every coverage level (90% interval: 84.4% vs 77.8%), and
reliability sits close to the diagonal — races called at 42% won 39% of the time, at 82%
won 86%. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §8b, including why the
backtest's apparent demand for 1.2× wider scales must **not** be copied into the config.

## The map

An **Albers USA** projection — the familiar composite with Alaska and Hawaii inset —
computed at build time from vendored TopoJSON and emitted as plain SVG paths. No mapping
library ships to the browser, which keeps the page a stylesheet and two scripts.

Three deliberate choices:

- **Equal-area projection.** No state's apparent size is inflated relative to another's.
  That is the least-bad property available when the quantity shown (one Senate seat) has
  nothing to do with land area.
- **Only the 35 races are coloured.** The other 15 states are greyed out, so empty land
  cannot shout a result it does not carry. A map fundamentally cannot show that Wyoming
  and Rhode Island are worth exactly one seat each — the seat histogram below it is the
  equal-weight truth, and the page says so.
- **Five rating steps, not seven.** A seven-step ramp was built and measured first; near
  the neutral midpoint its adjacent pairs fell to a perceptual separation of 4–10, well
  under the floor of 15, meaning readers with ordinary colour vision could not tell Lean
  from Likely. Toss-up additionally carries a diagonal hatch, so the one step sitting
  between the two hues never depends on colour alone. Exact probabilities are always in
  the hover detail.

Below 820px the inline state labels would render under 8px, so they drop out; below 640px
the map becomes a pure overview and the race table takes over as the interface. Rhode
Island is 3x4 pixels on a phone — no amount of tuning makes that tappable.

### The House cartogram

The same projection **cannot** be used for the House, and the reason is worth stating
plainly. Congressional districts are drawn to hold equal numbers of people, so they
differ in area by three orders of magnitude — Wyoming's single at-large seat covers more
ground than the twenty that cover New York City. Shading real district shapes would hand
almost all of the ink to the emptiest seats and shrink the ones that decide the chamber
to specks. The map would be beautiful and would systematically mislead.

So the House gets a **cartogram**: 435 squares of identical size, grouped by state, with
the states placed roughly where they belong. Every seat counts once, and the picture and
the arithmetic agree.

The layout is **derived, not hand-drawn**. Published tile-grid layouts are hand-tuned and
carry their own licensing questions; this one is computed by `src/midterms/cartogram.py`
from the state centroids already in `site/data/us-states.json` — the same Census
boundaries the Senate map is built from. Each state's block starts at its true centroid
and blocks are then pushed apart until none overlap. It needs no new data and cannot fall
out of step with the map beside it.

Two properties are enforced by tests, because neither failure raises an exception:

- **No district is missing a square.** A district with no tile vanishes from the picture
  while still counting toward 218 — the one thing this drawing could hide from a reader.
- **The layout still looks like America.** Relaxation is iterative, and a version that
  stopped converging would still produce 435 non-overlapping squares. The measured
  ordering is 1225/1225 state pairs correct east–west and 1221/1225 north–south; the test
  fails below 99%.

Districts with **no polling of their own** — roughly nine in ten — are drawn at reduced
opacity. Their forecast is as real as any other, but it rests on the district's
presidential lean and the national swing rather than on anyone having asked voters there,
and a reader should be able to see at a glance how much of the chamber that covers.

## Charts

Every chart carries a hover layer: a crosshair and tooltip on the time series, a
per-bar tooltip on the seat histogram. The seat chart reports the cumulative
probability as well as the exact one, because "what are the odds of *at least* 51
seats" is the question a reader actually brings to a seat histogram — and it is a
useful internal check, since it should reconcile with the headline control
probability.

Axes scale to the data rather than to a fixed range, on round tick values. The
forecast-over-time chart is why: pinned to 0–100% its line was a flat squiggle
using a tenth of the panel, because every run so far has sat inside a ten-point
band. It now scales to the data but always keeps 50% in the domain, since crossing
that line is the one change that alters the story — so the tighter axis cannot
flatter the movement.

## Visit counting

The hosted page carries a [GoatCounter](https://www.goatcounter.com) tag —
cookieless, ~3KB, loaded asynchronously. Stats live at
`https://whyalwaysrose.goatcounter.com`.

It records aggregate visits only: totals over time, country, referrer, browser.
Nothing that identifies a person, and no cookies, so no consent banner is needed;
the footer says so plainly anyway.

**The standalone bundle deliberately does not carry it.** `midterms bundle`
strips any script loading from another host before writing
`outputs/dashboard.html`. A file someone was handed, or opened from disk, should
not quietly report that back — and the artifact viewer's CSP would block the
request regardless, so leaving it in would only produce a console error. Four
tests in `tests/test_frontend_contract.py` pin both halves: the tag is present on
the hosted page, and absent from the bundle.

Treat the numbers as a floor. Ad blockers and Safari's tracking protection drop
some share of requests — true of every analytics tool, paid ones included.

## Data source and licensing

Polling comes from the **VoteHub Polling API** (`https://api.votehub.com`), which
aggregates from hundreds of pollsters and is offered free to researchers, journalists and
developers. It is licensed **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
which permits redistribution and derivative works with attribution. Attribution appears in
the dashboard footer and in `data_source` inside every `forecast.json`.

This choice, the alternatives considered, and what changed after FiveThirtyEight's
shutdown: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

State boundaries come from **[us-atlas](https://github.com/topojson/us-atlas)** (ISC),
derived from U.S. Census Bureau cartographic boundary files (public domain). The TopoJSON
is vendored under `data/geo/` so the daily run never depends on a CDN.

Raw API snapshots are cached under `data/raw/` and are **not** committed — they are
re-fetchable, and only derived artefacts belong in the repository.

---

## Testing

```bash
pytest -q          # 129 tests, ~5 seconds, no sampling required
ruff check src tests
```

The most valuable tests are the seat-arithmetic ones in `tests/test_config.py`. A
miscounted chamber baseline is the most damaging silent error a seat model can have: it
shifts every control probability without changing any individual race, so nothing looks
wrong on the dashboard.

---

## Honest limitations

- The model predicts **polls** well. Whether it predicts the **election** well depends on
  whether the polls are systematically biased, which cannot be determined in advance. The
  `election_day_error` block in `config/model.yaml` is the explicit representation of that
  risk, calibrated to the historical record rather than to this cycle's data.
- The fundamentals prior uses presidential lean and incumbency. It has no notion of
  candidate quality, fundraising, or scandal. In races where those dominate — Nebraska is
  the obvious example — the polls have to do all the work.
- Backtesting against real outcomes is impossible until November 2026. What
  `midterms backtest` provides is a necessary condition (are the predictive intervals the
  right width?), not a sufficient one.
- **Presidential approval is ingested but off by default.** It works — the fitted
  correlation with the generic ballot is −0.54, correctly negative — but it tightens the
  national environment by only 6.6% while cutting minimum effective sample size from 455
  to 296. Measured, then disabled; `scripts/ab_approval.py` reproduces the test.
- **Measured calibration (33 held-out polls):** median error 2.27 points of margin, and the
  poll-level intervals are somewhat **too wide** — 50% intervals contained 69.7% of held-out
  polls. Deliberately not retuned: at n=33 the standard error on that figure is ~8.7 points,
  and tightening the likelihood to match 33 polls would be fitting the diagnostic rather
  than the phenomenon. Re-run `midterms backtest` as held-out polls accumulate. Note this
  says little about the *chamber* forecast, whose width is dominated by the correlated
  election-day error that no poll-based diagnostic can test.

## License

MIT for the code. Polling data is CC BY 4.0 from VoteHub and remains subject to that
license.
