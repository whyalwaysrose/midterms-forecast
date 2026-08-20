# 2026 U.S. Midterms — Bayesian Forecast

A hierarchical Bayesian dynamic model of the 2026 U.S. Senate elections, refit daily
against new polling, publishing to a static dashboard with automatically generated
day-over-day commentary.

- **Model:** PyMC. Latent random-walk state per race, a shared national environment,
  pollster house effects, screen and partisan-sponsor adjustments, and a correlated
  election-day error that produces a full chamber seat distribution.
- **Data:** [VoteHub Polling API](https://votehub.com/polls/api/) — free, no API key,
  **CC BY 4.0**.
- **Output:** static JSON + a dependency-free dashboard, deployed to GitHub Pages by a
  daily GitHub Actions cron.

---

## ⚠️ SETUP — things only you can do

Everything else runs itself. These five steps are manual and need your GitHub account.

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
finishes, your dashboard is live at:

```
https://<your-username>.github.io/midterms-forecast/
```

That is the link to share.

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
  outputs.py                  forecast.json / history.json
  commentary.py               day-over-day diff → changelog
  backtest.py                 calibration against held-out polls
  cli.py                      entry points

site/                         the dashboard (plain HTML/CSS/JS, no build step)
outputs/runs/<date>/          slim archived runs — the memory commentary diffs against
.github/workflows/            daily refresh + CI
```

**Nothing above `model/` knows the chamber is the Senate.** Races come from config, and
the model is written over a generic `race × time` grid. Adding the House means adding
`config/races_house_2026.yaml` with the same schema plus its own control arithmetic — not
rewriting the model. The two things that would need extending are the VoteHub subject
mapping (House subjects are `"2026 AZ-06"` rather than `"2026 Arizona"`) and district-level
fundamentals in place of statewide presidential results.

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

## Data source and licensing

Polling comes from the **VoteHub Polling API** (`https://api.votehub.com`), which
aggregates from hundreds of pollsters and is offered free to researchers, journalists and
developers. It is licensed **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
which permits redistribution and derivative works with attribution. Attribution appears in
the dashboard footer and in `data_source` inside every `forecast.json`.

This choice, the alternatives considered, and what changed after FiveThirtyEight's
shutdown: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

Raw API snapshots are cached under `data/raw/` and are **not** committed — they are
re-fetchable, and only derived artefacts belong in the repository.

---

## Testing

```bash
pytest -q          # 80 tests, ~5 seconds, no sampling required
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
