# Notes for Claude Code

Context for working in this repo. Read `docs/METHODOLOGY.md` before changing anything in
`src/midterms/model/`.

## Environment

- Windows, Python 3.12 in `.venv`. Use `.venv\Scripts\python.exe`.
- Always set `PYTENSOR_FLAGS=cxx=` — there is no C++ compiler and none is needed. Sampling
  goes through **nutpie** (Numba). The `g++ not available` warning is expected noise.
- `pytest -q` runs in ~5 seconds and never samples. Keep it that way; sampling belongs in
  `midterms run`, not in tests.

## Invariants — do not break these silently

1. **Seat arithmetic must close to 100.** `config/races_senate_2026.yaml` `control:` is
   validated on load and asserted in `tests/test_config.py`. A miscounted baseline shifts
   every control probability without changing any race, so nothing looks wrong.
2. **House effects must stay sum-to-zero.** Remove the constraint and the model is
   unidentified — a constant can move between the house effects and the latent state for an
   identical likelihood.
3. **Random walks must stay non-centred.** Cumulative sums of standard normals scaled by a
   separately sampled SD. Centring them reintroduces funnel geometry; this was measured —
   min ESS went from ~19 to ~670 when the house effects were non-centred.
4. **No second drift-to-election-day term.** The time grid already ends on election day, so
   the fitted walk covers it. Adding drift on top double-counts and inflates every interval.
5. **The roster fails closed.** An unrecognised candidate name resolves to `other` and the
   poll is skipped. Never guess a party — a wrong assignment corrupts a race's whole
   polling history, while skipping only loses information.
6. **Focus and keydown must be bound per element on the map, never delegated.**
   Those events do not reliably bubble from an SVG child to an HTML ancestor. A
   delegated listener on the map container never fires even though
   `document.activeElement` is correctly the focused `<path>`, which silently
   removes keyboard access to all 35 races. Pointer events *do* delegate fine.
7. **The error scales in `election_day_error` are FITTED. Do not hand-edit them.**
   They come from `midterms calibrate` against vendored historical poll errors, and
   `tests/test_calibration.py` asserts the config still matches the fit. If you change
   the window or the era, re-run calibrate and update both together.
8. **Do not inflate the scales to match `backtest-history`.** That backtest adds error to
   a point estimate with no uncertainty of its own, so it appears to want ~1.2x wider
   scales. The real model's posterior already supplies that width (measured ratio 1.05).
   Copying the factor across double-counts estimation error.
9. **Design and model must agree about the approval switch.** `build_model_data` drops
   approval polls when the feature is off and `build_model` skips their latents; if the
   two disagree the observation and latent vectors differ in length. There is an explicit
   check that raises a readable error, added after this bit as a raw PyTensor shape error.
10. **A model change must not be reported as a polling change.** Every run records a
   `model_fingerprint` (a hash of model.yaml plus the modules that define the arithmetic).
   When it differs between runs the commentary says so instead of blaming the polls. This
   was added after a recalibration was announced as a 5.6-point move "on 1 new poll".
11. **`site/data/*.json` and `outputs/runs/**` must stay committed.** The daily workflow
   diffs against the previous archived run to write commentary. Un-commit them and every
   run believes it is the first.

## Where things live

| Task | File |
|---|---|
| Change a prior or hyperparameter | `config/model.yaml` (never hard-code in Python) |
| Add/correct a race | `config/races_senate_2026.yaml` |
| Add a new nominee after a primary | `config/candidates_senate_2026.yaml` (`midterms audit-roster` finds them) |
| Change the model itself | `src/midterms/model/hierarchical.py` |
| Change seat/tipping-point logic | `src/midterms/model/simulate.py` |
| Change the dashboard | `site/` — plain HTML/CSS/JS, no build step |
| Change the map | `site/js/map.js` (render + interaction), `src/midterms/geo.py` (projection) |
| Change a chart | `site/js/charts.js` |
| Re-fit the error scales | `midterms calibrate`, then update `config/model.yaml` |
| Score against real elections | `midterms backtest-history` |
| Change what the JSON contains | `src/midterms/outputs.py` — **bump `SCHEMA_VERSION` and the matching constant in `site/js/app.js`** |

## Adding the House or Governors

The model is chamber-agnostic; races come from config. What is actually needed:

1. `config/races_house_2026.yaml` in the same schema (`unit` becomes the district, e.g.
   `AZ-06`) plus its own `control:` block.
2. VoteHub subject mapping — House subjects are `"2026 AZ-06"`, not `"2026 Arizona"`. This
   already works via `Race.subject_for()` if `name` is set to the district code, or via the
   `votehub_subject` override field.
3. District-level fundamentals instead of statewide presidential results.
4. Poll type `us-representative` (governors: `governor`) in the CLI's fetch list.

`src/midterms/model/` needs no changes for this.

### The actual blocker is data, not code (investigated 2026-08-24)

The mechanics above are the easy half. The House was scoped and deliberately
deferred, because 398 of 435 districts would rest entirely on fundamentals and
there is no cleanly-licensed source for the input they need.

**Poll coverage.** VoteHub carries 77 general-election House polls across 37 of
435 districts — AK-01 (8), ME-02 (6), AZ-06 (4), then a long tail of one or two.
So ~91% of the chamber is fundamentals-only, which makes the district baseline
the whole model rather than a prior the polls quickly overwhelm.

**What is needed.** Presidential two-party lean per district, ideally 2024 and
2020. Nothing else is missing.

**What exists, and why each was rejected:**

| Source | Has it? | License | Verdict |
|---|---|---|---|
| [ElectIndex](https://github.com/ElectIndex/26_us_forecast_data) `historical.csv` | Yes — pres 2016/2020/2024 by district, 486 rows | **none** (`license: null`) | Exactly right, cannot redistribute. **Re-check this first** — a license here turns a multi-hour job into an afternoon. |
| [tonmcg](https://github.com/tonmcg/US_County_Level_Election_Results_08-24) | County-level presidential | MIT | Usable, but counties split across districts; needs Census crosswalk + population weighting, and error concentrates in the big urban districts that decide the chamber. |
| Census county↔CD relationship files | Crosswalk | public domain | Pairs with the above. |
| Wikipedia state presidential pages | By-district tables | CC BY-SA | Accurate (reported, not apportioned) but 50 pages of brittle parsing. |
| [michaelminn.net](https://michaelminn.net/tutorials/data/) `2024-electoral-districts.csv` | 2024 **House** results by district | unstated | Not presidential. Embeds incumbency and candidate quality, and 38 of 441 districts were uncontested so have no two-party margin at all. Worst option. |

**Why not just ship the weak version.** A House forecast built on prior House
results would look exactly as authoritative as the Senate one while being mostly
prior, and would be most wrong in open seats — the competitive ones. The
dashboard's credibility rests on its numbers meaning what they appear to mean.

## Uncertainty is split, not stacked

`election_day_error` is fitted at **0-14 days out**, where the polling miss is almost
entirely systematic. The 45-120 day figure is larger because it also contains the
opinion change between poll and election — and the model already represents that, by
random-walking to election day. Using the far figure would count drift twice.

`scripts/measure_drift.py` is what establishes that the walk really does carry it
(model 2.5 pts against an empirical 3.9 over the same horizon). If either half is
re-fitted, re-run it: the two numbers are only correct together.

## Pollster quality multipliers are centred, and must stay that way

`polls.pollster_ratings` scales each poll's non-sampling noise by its pollster's
538 record. The multipliers are divided by their poll-weighted mean, so that mean
is exactly 1.0.

This is not cosmetic. `excess_sd_prior` was fitted against all historical polls,
so it already describes an average pollster, while 538's predictive plus-minus
averages **+0.49 points, not zero**. Applying the ratings raw would inflate every
poll's noise and undo the calibration that sets the width of the seat
distribution -- with nothing in the output looking wrong. Measured after the
change: median per-race 90% interval moved 23.80 -> 23.87 pts, a ratio of 1.0025,
with individual races moving both ways (0.973-1.033). That is the property to
re-check if the weighting is ever touched.

Only `sigma_excess` is scaled, never `sampling_var`. The plus-minus measures error
*beyond* what sample size explains, so scaling the sampling term too would
penalise a bad pollster twice.

## Measured, then rejected

Keep these decisions unless new evidence overturns them; each cost real time to
establish and the numbers are in `docs/METHODOLOGY.md`.

- **Student-t election-day error** — tested, made coverage worse at every level. The
  residual mis-calibration is width, not tail shape.
- **Presidential approval in the national environment** — implemented and works
  (correlation -0.54, correctly signed), but buys 6.6% tighter eta for a 35% ESS loss.
  Disabled by default; `scripts/ab_approval.py` reproduces it.
- **Changing the correlation kernel's parameters** — `scripts/fit_correlation.py`
  measures which states actually missed together, over 13 cycles and 135 pairs. The
  covariates are **validated**: agreement between the kernel's ranking and the
  measured one is +0.17, positive at every setting tried. But agreement rises as
  `region_weight` falls, monotonically, which looks like a finding and is not:
  bootstrapping over cycles, 0.0 beats 0.6 in 71% of resamples with a 90% interval
  of -0.030 to +0.063. The committed values stand.
- **Comparing raw kernel correlations to demeaned measurements** — a trap of the same
  shape as the backtest appearing to want wider error scales. Removing each cycle's
  mean error (necessary, or the national miss is double-counted) forces average
  pairwise correlation to about -1/(n-1) whatever the truth is. Against that, the
  kernel looks wildly over-correlated (+0.329 vs -0.064) and the fix looks like
  shrinking it — which would let errors cancel across 35 races and leave the seat
  distribution far too narrow. Demean both sides; then it reads -0.068 vs -0.064.
- **Fundraising as a covariate** — measured against 171 historical Senate races
  (`scripts/fit_fundraising.py`). Predicting polling error, the money effect is
  entirely incumbency: t=+3.18 alone, t=+0.27 once incumbency is included, and the
  fundamentals prior already has incumbency. Robust, because the lookahead below
  would have helped it. Predicting the result beyond state and incumbency it looks
  strong (t=+7.72) but FEC `weball` is end-of-cycle money, and a live forecast has
  filings through June 30 — so that fit partly uses the outcome to predict the
  outcome. Not rejected, **unmeasured**: doing it properly needs period-level data
  from the FEC API, which needs a key.
- **Seven-step map colour ramp** — adjacent pairs fell below the perceptual separation
  floor near the neutral midpoint. Five steps is what the colour space supports.
  `scripts/check_palette.py` re-measures this and runs in the suite.
- **Low `min_ess_bulk` is not a convergence problem here** — it is always `eta_start`,
  the national environment 16 months before the election, which nothing constrains and
  the forecast does not depend on. Election-day `theta`, which it does depend on, runs
  ESS 4400+. Check the parameter before chasing the number.
- **Drawing a reference line as a tick** — the candidate chart's 50% line was whichever
  tick equalled 50, so it vanished on the 20-80% races where the ladder steps by 20.
  Reference values are drawn from the domain; ticks are for ticks.
- **Naming a candidate from the roster when the primary was contested** — the roster
  carries whole primary fields for ten 2026 races and is not ordered by winner, so the
  first entry is a guess. Unpolled races with a contested field are left unnamed.

## Known rough edges

Listed in the README under "Known data-quality items" and `docs/METHODOLOGY.md` §9. The
highest-value improvement is demographic covariates (education, urbanicity) in the
election-day correlation kernel — currently it uses political covariates only.
