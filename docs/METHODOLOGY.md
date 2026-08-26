# Methodology

Full specification of the model, why each piece is there, and where it can go wrong.

---

## 1. Scale

Everything is on the **logit of the two-party Democratic share**. Working in logits keeps
the latent state unbounded (so a random walk cannot wander outside `[0, 1]`) and makes the
measurement variance approximately additive.

Conversions near a competitive race (`p = 0.5`):

```
d(logit)/d(share) = 1 / (p(1-p)) = 4
1 point of SHARE   ≈ 0.04 logit
1 point of MARGIN  ≈ 0.02 logit        (margin = 2 × deviation from 50% share)
```

So a prior SD of `0.10` logit is "about 5 points of margin". Every scale in
`config/model.yaml` is annotated in these terms.

Third-party support is excluded from the denominator: a poll showing 48/32/20 is treated
as a 60% two-party Democratic share, with the sample size discounted accordingly
(§4.2).

---

## 2. The latent state

### 2.1 National environment

`η_t` is the national two-party Democratic logit share on the generic congressional
ballot, modelled as a Gaussian random walk on a 7-day grid:

```
η_0 ~ Normal(0, 0.10)
η_t = η_{t-1} + Normal(0, σ_η²)
σ_η ~ HalfNormal(0.004 per day × √7)
```

`η = 0` means a tied generic ballot. The per-day scale of `0.004` accumulates to
`0.004 × √365 ≈ 0.076` logit (~3.8 points of margin) of drift over a year, which is the
right order for generic-ballot movement across a cycle.

### 2.2 Per race

```
θ_{r,t} = α_r + λ_r · η_t + ε_{r,t}
```

**`α_r` — baseline under a tied national environment.** This is the fundamentals prior
(§3). Defining it *at a tied environment* is what prevents double-counting: the national
swing enters through `η` and nowhere else.

**`λ_r` — elasticity.** How much of the national swing this race absorbs.
`TruncatedNormal(mean 1, sd 0.25, lower 0.2)`. Partially pooled around 1, and kept
strictly positive so the sign of the national environment cannot flip for one race.

**`ε_{r,t}` — race-specific drift.** A random walk pinned to zero at the grid start, so it
captures *movement* while `α` captures *level*. This is where candidate news, advertising
and scandal show up.

It is **sum-to-zero across races at every step**. A race's drift is its movement relative
to the field; the field's own movement is `η`'s job, fitted to hundreds of generic-ballot
polls. Left unconstrained, `ε` carries a free common component — a second national swing —
and the two trade off along a ridge the likelihood cannot resolve. The symptom was a House
forecast that did not move when ten states' new maps shifted its fundamentals by thirteen
seats, and a Senate fit whose tail ESS was a third lower than it needed to be.

### 2.3 The time grid

Anchored **backwards from election day**, so the final grid point is exactly
2026-11-03. Two consequences, both deliberate:

- The forecast target is a real grid point, not an interpolation.
- The grid does not shift underneath the model as new polls arrive.

Polls snap to the nearest grid point. At 7-day resolution the worst-case snapping error is
3.5 days, far below the week-to-week noise in the polls themselves.

Because the walks run all the way to election day, **drift between the last poll and the
election is already accounted for**. There is deliberately no separate "drift to election
day" term; adding one would double-count and inflate every interval.

---

## 3. Fundamentals prior

For each race, before any polls:

```
lean_r    = logit(state presidential share) − logit(national presidential share)
blended   = 0.75 · lean_2024 + 0.25 · lean_2020
α_prior_r = 0.95 · blended  ±  incumbency_r
α_r       ~ Normal(α_prior_r, 0.15²)
```

- **Recentring against the nation** turns a raw presidential result into a partisan
  *lean*, which is what generalises across cycles.
- **`0.95` shrinkage** reflects that Senate results track presidential lean closely but not
  perfectly, and safe states are slightly less lopsided downballot.
- **Incumbency** is `+0.060` logit (~3 points of margin) for an elected incumbent on the
  ballot, `+0.020` for an appointed one, `0` for an open seat — signed toward the party
  holding the seat. The modest size reflects the well-documented post-2016 decline in the
  personal incumbency advantage.
- **`prior_sd = 0.15`** (~7.5 points of margin) is wide on purpose. In well-polled races
  the polls dominate it; in unpolled races it is what carries the forecast.

**What it does not know:** candidate quality, fundraising, scandal, or ballot access.
Nebraska is the clearest illustration — Dan Osborn polls far better than Nebraska's
partisan lean, and only the polls can tell the model that.

### 3.1 The House: the same model, twelve times the races

Nothing under `model/` is Senate-specific. It takes whatever races the config gives it, so
the House is the same hierarchical model fitted to 435 units instead of 35, with the same
generic-ballot latent state, the same house effects, and the same election-day error
structure. What differs is the **evidence available per race** and what that implies.

**The data problem, and how it was solved.** A Senate race is a state, and state
presidential results are published. A congressional district is not an administrative unit
anyone reports results for, and district-level presidential results are the input every
serious House model needs. The usual solutions are to license a commercial dataset or to
run a GIS overlay of precinct shapefiles onto district boundaries — both were rejected as
a paid dependency and as a heavy one respectively.

Instead `scripts/build_house_lean.py` derives the lean from **MEDSL precinct returns
(CC0)** by joining the presidential and US House rows *within each precinct*, which needs
no geometry at all: the precinct already knows which district it voted in, because it cast
a House ballot. Votes in split precincts are allocated proportionally. This places 98.97%
of two-party presidential votes directly.

The two districts it cannot place that way — FL-20 and OK-03, whose 2024 House races were
uncontested and so have no precinct rows at all — are recovered by subtraction: where a
state is missing exactly one district, every unplaced presidential vote in it belongs to
that district. **All 435 districts have a measured lean**, and the chamber that falls out
of it (215 D / 220 R) matches reality exactly. See
`data/history/README_house_lean.md` for the five state-specific data quirks the builder
handles, the guard on the subtraction, and what is still approximate.

**Two consequences worth stating plainly:**

1. **`pres_2020` is not available under current lines.** 2020 results describe the
   pre-2022 districts, which are different places. Rather than blend across a
   redistricting, the 2020 field repeats the 2024 value, making the blend above a no-op
   for the House whatever weights it is given. This is stated in the generated config
   rather than left to be discovered.

2. **About nine districts in ten have no polls at all.** Where the Senate has 14 unpolled
   races out of 35, the House has roughly 397 out of 435. The chamber number is therefore
   overwhelmingly the seat-by-seat lean plus the national environment — which is what a
   House forecast *is*, and is why the generic ballot carries so much more weight here.
   The dashboard draws unpolled districts at reduced opacity so this is visible rather
   than merely documented.

**It scales.** Across three 435-race fits: `max_r_hat` 1.01–1.02, **zero divergences**,
minimum ESS 247–495, sampled in 679–701 s against the Senate's ~60 s. The
parameterisation holds; the cost is time, not geometry. (One caveat: the lowest tail ESS
seen, 247, is under the 400 floor quoted in §8 for the Senate. It varies run to run and
sits well above the point where estimates are unusable, but it is the thing to watch if
the House model is extended.)

**The central estimate is validated; the interval is not.** These are separate claims and
only the first is currently supported.

The seat median lands where the district leans say it should. With the model's own
generic-ballot estimate of D+6.9, a plain uniform swing over the 435 measured leans gives
**238** Democratic seats. The model gives **241**, and its expected value — the sum of the
per-district win probabilities — is 242.8. That agreement is the check that mattered: it
says 435 districts, almost none of them polled, are being carried correctly by their lean
and the national environment rather than by anything accidental.

The 90% interval is another matter. Uniform swing across the *generic ballot's own* 90%
range (D+5.2 to D+8.7) spans 233–245 seats — twelve seats. The model reports **195–296**,
a hundred and one. So essentially none of the width comes from uncertainty about where
opinion is now; almost all of it comes from the election-day error in §5.

Why that term is so much larger here than in the Senate, decomposed:

| | Senate | House |
|---|---|---|
| Races | 35 | 435 |
| Seats within 5 points | 10 | 51 |
| Seat-count SD if races were independent | 1.86 | 5.62 |
| Seat-count SD actually produced | 4.26 | 30.70 |
| Ratio | 2.29x | **5.46x** |
| Mean pairwise correlation | 0.291 | 0.307 |
| Effective independent units | 3.2 | 3.2 |

The correlation kernel is **not** misbehaving at scale — that was the thing to rule out,
and the mean pairwise correlation and its whole distribution are near-identical across the
two chambers. What changes is the arithmetic downstream of it. Roughly three independent
shocks of a few points each, applied to a chamber with 51 seats within five points of the
line, move far more seats than the same shocks applied to ten.

Whether the *magnitude* is right for districts is untested, and it is the open item in §9.
Two specific reasons for caution, in opposite directions:

- The kernel has **no state term**. Districts in one state share a state polling
  environment, state media, and statewide candidates, and the kernel knows only region and
  presidential lean. Adding it would push correlation, and the interval, *wider*.
- The 5.71-point idiosyncratic scale was fitted on *state* polling. With ~3.2 effective
  units it behaves as a near-common shock over roughly 136 districts at a time, which may
  overstate how much genuinely independent district-level error there is — pushing the
  interval *narrower*.

Until one of those is measured rather than argued, read the House seat interval as an
upper bound on precision, not a calibrated one. The dashboard says so on the page.

**There is no tie to break.** 435 is odd, so unlike the Senate one side always clears 218
and the Vice President never enters into it. `tiebreaker_party: none` in the config records
that rather than inventing one.

---

## 4. Measurement model

```
logit(yᵢ) ~ StudentT(ν=4, μᵢ, σᵢ)

μᵢ = θ_{rᵢ,tᵢ} + house_{pᵢ} + screen_{popᵢ} + ρ · partisanᵢ
σᵢ = √(vᵢ + σ²_excess)
```

### 4.1 House effects

`house_p` is estimated per pollster, hierarchically, and **constrained to sum to zero**.

The constraint is not cosmetic. Without it the model is not identified: you could add a
constant to every house effect and subtract it from the latent state and obtain exactly
the same likelihood, so the sampler would drift along that ridge indefinitely.

It is written **non-centred** — a unit-scale `ZeroSumNormal` multiplied by a separately
sampled `σ_house`. Scaling preserves the sum-to-zero property while decoupling the
geometry. Writing `ZeroSumNormal(sigma=σ_house)` directly produced a funnel: most
pollsters appear once or twice, their effects are prior-dominated, and the sampler then
has to squeeze through a narrowing neck as `σ_house` shrinks. In testing this change alone
moved minimum ESS from ~19 to ~670.

**Pollsters with fewer than 3 polls share a single pooled effect.** Most pollsters in the
feed appear once. Giving each a free parameter lets it absorb that poll's entire deviation
from the latent state, which is not a *house* effect — it is noise — and it makes each
individual poll far too influential. Pooling the long tail (63 of 116 pollsters, at the
time of writing) sends their deviation where it belongs.

### 4.2 Sampling variance

Delta method on the logit:

```
Var(logit(p̂)) ≈ 1 / (n_eff · p · (1−p)) × design_effect
```

with `design_effect = 1.5` for weighting and clustering, and

```
n_eff = n × (dem_pct + rep_pct) / 100
```

A poll of 1000 where 15% are undecided or backing a third candidate carries only about 850
respondents' worth of information about the D-vs-R split.

### 4.3 Screen and sponsor adjustments

- **Screen.** Likely voters are the reference (effect pinned to exactly zero, for
  identifiability); registered-voter and adult samples get estimated offsets. The model
  recovers roughly **+1.4 points of margin** for adult samples relative to likely voters,
  matching the usual finding.
- **Partisan sponsor.** `ρ · sᵢ` with `sᵢ ∈ {+1, 0, −1}`. `ρ` is constrained positive
  (partisan polls favour their sponsor) and shrunk toward zero. Estimated at roughly
  **+0.5 points of margin**.

### 4.4 Why Student-t, and why `σ_excess` ends up near zero

The `ν = 4` likelihood is for robustness: one wild poll should not drag the latent state.

`σ_excess` — extra poll-level error beyond sampling — consistently estimates near zero.
**This is expected, not a bug.** For a typical n=600 poll:

```
binomial margin SD              ≈ 4.1 pts
× √design_effect (1.5)          ≈ 5.0 pts   ← the t-distribution's scale
× √(ν/(ν−2)) = √2 for ν=4       ≈ 7.1 pts   ← its actual SD
```

Roughly 7 points of margin error per poll is already at or slightly above the historical
record for single polls at this distance from an election. There is simply no unexplained
variance left for `σ_excess` to claim. It is retained as a diagnostic — if a future cycle's
polls are noisier than the design effect implies, it will grow.

---

## 5. Election-day error

This is applied at **simulation time**, on top of the posterior, and it is the piece that
makes the chamber forecast honest.

A race-by-race model with independent errors produces a wildly overconfident seat
distribution: 35 near-independent coin flips concentrate hard around their mean. Real
polling misses are correlated — when the polls miss in Ohio they tend to miss the same way
in Iowa — and that correlation fattens the tails enormously.

```
θ_final_r = θ_{r,T} + b_national + κ_r

               Senate                    House
b_national ~ Normal(0, 0.065²)   Normal(0, 0.0756²)     3.3 / 3.8 pts of margin
κ          ~ MVN(0, 0.079² · C)  MVN(0, 0.0922² · C)    4.0 / 4.6 pts of margin
```

Total ≈ **5.1 points of margin** for the Senate, **6.0** for the House.

**These are fitted per chamber, not asserted and not shared.** `midterms calibrate
[--chamber house]` decomposes observed poll error into three levels — a cycle-wide
national miss, a race-specific miss, and poll noise — after removing the poll noise each
race's mean error still carries. The earlier values (0.075 / 0.085) came from the
literature and were wrong in a specific way: they understated total race-level error by
14% and, more damagingly, mis-split it, overstating the perfectly-correlated national
component by 20% while understating the race-specific one by 34%.

**The window is 0–14 days, not the forecast horizon**, and that is deliberate. This term
is the systematic miss the polls were always going to make, with drift stripped out —
drift lives in the random walk, where it grows with the time remaining. Fitting at 45–120
days folds roughly 3.9 points of drift in here, which would make the total about right
today while guaranteeing it stayed just as wide on the eve of the election.

Zero to fourteen days is not zero days, so a sliver of drift does survive into this term
and is counted twice — the walk runs to election day and already carries those fourteen
days. At the national scale of `0.004` logit per day that is `0.004·√14 ≈ 0.75` points,
which would take `national_sd` from 3.78 to 3.71: **2%**. Left alone rather than corrected,
because the quantity that matters is the total, and the total is checked against outcomes
below (3.42 modelled against 3.46 measured). Narrowing the window further would trade a
2% bias for a much noisier estimate.

### 5.1 Why the House gets its own scales

Because they are measurably different, and because using the Senate's was never a choice
anyone made — it was what happened when a second chamber was added to a model that had
only ever had one.

| Fitted at 0–14 days, 2010– | Senate | House |
|---|---|---|
| national (cycle-wide) | 3.32 | **3.78** |
| race-specific | 3.96 | **4.61** |
| poll noise | 3.97 | **5.85** |
| design effect | 1.18 | **1.04** |

Both election-day scales are **larger** for the House, so calibrating it correctly made
the seat interval *wider*. That is worth stating because the intuition runs the other
way: a 100-seat interval looks like something to shrink.

The design effect goes the opposite direction, and the two facts are consistent. House
polls carry far more raw noise (5.85 against 3.97) but at a median sample of 496 against
735 — so once sample size is accounted for they are, if anything, marginally *tighter*
than binomial. The sampling-variance term already knows about sample size; carrying the
Senate's 1.18 was inflating district poll variance by 13% and discounting the only direct
evidence 38 districts have. Note the design effect is fitted at **45–120 days**, where the
model actually weighs polls, while the election-day scales are fitted at 0–14; mixing the
two bases would make the chambers incomparable.

**Independent validation of the national term.** The House national component can be
checked against something the Senate's cannot: the actual national House vote. Across 13
cycles (1998–2022) the generic ballot's error has an RMSE of 3.46 points about zero. The
model carries 3.42 — 3.25 of election-day error plus 1.06 of latent drift to election day.
That is the term which dominates the House seat interval, contributing roughly 58 of its
101 seats at ~5.2 seats per point, and it is right.

**The bias, however, is not modelled.** Those 13 cycle errors average **+2.47 points
toward Democrats**, positive in 9 of 13. The model's national error is mean-zero, so its
spread covers this but its centre does not. See known weakness 13.

The split matters as much as the total. National error moves all 35 races together and
fattens the tails of the seat distribution; state error partly cancels across the map.
Getting the mix wrong changes the *shape* of the chamber forecast without changing any
single race's win probability.

The correlation matrix is

```
C_{ij} = (1 − nugget) · exp(−d_{ij} / ℓ) + nugget · [i = j]
```

where `d` is Euclidean distance in standardised covariate space — 2024 lean, 2020 lean,
and a scaled region indicator — with `ℓ = 1.25`. The exponential kernel is positive
definite for Euclidean distance (it is the Matérn-½ kernel), and the nugget keeps it
comfortably so, which matters because it gets Cholesky-factored on every run.

**Limitation:** the covariates are political (past presidential results plus region) rather
than demographic. Education and urbanicity have been the axes along which recent polling
errors actually correlated. Adding demographic covariates is the single highest-value
improvement available, and was deferred here rather than fabricated from unverified
numbers.

---

## 6. From posterior to seats

```
dem_seats = seats_not_up_D + Σ_r 1[θ_final_r > 0]
```

With 34 Democratic-caucus seats not up and 31 Republican, Democrats need **51** of 100 for
control and Republicans need **50**, because the Vice President breaks a 50-50 tie for the
Republicans. `tests/test_config.py` asserts this arithmetic closes to 100 and matches the
per-race incumbent parties — a miscounted baseline is the most damaging silent error a seat
model can have, because it shifts every control probability without changing any
individual race.

Each posterior draw is recycled `sims_per_draw = 10` times with fresh election-day error,
giving 40,000 simulations from 4,000 draws.

### Tipping point

In each simulation, order the contested races from most to least Democratic and walk down
accumulating seats. The race at which the running total reaches the majority threshold
decided the chamber. Aggregated across simulations this gives a **tipping-point
probability** per race — usually more informative than any single race's win probability,
because it answers "which seat would both parties most want?"

---

## 7. Sampling

- **nutpie** (Numba-compiled NUTS). No C++ toolchain required, so Windows and Linux CI run
  the identical implementation.
- 1,000 draws × 4 chains, 1,500 tuning steps, `target_accept = 0.95`. The generous tuning
  is for the variance-scale parameters (`σ_ε`, `σ_η`), which sit near zero where the
  geometry is tightest.
- Typical run: **~45 seconds**, `max R̂ ≈ 1.01`, `min ESS ≈ 660`, **zero divergences**.

Diagnostics are written into every `forecast.json` and displayed on the dashboard. A
forecast whose sampler did not converge is not a forecast, and the CLI emits a loud warning
when `R̂ > 1.05` or any divergence occurs.

---

## 8. Calibration

Real backtesting is impossible before November 2026. What is possible continuously is
**out-of-sample poll prediction**:

1. Refit using only polls fielded before `today − holdout_days`.
2. Form the posterior predictive for each poll fielded after the cutoff, including house
   effects, screen adjustments, and the measurement noise the model believes in.
3. Measure interval coverage and PIT uniformity.

If the 80% intervals contain 80% of held-out polls, the uncertainty is about right. The
PIT values sharpen this: for a well-calibrated model they are uniform on `[0, 1]`, and
their departure from uniformity says *how* the model is wrong, not merely that it is.

```bash
midterms backtest --holdout-days 30
```

### Measured result (2026-08-20, 33 held-out polls)

```
Mean   |error| vs poll margin:  3.66 pts
Median |error| vs poll margin:  2.27 pts

50% interval -> 69.7%   UNDERCONFIDENT
80% interval -> 87.9%   UNDERCONFIDENT
90% interval -> 97.0%   well calibrated
95% interval -> 97.0%   well calibrated

PIT KS statistic: 0.120
```

The intervals are **too wide at the poll level**, not too narrow. A typical held-out
poll misses by 2–4 points of margin, against a predictive SD of roughly 7 points
(§4.4). The likely culprits, in order: `design_effect = 1.5` and the heavy `ν = 4`
tail, which compound to inflate the SD by about 1.7× over binomial.

This has **not** been retuned, deliberately. With n=33 the standard error on the 50%
coverage figure is about 8.7 points, so the result is suggestive rather than decisive,
and tightening the likelihood to fit 33 polls would be fitting the diagnostic instead
of the phenomenon. The correct next step is to accumulate held-out polls across daily
runs and revisit `design_effect` and `ν` once the sample supports it.

Two things this does *not* imply. First, being underconfident about **polls** is much
safer than being overconfident. Second, it says little about the chamber forecast: the
seat distribution's width is dominated by the correlated election-day error of §5,
which this diagnostic cannot test at all, because it is calibrated to historical
election outcomes rather than to polls.

This is a **necessary** condition for calibration, not a sufficient one. A model can
predict polls beautifully and still be wrong about the election if the polls themselves are
biased. That residual risk is exactly what §5 represents, and it can only be validated
against real results.

---

## 8b. Backtest against real elections

`midterms backtest-history` scores win probabilities against 167 Senate races from
2010–2022 — the check that `backtest.py` cannot make, because predicting polls is not
predicting elections.

Fitted scales versus the previously asserted ones, on identical races:

| Coverage of the actual margin | Fitted 3.0/5.7 | Asserted 3.75/4.25 | Target |
|---|---|---|---|
| 50% interval | 44.3% | 40.7% | 50% |
| 80% interval | 70.7% | 65.9% | 80% |
| 90% interval | 84.4% | 77.8% | 90% |

The fitted scales are better at every level, and reliability sits close to the diagonal:
races called at 42% won 39% of the time, at 59% won 53%, at 82% won 86%, at 98% won 98%.
Brier score 0.091, a 34% improvement on always calling the poll leader.

**Two things this backtest does not say.** First, its point estimate is a
recency-weighted poll average, not the full posterior, so its error is an upper bound on
the model's. Second — and this is the trap — it wants roughly 1.2× the fitted scales to
be perfectly calibrated, but that factor must **not** be copied into the config. It is
absorbing the point estimate's own error, which the real model carries separately in its
posterior. Measured directly: the model's posterior contributes a median 4.9 points of
SD per race, which combined with the 6.4-point election-day error gives 8.1 points
total, against the 7.7 the backtest finds well-calibrated — a ratio of 1.05. Inflating
the config would double-count.

A fat-tailed (Student-t) election-day error was tested and **rejected**: it made coverage
worse at every level. The residual gap is width, not shape.

## 9. Known weaknesses

0. ~~**The House model is built on the wrong district map.**~~ **Fixed.** Ten states
   redrew their congressional boundaries mid-decade for 2026 — Texas, California, Florida,
   Ohio, North Carolina, Missouri, Tennessee, Louisiana, Alabama and Utah, covering **181
   of 435 seats**. Every lean is now measured on the new lines. Harris-carried districts
   fall from 203 under the old map to 200 under the new one, and per state the change
   tracks published estimates (CA +6, FL −4, TX −3, UT +1).

   The precinct-join method of §3.1 could not produce these, and not through any fault of
   its own: it joins a precinct's presidential row to that precinct's own *2024 US House
   row*, and no 2024 election was held under the new boundaries. The numbers come instead
   from The Downballot, which has **no stated licence** — the one dependency in the
   project without a clear one, recorded as a deliberate trade in
   [DATA_SOURCES](DATA_SOURCES.md) rather than presented as equivalent to the CC0 sources.

   The swap also exposed a defect in our own data. Cross-checking the two sources on the
   254 districts whose lines did not change — where they must agree — found 35 of 40
   states agreeing to within half a point, and five that did not, every one of them our
   error. Oregon was 7.00 points off on average and OR-03 (Portland) by **30 points**;
   Washington 2.63. Both vote entirely by mail and report in aggregated batches rather
   than by polling place, so the join had far less to work with than its 98.97% placement
   rate implied. That was live in a published forecast and invisible until two independent
   sources were compared.

1. **Correlation covariates are political, not demographic** (§5). Highest-value fix.
2. **The fundamentals prior has no candidate-quality term.** No fundraising, no scandal, no
   prior-office measure.
3. **Fourteen races have no qualifying polls.** They rest entirely on fundamentals and the
   national environment; their intervals are wide but their centres are only as good as the
   prior.
4. ~~The historical error scales in §5 are asserted from the literature.~~ **Fixed** —
   fitted from FiveThirtyEight's archived pollster-ratings dataset (§5, §8b). The
   fundamentals coefficients in §3 are still asserted and are the next candidate.
5. **A single elasticity per race** assumes the relationship to the national environment is
   constant over the cycle.
7. **Presidential approval is ingested but disabled.** The link is implemented and
   works — the fitted correlation between approval and generic-ballot innovations is
   −0.54, correctly negative — but A/B testing showed it tightens the national
   environment by only 6.6% while cutting minimum effective sample size from 455 to 296.
   Not worth the geometry at present. `scripts/ab_approval.py` reproduces the test.
8. ~~**Correlated race movement costs sampling efficiency.**~~ **Fixed, and the note was
   describing the cause without naming it.** "Competes with the national environment for
   the common signal" was precisely the problem: `eps` was free to move every race the
   same way at once, and a common component of `eps` *is* a national swing — one the model
   already has in `eta`. The two traded off along a ridge the likelihood cannot resolve.

   `eps` is now sum-to-zero across races at every step, the same constraint house effects
   have carried all along and for the same reason (§ identifiability rule 4 in
   `hierarchical.py`). On the Senate, minimum tail ESS went from 604 to **953** and bulk
   from 582 to 632, with divergences still zero.

   It was not only a sampling cost. Because the ridge let `eps` absorb level, the House
   forecast did not move when ten states' new maps shifted its fundamentals thirteen seats
   — and 38 polled districts running 2.6 points more Democratic than their fundamentals
   were setting the level for all 397 unpolled ones.
6. **Nebraska's independent** is forced onto the Democratic side for chamber arithmetic
   (§ README known issues). There is no unarguably correct treatment.
9. ~~**The House interval is calibrated only by inheritance.**~~ **Fixed, and it went the
   other way.** The scales are now fitted on House polling (§5.1): national 3.78 against
   the Senate's 3.32, race-level 4.61 against 3.96. Both larger, so the interval widened.
   The national component is independently validated against 13 cycles of generic-ballot
   error — the model carries 3.42 points against a measured RMSE of 3.46. What remains
   open is the correlation kernel, below.
10. ~~**Two districts have no measured lean.**~~ **Fixed.** FL-20 and OK-03 held
    uncontested 2024 House races, so their presidential votes had no House row to join to.
    Both are now recovered by subtraction — where a state is missing exactly one district,
    the unplaced presidential votes in it belong to that district. All 435 have a measured
    lean, and the derived chamber (215 D / 220 R) matches reality exactly.

11. **The correlation kernel is shared across chambers and confirmed for neither.**
    Its region contrast (+0.230, demeaned) sits inside the 90% bootstrap interval of the
    measured one for both chambers — House +0.088 [+0.035, +0.284], Senate +0.167
    [+0.047, +0.490]. Seven cycles cannot separate them, so it is not demonstrably wrong
    and not demonstrably right. **This is now the largest open item**, and it matters more
    for the House because 51 seats sit within five points of the line against ten in the
    Senate. Note the trap documented in `scripts/fit_correlation.py` and re-encountered
    here: comparing the raw kernel against demeaned measurements makes it look 5x
    over-correlated, and shrinking it on that basis would make the interval far too narrow.

12. **The kernel has no state term.** Districts in one state share a state polling
    environment, state media markets and statewide candidates; the kernel knows only
    region and presidential lean. Adding it would widen the House interval rather than
    narrow it, so it is a correction to the model rather than to the width complaint.

13. **The generic ballot has a systematic pro-Democratic bias the model does not
    correct.** Across 13 cycles it overstated the Democratic national House margin by an
    average of 2.47 points, positive in 9 of 13. The model treats national error as
    mean-zero, so its *spread* covers this (3.42 against a measured RMSE of 3.46 about
    zero) but its *centre* does not. Correcting it would move the House forecast down by
    roughly 13 seats. Not done here: house effects and the partisan-sponsor adjustment
    already absorb an unknown share of it, and VoteHub's pollster mix is not the
    historical one, so the correction is not simply −2.47. It is the most consequential
    unaddressed item in this document.
