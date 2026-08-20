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
7. **`site/data/*.json` and `outputs/runs/**` must stay committed.** The daily workflow
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

## Known rough edges

Listed in the README under "Known data-quality items" and `docs/METHODOLOGY.md` §9. The
highest-value improvement is demographic covariates (education, urbanicity) in the
election-day correlation kernel — currently it uses political covariates only.
