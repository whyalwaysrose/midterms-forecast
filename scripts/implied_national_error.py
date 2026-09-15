"""Why is this model's House interval so much wider than other people's?

50+1 publishes a median House outcome of 236 seats. So does this model. Identical
centre, 97% against 71.9% -- so the entire difference is the width of the
distribution around that centre, which makes the disagreement unusually
tractable. This script asks which parameter it lives in.

THE HYPOTHESIS THIS WAS BUILT TO TEST, AND WHY IT IS WRONG
----------------------------------------------------------
`scripts/kernel_sensitivity.py` had already shown the width is not the
correlation kernel's doing: zero the kernel entirely and the 90% seat span only
falls from 89 to 65. It attributed the floor to the national election-day error,
one miss applied to every district at once. The obvious next step was to invert
each rival's probability into the national error it implies and put those next
to the thirteen cycles of generic-ballot misses the config was fitted against.

That inversion does not work, and the reason is the finding:

    with the national term at 0.5 points -- effectively none -- this model
    still only reaches 82.5%.

No national-error assumption reaches 97%, or even 86%. The national term is
worth about 32 seats of the 100-seat span and roughly nine points of
probability. It is not where the disagreement lives.

WHAT THE DISAGREEMENT ACTUALLY LIVES IN
---------------------------------------
Independent per-race error averages out across 435 seats; correlated error does
not. So the chamber's width is set almost entirely by the two *correlated*
terms together -- the national miss and the kernel-correlated state term -- and
the per-race posterior, wide as it is per district (a median 8.9 points on the
397 districts with no poll), contributes only about five seats of span.

The single knob that matters is therefore the total correlated election-day
error, and inverting on that gives every rival a number instead of "off scale".

METHOD
------
Re-simulate from the published per-race margins, scaling both correlated terms
by a common factor and holding the posterior fixed. Base margins come from the
forecast rather than a refit, so this measures sensitivity around the actual
answer with the right distribution of near-line districts.

The posterior has to be put back, and this is the trap `_sensitivity.py`
documents: published margins come from `margin_quantiles`, computed *after*
election-day error is added, so rebuilding a distribution as `median +
election-day error` gives it the election-day variance alone. It is recovered
per race from the published 90% interval at the committed scales, then held
fixed -- the posterior is a property of the fit, not of the counterfactual.

Two honest limits on what follows. The posterior is drawn independently across
races, when the real one shares a national component through `eta`; that is why
the committed row reproduces the headline to about a point rather than exactly.
And a rival's implied scale is not a claim about what that forecaster actually
does -- it is what this model's machinery would need in order to agree with
them.

Usage:
    python scripts/implied_national_error.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from midterms import fundamentals as F  # noqa: E402
from midterms import paths  # noqa: E402
from midterms.config import load_all  # noqa: E402
from midterms.model.correlation import cholesky_factor, correlation_matrix  # noqa: E402

DRAWS = 60_000
CHUNK = 10_000          # 435 races x 60k draws in one array is ~200 MB
Z90 = 1.6449

#: One logit unit is 50 points of margin, throughout this project.
PTS = 50.0

#: House probabilities to invert. The three forecasts are from
#: data/rivals/forecasts.csv; the Polymarket price is from the same day's
#: snapshot and is a market rather than a model, shown for context.
RIVALS = (
    ("50+1", 0.97),
    ("Polymarket", 0.860),
    ("this model", 0.7187),
    ("DDHQ", 0.68),
)

#: The generic ballot's own measured miss across thirteen cycles
#: (METHODOLOGY 5.1). The config's national term was fitted against this.
MEASURED_RMSE_PTS = 3.46


def seat_draws(base, posterior_sd, chol, national_sd, state_sd, rng,
               use_national=True, use_state=True, use_posterior=True):
    """Simulated Democratic seat totals under the given error scales."""
    n = len(base)
    seats = np.empty(DRAWS, dtype=np.int32)
    for start in range(0, DRAWS, CHUNK):
        size = min(CHUNK, DRAWS - start)
        x = np.broadcast_to(base, (size, n)).copy()
        if use_national:
            x += rng.normal(0.0, national_sd, (size, 1))
        if use_state:
            x += state_sd * (rng.standard_normal((size, n)) @ chol.T)
        if use_posterior:
            x += rng.standard_normal((size, n)) * posterior_sd
        seats[start : start + size] = (x > 0).sum(axis=1)
    return seats


def main() -> int:
    races, cfg = load_all(chamber="house")
    payload = json.loads(
        (paths.SITE_DATA_DIR / "forecast_house.json").read_text(encoding="utf-8")
    )
    control = payload["chamber_forecast"]
    need = control["dem_seats_for_majority"]
    published = control["dem_control_prob"]

    by_id = {r["id"]: r for r in payload["races"]}
    if any(r.id not in by_id for r in races.races):
        print("Published roster does not match the config; refusing to guess.")
        return 1

    margins = [by_id[r.id]["margin"] for r in races.races]
    base = np.array([m["p50"] for m in margins]) / PTS
    total_sd = np.array([(m["p95"] - m["p05"]) / (2 * Z90) for m in margins]) / PTS

    ede = cfg.election_day_error
    # The model widens the national term by the generic-ballot correction's own
    # standard error, in quadrature, so the committed value is the hypotenuse.
    bias_se = getattr(cfg.national_environment, "generic_ballot_bias_se", 0.0) or 0.0
    national = float(np.hypot(ede.national_sd, bias_se))
    state = ede.state_sd

    posterior_sd = np.sqrt(
        np.clip(total_sd**2 - (national**2 + state**2), 0.0, None)
    )
    chol = cholesky_factor(correlation_matrix(F.compute(races, cfg), ede.correlation))

    polled = np.array([by_id[r.id].get("poll_count", 0) > 0 for r in races.races])

    def run(scale=1.0, **flags):
        return seat_draws(base, posterior_sd, chol, national * scale,
                          state * scale, np.random.default_rng(11), **flags)

    def summarise(seats):
        lo, hi = int(np.quantile(seats, 0.05)), int(np.quantile(seats, 0.95))
        return float((seats >= need).mean()), int(np.median(seats)), lo, hi

    print("=" * 74)
    print("  WHERE THE HOUSE INTERVAL'S WIDTH ACTUALLY COMES FROM")
    print("=" * 74)
    print(f"  Published run {payload['run_date']}: D {published:.1%}, "
          f"median {control['dem_seats']['median']} seats, "
          f"90% {control['dem_seats']['p05']}-{control['dem_seats']['p95']}")
    print(f"  Correlated error: national {national * PTS:.2f} pts, "
          f"state {state * PTS:.2f} pts")
    print(f"  Posterior SD: {np.median(posterior_sd[polled]) * PTS:.2f} pts on the "
          f"{polled.sum()} polled districts, "
          f"{np.median(posterior_sd[~polled]) * PTS:.2f} on the {(~polled).sum()} unpolled")
    print()

    # --- check ---------------------------------------------------------------
    prob, median, lo, hi = summarise(run())
    drift = abs(prob - published)
    print(f"  CHECK: re-simulation at the committed scales gives {prob:.1%} "
          f"against a published {published:.1%} (drift {drift:.1%})")
    if drift > 0.02:
        print("  Too far off to interpret. Stop here.")
        return 1
    print()

    # --- component decomposition ---------------------------------------------
    print("  Each error term switched off in turn:")
    print(f"  {'':<18}{'P(D House)':>11}{'median':>8}{'90% interval':>15}{'span':>6}")
    print("  " + "-" * 58)
    for label, flags in (
        ("all three", {}),
        ("no national", {"use_national": False}),
        ("no state/kernel", {"use_state": False}),
        ("no posterior", {"use_posterior": False}),
    ):
        p, m, lo, hi = summarise(run(**flags))
        print(f"  {label:<18}{p:>10.1%}{m:>8}{f'{lo}-{hi}':>15}{hi - lo:>6}")

    p_nonat, _, _, _ = summarise(run(**{"use_national": False}))
    print()
    print(f"  Independent per-race error averages out across 435 seats and "
          f"correlated\n  error does not, which is why a posterior of ~9 points a "
          f"district is worth\n  only a few seats of span. Even with the national "
          f"term gone entirely this\n  model reaches {p_nonat:.1%} -- so no national-"
          f"error assumption explains a 97%.")
    print()

    # --- invert on total correlated error ------------------------------------
    print("  Scaling BOTH correlated terms by a common factor:")
    print(f"  {'scale':>7}{'national':>10}{'state':>8}{'P(D House)':>12}"
          f"{'90% interval':>15}{'span':>6}")
    print("  " + "-" * 58)
    scales = np.concatenate([np.arange(0.0, 1.01, 0.1), [1.25, 1.5]])
    curve = []
    for scale in scales:
        p, m, lo, hi = summarise(run(scale=scale))
        curve.append(p)
        if abs(scale * 10 % 2) < 1e-9 or abs(scale - 1.0) < 1e-9:
            marker = "  <- committed" if abs(scale - 1.0) < 1e-9 else ""
            print(f"  {scale:>7.2f}{national * PTS * scale:>10.2f}"
                  f"{state * PTS * scale:>8.2f}{p:>11.1%}"
                  f"{f'{lo}-{hi}':>15}{hi - lo:>6}{marker}")
    curve = np.array(curve)

    print()
    print("  What each forecast implies this model would need:")
    print(f"  {'forecaster':<14}{'P(D House)':>11}{'implied scale':>15}"
          f"{'national':>10}{'state':>8}")
    print("  " + "-" * 58)
    order = np.argsort(curve)
    for name, prob_target in RIVALS:
        if prob_target > curve.max():
            print(f"  {name:<14}{prob_target:>10.1%}"
                  f"{'below 0.00 — unreachable':>39}")
            continue
        if prob_target < curve.min():
            print(f"  {name:<14}{prob_target:>10.1%}{'above ' + f'{scales[-1]:.2f}':>15}")
            continue
        scale = float(np.interp(prob_target, curve[order], scales[order]))
        print(f"  {name:<14}{prob_target:>10.1%}{scale:>15.2f}"
              f"{national * PTS * scale:>10.2f}{state * PTS * scale:>8.2f}")

    print()
    print(f"  The national term alone was fitted against a measured "
          f"{MEASURED_RMSE_PTS:.2f} points\n  RMSE of generic-ballot miss over "
          f"thirteen cycles, so a scale much below 1\n  is a claim that midterm "
          f"polling is more reliable than that record.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
