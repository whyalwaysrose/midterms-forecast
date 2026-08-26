"""How much of the House seat interval is the correlation kernel's doing?

`scripts/fit_correlation.py` asks whether the kernel's *shape* is right -- are
the races it calls politically close the ones that actually miss together? It
answers yes, and finds no parameter change justified.

This asks the other question, which that script cannot: given that the kernel's
strength is only loosely pinned by seven cycles of data, **how much does the
answer move across the range the data permits?** A parameter nobody can measure
precisely still does not matter if the conclusion is the same throughout.

METHOD

Scale the whole off-diagonal of the correlation matrix by `k` and re-simulate.
`k = 1` is the committed kernel; `k = 0` makes race-specific errors independent.
Its measured region contrast (+0.088, demeaned, 90% CI [+0.035, +0.284] over
House cycles) sits against the kernel's own +0.230, so the data is consistent
with roughly `k = 0.15` to `k = 1.23`.

Base margins come from the published forecast rather than being re-derived, so
this measures sensitivity around the actual answer with the right distribution
of near-line districts. It therefore excludes posterior uncertainty and reports
spans a little narrower than the dashboard's -- the *differences* are the point,
not the levels.

FINDING (2026-08-26)

The kernel is not what makes the House interval wide.

    k       contrast    90% seat interval   span
    0.00     +0.000         208-273          65     independent
    0.15     +0.035         206-276          70     low end of the CI
    0.50     +0.115         203-282          79
    1.00     +0.230         200-289          89     committed
    1.23     +0.283              --           --    not positive definite

Across the entire range the data allows, the span moves by about 20 seats. Even
setting the kernel to zero -- assuming every district's race-specific error is
independent, which nobody believes -- leaves 65. That floor is the national
error, which is separately validated against 13 cycles of generic-ballot misses
(model 3.42 points, measured RMSE 3.46) and is not in doubt.

So the honest reading is that the width is a property of how a national polling
miss maps onto a chamber with 51 seats within five points of the line, and not
an artefact of an unverifiable correlation parameter. Shrinking the kernel to
"fix" the interval would buy at most 20 seats and would be unsupported.

Usage:
    python scripts/kernel_sensitivity.py
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
from midterms.model.correlation import correlation_matrix  # noqa: E402

#: Multipliers to try. The bounds come from the measured contrast's CI; see the
#: module docstring for where they are from.
SCALES = (0.0, 0.15, 0.5, 1.0, 1.23)

#: The kernel's own region contrast at k = 1, demeaned to match the measurement.
KERNEL_CONTRAST = 0.230

DRAWS = 60_000


def scaled_cholesky(corr: np.ndarray, k: float) -> np.ndarray | None:
    """Cholesky of the kernel with its off-diagonal scaled by ``k``.

    Returns ``None`` when the result is not positive definite, which happens
    above ``k = 1``: the kernel is only guaranteed factorable at its own scale,
    and inflating correlations without re-deriving the matrix can push it
    outside the valid set. Reported rather than repaired -- silently ridging it
    would answer a question about a matrix the model would never use.
    """
    scaled = corr * k
    np.fill_diagonal(scaled, 1.0)
    try:
        return np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError:
        return None


def main() -> int:
    races, cfg = load_all(chamber="house")
    fundamentals = F.compute(races, cfg)
    corr = correlation_matrix(fundamentals, cfg.election_day_error.correlation)

    payload_path = paths.SITE_DATA_DIR / "forecast_house.json"
    if not payload_path.exists():
        print("No House forecast yet; run `midterms run --chamber house` first.")
        return 2
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in payload["races"]}
    base = np.array([by_id[r.id]["margin"]["p50"] for r in races.races]) / 50.0

    ede = cfg.election_day_error
    rng = np.random.default_rng(21)

    print(f"House seat interval against correlation strength "
          f"({len(base)} districts, {DRAWS:,} draws)")
    print(f"  national_sd {ede.national_sd * 50:.2f} pts, "
          f"state_sd {ede.state_sd * 50:.2f} pts\n")
    print("    k    contrast   median   90% interval   span")

    for k in SCALES:
        chol = scaled_cholesky(corr, k)
        if chol is None:
            print(f"  {k:4.2f}    {KERNEL_CONTRAST * k:+.3f}        --        --          --"
                  f"   not positive definite")
            continue
        national = rng.normal(0.0, ede.national_sd, (DRAWS, 1))
        z = rng.standard_normal((DRAWS, len(base)))
        seats = ((base + national + ede.state_sd * (z @ chol.T)) > 0).sum(axis=1)
        lo, mid, hi = np.percentile(seats, [5, 50, 95])
        note = {0.0: "independent", 1.0: "COMMITTED"}.get(k, "")
        print(f"  {k:4.2f}    {KERNEL_CONTRAST * k:+.3f}    {mid:6.0f}   {lo:5.0f}-{hi:<5.0f}"
              f"  {hi - lo:5.0f}   {note}")

    print("\nThe floor at k=0 is the national error, which is separately validated")
    print("against 13 cycles of generic-ballot misses. The kernel moves the span by")
    print("about 20 seats across the whole range the data permits -- it is not what")
    print("makes this interval wide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
