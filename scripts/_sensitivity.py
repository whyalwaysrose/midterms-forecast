"""Re-simulate the Senate from a published payload, for sensitivity questions.

Shared by ``compare_to_markets.py`` and ``compare_to_forecasters.py``. Both need
the same thing: what the chamber probability would be under some counterfactual
-- most often Nebraska's independent not caucusing with the Democrats -- without
refitting the model, which takes twenty minutes and is not available for an
archived run at all.

WHY THE POSTERIOR SPREAD HAS TO BE PUT BACK
-------------------------------------------
The obvious approach is to take each race's published ``margin.p50``, add the
config's election-day error, and count seats. That is what this did originally,
and it is wrong in a way that hides well.

The published per-race margins come from ``SimulationResult.margin_quantiles``,
which is computed from ``simulated_logit`` -- the draws *after* election-day
error has been added. Taking the median is fine, because adding symmetric
zero-mean noise does not move a median. But rebuilding a distribution as
``median + election-day error`` gives it the variance of the election-day error
alone, and throws away the posterior spread the real simulation carries
underneath: a median 4.5 points of SD per race, against the election-day
error's 5.1 (§8b of METHODOLOGY quotes 4.9 and 6.4 on a slightly different
basis).

Too narrow a distribution pushes a probability away from 50%, so the error is
invisible when the forecast is near a coin flip and grows as it moves away.
Measured on 2026-09-14, with the Senate at 67.6%:

    median + election-day error only   0.6965   (2.01 points off)
    posterior spread restored          0.6768   (0.04 points off)

``compare_to_markets.py`` documented a 0.1-point tolerance, which held when it
was written and the forecast sat near 56%. It had quietly stopped holding.

The posterior spread is recovered per race from the published 90% interval,
which already contains everything: total variance minus the election-day
variance leaves the posterior's. It is then drawn independently across races,
which understates how much of it is a shared national component -- so this
remains an approximation, and the caller is expected to check the result
against the payload's own headline before trusting a number derived from it.
``reproduces_headline`` exists for exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from midterms import fundamentals as F  # noqa: E402
from midterms.config import load_all  # noqa: E402
from midterms.model.correlation import cholesky_factor, correlation_matrix  # noqa: E402

DRAWS = 200_000

#: Half-width of a 90% interval in standard deviations.
Z90 = 1.6449

#: How far the re-simulation may land from the payload's own headline before a
#: number derived from it should not be quoted. The re-simulation is an
#: approximation; this is the check that it is still approximating the model
#: rather than something else.
HEADLINE_TOLERANCE = 0.01


def senate_both_ways(payload: dict) -> tuple[float, float] | None:
    """P(D Senate) counting Nebraska's independent, and not counting him.

    Returns None when the payload's roster does not match the current config,
    because a race since added or dropped would silently change what is being
    compared. Archived payloads carry no ``unit`` field, so Nebraska is
    identified from its race id.
    """
    races, cfg = load_all(chamber="senate")
    by_id = {r["id"]: r for r in payload.get("races", [])}
    if not by_id or any(r.id not in by_id for r in races.races):
        return None

    fundamentals = F.compute(races, cfg)
    chol = cholesky_factor(
        correlation_matrix(fundamentals, cfg.election_day_error.correlation)
    )

    margins = [by_id[r.id]["margin"] for r in races.races]
    base = np.array([m["p50"] for m in margins]) / 50.0

    ede = cfg.election_day_error
    # What the published interval implies in total, less what the election-day
    # error contributes, is what the posterior contributes. Clipped at zero: a
    # race whose interval is narrower than the election-day error alone would
    # imply a negative variance, which means the interval is telling us
    # something other than what this arithmetic assumes.
    total_sd = np.array([(m["p95"] - m["p05"]) / (2 * Z90) for m in margins]) / 50.0
    ede_var = ede.national_sd**2 + ede.state_sd**2
    posterior_sd = np.sqrt(np.clip(total_sd**2 - ede_var, 0.0, None))

    rng = np.random.default_rng(9)
    national = rng.normal(0.0, ede.national_sd, (DRAWS, 1))
    z = rng.standard_normal((DRAWS, len(base)))
    posterior = rng.standard_normal((DRAWS, len(base))) * posterior_sd
    wins = (base + national + ede.state_sd * (z @ chol.T) + posterior) > 0

    control = payload["chamber_forecast"]
    held, need = control["seats_not_up"]["D"], control["dem_seats_for_majority"]
    keep = [i for i, r in enumerate(races.races) if not r.id.endswith("-NE")]
    return (
        float(((held + wins.sum(axis=1)) >= need).mean()),
        float(((held + wins[:, keep].sum(axis=1)) >= need).mean()),
    )


def reproduces_headline(payload: dict, simulated: float) -> tuple[bool, float]:
    """Is the re-simulation still measuring the model? Returns (ok, drift)."""
    published = payload["chamber_forecast"]["dem_control_prob"]
    drift = abs(simulated - published)
    return drift <= HEADLINE_TOLERANCE, drift
