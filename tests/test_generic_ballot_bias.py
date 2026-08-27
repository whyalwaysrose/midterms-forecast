"""The generic ballot is not the national vote, and the gap is corrected.

`eta` is fitted to generic-ballot polls, so it estimates *the generic ballot*
faithfully. The model then uses it as the national level for every race, which
is a different quantity — and across 2010-2022 the generic ballot ran 1.65
points more Democratic than the vote actually cast.

The tests that matter here are about **scope and honesty**, because both are
easy to get wrong in ways nothing would raise:

* Applied to the wrong chamber, it silently biases 35 Senate races by a figure
  measured against House results.
* Applied without its standard error, a seven-cycle estimate is presented as a
  constant.
* Applied at election-day simulation instead of inside `theta`, it would move
  races that their own unbiased polls have already pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from midterms import calibration as C
from midterms.config import ModelConfig

# --- what was measured ------------------------------------------------------


def test_only_the_generic_ballot_is_biased():
    """The premise of the whole correction, and the reason for its shape.

    If race-level polls were biased too, this would be a polling correction and
    would belong at election-day simulation, applied to every race alike. They
    are not: the generic ballot is off on its own, so the fix belongs where the
    generic ballot is used as a stand-in for the national vote.
    """
    df = C.load_history()
    lo, hi = C.ELECTION_DAY_WINDOW

    def cycle_mean_error(race_type: str) -> np.ndarray:
        sub = df[df["type_simple"].astype(str) == race_type].copy()
        sub["error"] = sub["margin_poll"] - sub["margin_actual"]
        sub = sub.dropna(subset=["error", "time_to_election", "cycle"])
        sub = sub[
            (sub["time_to_election"] >= lo)
            & (sub["time_to_election"] <= hi)
            & (sub["cycle"] >= 2010)
        ]
        return sub.groupby("cycle")["error"].mean().to_numpy()

    generic = cycle_mean_error("House-G-US")
    assert generic.mean() > 1.0, "the generic ballot's lean has disappeared"

    for race_type in ("House-G", "Sen-G", "Gov-G"):
        per_cycle = cycle_mean_error(race_type)
        se = per_cycle.std(ddof=1) / np.sqrt(len(per_cycle))
        assert abs(per_cycle.mean()) < 2 * se, (
            f"{race_type} polls now look biased too ({per_cycle.mean():+.2f} "
            f"against se {se:.2f}) — the correction may belong somewhere else"
        )


def test_the_bias_is_fitted_on_the_modern_regime_only():
    """Fitting all thirteen cycles would import a regime that ended.

    1998-2006 averaged +4.36 and 2008-2022 +1.29, a decline significant at
    p = 0.011. The default cutoff is the 2010 every other calibration here uses.
    """
    modern = C.estimate_generic_ballot_bias(min_cycle=2010)
    everything = C.estimate_generic_ballot_bias(min_cycle=1998)

    assert modern.min_cycle == 2010
    assert modern.n_cycles == 7
    assert everything.mean_pts > modern.mean_pts + 0.5, (
        "the full-period estimate is no longer clearly larger; re-check whether "
        "the regime split still holds before trusting either"
    )


def test_the_estimate_carries_its_own_uncertainty():
    """Seven cycles is thin, and the config must not present it as exact."""
    bias = C.estimate_generic_ballot_bias()
    assert bias.se_pts > 0.5, "a seven-cycle estimate should not look precise"
    # Thin enough that it does not clear a conventional bar — which is a reason
    # to carry the uncertainty, not a reason to drop the estimate.
    assert bias.mean_pts / bias.se_pts < 2.0


# --- how it is applied ------------------------------------------------------


def test_the_correction_is_house_only():
    """Measured against the national House vote, so applied only there.

    Extending it to the Senate would assert a number nothing measured. There is
    no national Senate vote to have measured it against: the map changes every
    cycle.
    """
    assert ModelConfig.load(chamber="senate").national_environment.generic_ballot_bias == 0.0
    assert ModelConfig.load(chamber="house").national_environment.generic_ballot_bias > 0.0


def test_the_house_config_matches_what_the_estimator_reports():
    """The committed number must be reproducible from the command that made it."""
    fitted = C.estimate_generic_ballot_bias()
    cfg = ModelConfig.load(chamber="house").national_environment
    assert cfg.generic_ballot_bias == pytest.approx(fitted.as_logit, abs=0.002)
    assert cfg.generic_ballot_bias_se == pytest.approx(fitted.se_logit, abs=0.002)


def test_the_correction_points_the_right_way():
    """Positive bias means the ballot overstates Democrats, so it is subtracted.

    A sign error here would be invisible: the forecast would still run, still
    converge, and be wrong by twice the correction.
    """
    bias = C.estimate_generic_ballot_bias()
    assert bias.mean_pts > 0, "history says the ballot overstates Democrats"

    cfg = ModelConfig.load(chamber="house")
    assert cfg.national_environment.generic_ballot_bias > 0

    # And the model subtracts it: theta = alpha + lambda*(eta - bias) + eps.
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "midterms" / "model" / "hierarchical.py"
    ).read_text(encoding="utf-8")
    assert "eta - bias" in source, (
        "the model no longer subtracts the bias from eta; check the sign"
    )


def test_the_uncertainty_widens_the_election_day_error():
    """Subtracting an estimate and then quoting an exact interval overclaims."""
    from midterms.model import simulate

    source = (
        __import__("pathlib").Path(simulate.__file__)
    ).read_text(encoding="utf-8")
    assert "generic_ballot_bias_se" in source, (
        "simulation no longer folds the estimate's standard error into the "
        "national error, so the interval understates what is actually known"
    )
