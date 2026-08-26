"""Each chamber's error scales must come from that chamber's own polling.

Applying Senate numbers to 435 districts was never a decision anyone took -- it
was what happened when a second chamber was added to a model that had only ever
had one. These pin the pieces that make it a decision.

The failure modes here are all silent. A substring race-type filter quietly
mixes primaries into general-election error. A config override that fails to
apply leaves the House on Senate scales while the file says otherwise. A
whole-block override forks the correlation kernel the moment someone edits it in
one place. None of these raise; they just change the width of a seat
distribution nobody can check until election night.
"""

from __future__ import annotations

import numpy as np
import pytest

from midterms import calibration as C
from midterms.config import ModelConfig

# --- race-type selection ----------------------------------------------------


def test_race_type_matching_is_exact_not_substring():
    """`contains("Sen")` also matches `Sen-P`; `contains("House")` matches the
    national House vote, which is one row per cycle rather than one per race."""
    df = C.load_history()
    for chamber, expected in C.RACE_TYPE.items():
        got = set(C.race_polls(df, chamber, (0, 400), 1990)["type_simple"].astype(str))
        assert got <= {expected}, f"{chamber} swept in {got - {expected}}"


def test_primaries_are_excluded_from_both_chambers():
    df = C.load_history()
    for chamber in C.RACE_TYPE:
        types = set(C.race_polls(df, chamber, (0, 400), 1990)["type_simple"].astype(str))
        assert not any(t.endswith("-P") for t in types), f"{chamber} includes primaries"


def test_the_national_house_vote_never_enters_the_per_race_pool():
    """`House-G-US` is the national result, not a district.

    One aggregate row per cycle among 40 district rows would distort both the
    cycle mean and the residual spread the split is built from.
    """
    df = C.load_history()
    house = C.race_polls(df, "house", (0, 400), 1990)
    assert "House-G-US" not in set(house["type_simple"].astype(str))
    assert len(house) > 0


def test_an_unknown_chamber_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown chamber"):
        C.race_polls(C.load_history(), "governors")


def test_senate_polls_still_works_for_existing_callers():
    df = C.load_history()
    assert len(C.senate_polls(df)) == len(C.race_polls(df, "senate"))


# --- the two windows --------------------------------------------------------


def test_the_design_effect_is_fitted_at_the_horizon_whatever_is_asked():
    """It describes how the model weighs polls, so it belongs at the horizon.

    The election-day scales must exclude drift and so are fitted at 0-14 days.
    Fitting both in one window gets one of them wrong -- and the two differ
    enough to matter: Senate 1.18 against 1.36, House 1.04 against 0.95.
    """
    eve = C.estimate_error_components(C.ELECTION_DAY_WINDOW, chamber="senate")
    horizon = C.estimate_error_components(C.HORIZON_WINDOW, chamber="senate")
    assert eve.design_effect == pytest.approx(horizon.design_effect, abs=1e-9)
    assert eve.design_effect_window == C.HORIZON_WINDOW
    # ...and the scales genuinely do depend on the window, or the note is moot.
    assert eve.state_sd < horizon.state_sd


def test_fitting_at_the_horizon_folds_in_drift():
    """The reason the election-day scales are not fitted where the model runs.

    Drift lives in the random walk. Fitting the terminal miss at 45-120 days
    would count it twice and keep the interval just as wide on election eve.
    """
    for chamber in ("senate", "house"):
        eve = C.estimate_error_components(C.ELECTION_DAY_WINDOW, chamber=chamber)
        horizon = C.estimate_error_components(C.HORIZON_WINDOW, chamber=chamber)
        assert horizon.state_sd > eve.state_sd * 1.1, (
            f"{chamber}: horizon fit should be visibly inflated by drift"
        )


# --- the config override ----------------------------------------------------


def test_the_house_config_matches_the_house_fit():
    """The committed numbers must be reproducible from the command that made them."""
    fitted = C.estimate_error_components(C.ELECTION_DAY_WINDOW, chamber="house")
    cfg = ModelConfig.load(chamber="house")
    assert cfg.election_day_error.national_sd * C.POINTS_PER_LOGIT == pytest.approx(
        fitted.national_sd, abs=0.05
    )
    assert cfg.election_day_error.state_sd * C.POINTS_PER_LOGIT == pytest.approx(
        fitted.state_sd, abs=0.05
    )
    assert cfg.polls.design_effect == pytest.approx(fitted.design_effect, abs=0.02)


def test_the_senate_config_matches_the_senate_fit():
    fitted = C.estimate_error_components(C.ELECTION_DAY_WINDOW, chamber="senate")
    cfg = ModelConfig.load(chamber="senate")
    for name, value in (
        ("national_sd", fitted.national_sd),
        ("state_sd", fitted.state_sd),
    ):
        assert getattr(cfg.election_day_error, name) * C.POINTS_PER_LOGIT == pytest.approx(
            value, abs=0.10
        ), name


def test_the_chambers_actually_differ():
    """If the override silently failed, everything above would still pass."""
    senate = ModelConfig.load(chamber="senate")
    house = ModelConfig.load(chamber="house")
    assert house.election_day_error.national_sd > senate.election_day_error.national_sd
    assert house.election_day_error.state_sd > senate.election_day_error.state_sd
    assert house.polls.design_effect < senate.polls.design_effect


def test_the_override_is_leafwise_so_the_kernel_stays_shared():
    """A whole-block override would fork the correlation kernel.

    It would work today and diverge the first time someone tuned the kernel and
    edited only the top-level copy -- with the House silently left behind.
    """
    senate = ModelConfig.load(chamber="senate").election_day_error.correlation
    house = ModelConfig.load(chamber="house").election_day_error.correlation
    assert house == senate


def test_an_unlisted_chamber_falls_back_to_the_base_config():
    base = ModelConfig.load(chamber="senate")
    same = ModelConfig.load(chamber="nonexistent")
    assert same.election_day_error == base.election_day_error


def test_house_scales_widen_rather_than_narrow():
    """Recorded deliberately, because the intuition runs the other way.

    The House interval looked too wide, so the expectation was that calibrating
    it would pull it in. Measured on House polling it does the opposite, and a
    future change that quietly reverses this should have to argue for it.
    """
    senate = ModelConfig.load(chamber="senate").election_day_error
    house = ModelConfig.load(chamber="house").election_day_error
    s_total = float(np.hypot(senate.national_sd, senate.state_sd))
    h_total = float(np.hypot(house.national_sd, house.state_sd))
    assert h_total > s_total


# --- the fingerprint --------------------------------------------------------


def test_a_house_only_config_change_does_not_flag_the_senate():
    """Otherwise the Senate suppresses attribution for a change it never saw.

    The fingerprint exists so that day-over-day commentary refuses to blame the
    polls for a model revision. It hashed the whole model.yaml, so adding the
    House override block would have made the very next Senate run announce "the
    model changed in this run" and decline to attribute real poll movement --
    the exact false alarm the fingerprint is there to prevent.
    """
    from midterms.outputs import model_fingerprint

    senate = ModelConfig.load(chamber="senate")
    assert model_fingerprint(senate) == model_fingerprint(), (
        "the Senate's fingerprint must not depend on another chamber's overrides"
    )


def test_the_two_chambers_have_different_fingerprints():
    """They ran on different scales, so a run of one is not a run of the other."""
    from midterms.outputs import model_fingerprint

    assert model_fingerprint(ModelConfig.load(chamber="senate")) != model_fingerprint(
        ModelConfig.load(chamber="house")
    )


def test_changing_a_house_scale_moves_only_the_house_fingerprint(tmp_path):
    import yaml

    from midterms import paths
    from midterms.outputs import model_fingerprint

    raw = yaml.safe_load(paths.MODEL_CONFIG.read_text(encoding="utf-8"))
    raw["chambers"]["house"]["election_day_error"]["national_sd"] = 0.099
    edited = tmp_path / "model.yaml"
    edited.write_text(yaml.safe_dump(raw), encoding="utf-8")

    before_h = model_fingerprint(ModelConfig.load(chamber="house"))
    before_s = model_fingerprint(ModelConfig.load(chamber="senate"))
    after_h = model_fingerprint(ModelConfig.load(edited, chamber="house"))
    after_s = model_fingerprint(ModelConfig.load(edited, chamber="senate"))

    assert after_h != before_h, "the House should notice its own scale changing"
    assert after_s == before_s, "the Senate should not"
