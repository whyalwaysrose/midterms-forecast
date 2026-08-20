"""Fitting error scales from history, and scoring against real outcomes.

These run on the vendored historical file, so they are deterministic and need
no network. They are slower than the rest of the suite (a few seconds) because
they decompress and group 20k rows, which is worth it: these functions set the
width of the seat distribution, and a silent regression here would make the
headline number confidently wrong while every other test still passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from midterms import backtest_history as BH
from midterms import calibration as C


@pytest.fixture(scope="module")
def components():
    return C.estimate_error_components()


@pytest.fixture(scope="module")
def race_table():
    return BH.build_race_table()


# ------------------------------------------------------------------ loading


def test_history_loads_with_the_columns_we_depend_on():
    df = C.load_history()
    assert len(df) > 15000
    for column in ("cycle", "type_simple", "race_id", "margin_poll",
                   "margin_actual", "time_to_election", "samplesize"):
        assert column in df.columns


def test_senate_filter_selects_only_senate_races_in_window():
    sen = C.senate_polls(C.load_history(), days_window=(45, 120), min_cycle=2010)
    assert len(sen) > 300
    assert sen["type_simple"].str.contains("Sen").all()
    assert sen["time_to_election"].between(45, 120).all()
    assert (sen["cycle"] >= 2010).all()


def test_tiny_cycles_are_excluded():
    """Off-year specials with one or two races would otherwise dominate the
    cycle-to-cycle spread that becomes the national error term."""
    sen = C.senate_polls(C.load_history(), min_cycle=1998)
    per_cycle = sen.groupby("cycle")["race_id"].nunique()
    assert (per_cycle >= C.MIN_RACES_PER_CYCLE).all()


# -------------------------------------------------------------- decomposition


def test_error_components_are_plausible(components):
    # Sanity bands, not exact values — the point is to catch a decomposition
    # that has silently broken, not to freeze the numbers.
    assert 1.0 < components.national_sd < 6.0
    assert 3.0 < components.state_sd < 9.0
    assert 2.0 < components.poll_sd < 7.0
    assert 0.8 < components.design_effect < 2.5


def test_state_error_exceeds_national_error(components):
    """Races miss individually more than whole cycles miss together. The
    reverse ordering was what the asserted configuration got wrong."""
    assert components.state_sd > components.national_sd


def test_removing_poll_noise_shrinks_the_state_term():
    """A race's mean poll error still carries poll noise divided by its poll
    count; leaving it in inflates the state term."""
    sen = C.senate_polls(C.load_history())
    grouped = sen.groupby(["cycle", "race_id"])["error"]
    naive_sd = float(grouped.mean().std(ddof=1))
    fitted = C.estimate_error_components()
    assert fitted.state_sd < naive_sd


def test_logit_conversion_round_trips(components):
    as_logit = components.as_logit()
    assert as_logit["national_sd"] == pytest.approx(
        components.national_sd / C.POINTS_PER_LOGIT, abs=1e-4
    )
    assert 0.0 < as_logit["state_sd"] < 0.5


def test_config_matches_the_fitted_scales(components):
    """The committed config must be the fitted one, not a stale hand-set value."""
    from midterms.config import ModelConfig

    cfg = ModelConfig.load()
    for name in ("national_sd", "state_sd"):
        configured = getattr(cfg.election_day_error, name) * C.POINTS_PER_LOGIT
        fitted = getattr(components, name)
        assert configured == pytest.approx(fitted, abs=0.25), (
            f"{name} in config/model.yaml is {configured:.2f} pts but history "
            f"fits {fitted:.2f}; re-run `midterms calibrate`"
        )


# ---------------------------------------------------------------- backtesting


def test_race_table_pairs_predictions_with_outcomes(race_table):
    assert len(race_table) > 100
    assert race_table["n_polls"].min() >= 1
    assert race_table["predicted"].notna().all()
    assert race_table["actual"].notna().all()


def test_poll_average_beats_a_coin_flip(race_table):
    """The point estimate must at least get the winner right most of the time."""
    correct = (race_table["predicted"] > 0) == (race_table["actual"] > 0)
    assert correct.mean() > 0.85


def test_fitted_scales_are_better_calibrated_than_the_asserted_ones(race_table):
    """The whole point of the calibration work.

    Scored on 167 real races: the fitted scales must cover the actual margin
    closer to nominal than the values previously asserted from the literature.
    """
    fitted = BH.score(3.00, 5.70, "fitted", race_table)
    asserted = BH.score(3.75, 4.25, "asserted", race_table)

    for level in (0.8, 0.9):
        fitted_gap = abs(fitted.coverage[level] - level)
        asserted_gap = abs(asserted.coverage[level] - level)
        assert fitted_gap < asserted_gap, (
            f"at the {level:.0%} level the fitted scales are no better: "
            f"{fitted.coverage[level]:.1%} vs {asserted.coverage[level]:.1%}"
        )


def test_win_probabilities_are_reliable(race_table):
    """A 30-50% call should win 30-50% of the time, not 10% or 80%."""
    result = BH.score(3.00, 5.70, "fitted", race_table)
    for _, row in result.reliability.iterrows():
        if row["n"] < 10:
            continue
        assert abs(row["mean_pred"] - row["actual"]) < 0.15, (
            f"bin {row['bin']}: predicted {row['mean_pred']:.2f}, "
            f"actual {row['actual']:.2f}"
        )


def test_scoring_beats_the_naive_favourite_rule(race_table):
    result = BH.score(3.00, 5.70, "fitted", race_table)
    assert result.brier_skill_vs_naive > 0.15
    assert 0.0 < result.brier < 0.2


def test_wider_scales_give_wider_coverage(race_table):
    """Monotonicity — a basic guard that the simulation is wired correctly."""
    narrow = BH.score(2.0, 3.0, "narrow", race_table)
    wide = BH.score(4.0, 8.0, "wide", race_table)
    for level in (0.5, 0.8, 0.9):
        assert wide.coverage[level] > narrow.coverage[level]


def test_national_component_is_shared_across_races():
    """The national term must be one draw per simulation, or chamber outcomes
    would not be correlated and the seat distribution would collapse."""
    import pandas as pd

    races = pd.DataFrame({"predicted": [0.0] * 40, "actual": [0.0] * 40,
                          "cycle": [2022] * 40})
    rng = np.random.default_rng(0)
    simulated = BH._simulate(races, national_sd_pts=10.0, state_sd_pts=0.01,
                             n_sims=4000, rng=rng)
    # With a large shared term and negligible independent term, every race in a
    # simulation should move together.
    correlation = np.corrcoef(simulated.T)
    off_diagonal = correlation[np.triu_indices(40, k=1)]
    assert off_diagonal.min() > 0.95
