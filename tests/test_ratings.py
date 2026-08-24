"""Pollster quality weighting.

The property that matters most here is the centring. ``excess_sd_prior`` was
fitted against all historical polls, so it already describes an average
pollster; 538's plus-minus averages +0.49 points rather than zero. Applying the
ratings uncentred would inflate every poll's noise and silently undo the
calibration that sets the width of the seat distribution -- with nothing in the
output looking wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from midterms.config import ModelConfig
from midterms.data.ratings import (
    MAX_MULTIPLIER,
    MIN_MULTIPLIER,
    PollsterRatings,
)


@pytest.fixture(scope="module")
def ratings():
    return PollsterRatings.load()


def test_the_vendored_file_parses(ratings):
    assert len(ratings.by_key) > 400
    assert 3.0 < ratings.mean_expected_error < 9.0


def test_a_marquee_pollster_is_found_and_rated_well(ratings):
    nyt = ratings.find("The New York Times/Siena College")
    assert nyt is not None
    assert nyt.grade.startswith("A")
    assert nyt.predictive_plus_minus < 0, "A+ pollster should beat expectation"


def test_a_sponsor_prefixed_name_falls_back_to_the_pollster(ratings):
    """Our feed says "CNN/SSRS"; 538 rates "SSRS". The track record is SSRS's."""
    assert ratings.find("CNN/SSRS") is not None
    assert ratings.find("CNN/SSRS").pollster == "SSRS"


def test_the_full_name_wins_over_its_components(ratings):
    """538 rates some partnerships in their own right; those must not be split."""
    found = ratings.find("The New York Times/Siena College")
    assert found.pollster == "The New York Times/Siena College"


def test_an_unrated_pollster_returns_nothing(ratings):
    """538 stopped updating in 2025, so 2026 entrants cannot be rated."""
    assert ratings.find("Quantus Insights") is None
    assert ratings.raw_multiplier("Quantus Insights") is None


def test_a_worse_pollster_gets_a_larger_multiplier(ratings):
    better = ratings.raw_multiplier("The New York Times/Siena College")
    worse = ratings.raw_multiplier("McLaughlin & Associates")
    assert better is not None and worse is not None
    assert better < 1.0 < worse, (better, worse)


def test_multipliers_are_clamped(ratings):
    values = [
        m for name in ratings.by_key.values()
        if (m := ratings.raw_multiplier(name.pollster)) is not None
    ]
    assert values
    assert min(values) >= MIN_MULTIPLIER - 1e-9
    assert max(values) <= MAX_MULTIPLIER + 1e-9


# --------------------------------------------------------------- centring


def test_multipliers_are_centred_on_the_rated_polls(ratings):
    """The whole point: redistribute trust without changing its total."""
    field = (
        ["The New York Times/Siena College"] * 12
        + ["McLaughlin & Associates"] * 18
        + ["Emerson College"] * 41
    )
    multipliers = ratings.noise_multipliers(field)
    mean = np.mean([multipliers[n] for n in field])
    assert mean == pytest.approx(1.0, abs=1e-9)


def test_centring_is_weighted_by_poll_count_not_pollster_count(ratings):
    """A pollster with forty polls must count forty times in the centring.

    Averaging over distinct pollsters instead would let a single-poll outfit
    pull the centre as hard as one that dominates the field.
    """
    heavy = ["Emerson College"] * 100 + ["McLaughlin & Associates"] * 1
    multipliers = ratings.noise_multipliers(heavy)
    assert np.mean([multipliers[n] for n in heavy]) == pytest.approx(1.0, abs=1e-9)
    # Emerson dominates, so its multiplier must sit very close to 1.
    assert multipliers["Emerson College"] == pytest.approx(1.0, abs=0.02)


def test_unrated_pollsters_are_treated_as_exactly_average(ratings):
    field = ["Quantus Insights"] * 5 + ["Emerson College"] * 5
    multipliers = ratings.noise_multipliers(field)
    assert multipliers["Quantus Insights"] == 1.0


def test_an_all_unrated_field_falls_back_to_uniform(ratings):
    multipliers = ratings.noise_multipliers(["Quantus Insights", "Verasight"])
    assert set(multipliers.values()) == {1.0}


def test_noise_multipliers_accepts_repeated_names_without_double_counting(ratings):
    once = ratings.noise_multipliers(["Emerson College", "McLaughlin & Associates"])
    twice = ratings.noise_multipliers(
        ["Emerson College"] * 2 + ["McLaughlin & Associates"] * 2
    )
    assert once == pytest.approx(twice)


# ------------------------------------------------------------------ bans


def test_banned_pollsters_are_identified(ratings):
    assert ratings.banned_names(["Big Data Poll", "Emerson College"]) == {
        "Big Data Poll"
    }


def test_banned_pollsters_are_dropped_from_the_table():
    """Fabricated data cannot be down-weighted into usefulness.

    A plausible invented number looks precise rather than noisy, so a variance
    adjustment would make the model more confident, not less.
    """
    pytest.importorskip("midterms.data.votehub")
    from midterms.config import load_all
    from midterms.data.polls import build_poll_table
    from midterms.data.roster import Roster
    from midterms.data.votehub import latest_snapshot, load_snapshot

    snapshot = latest_snapshot()
    if snapshot is None:
        pytest.skip("no local poll snapshot")

    races, cfg = load_all()
    assert cfg.polls.pollster_ratings.exclude_banned, "config changed; update this test"

    raw = load_snapshot(snapshot)
    table = build_poll_table(raw, races, cfg, Roster.load())
    ratings = PollsterRatings.load()
    surviving = {p.pollster for p in table.polls}
    assert not ratings.banned_names(surviving)
    assert any("banned by 538" in reason for reason in table.rejections), (
        "the drop must be counted and reported, not silent"
    )


# ------------------------------------------------------- the model's view


def test_the_design_matrix_carries_one_multiplier_per_poll():
    pytest.importorskip("midterms.data.votehub")
    from midterms import fundamentals as F
    from midterms.config import load_all
    from midterms.data import Roster, build_poll_table
    from midterms.data.votehub import latest_snapshot, load_snapshot
    from midterms.model.design import build_model_data

    snapshot = latest_snapshot()
    if snapshot is None:
        pytest.skip("no local poll snapshot")

    races, cfg = load_all()
    table = build_poll_table(load_snapshot(snapshot), races, cfg, Roster.load())
    data = build_model_data(table, races, cfg, F.compute(races, cfg))

    assert len(data.quality_multiplier) == len(data.y)
    assert np.all(data.quality_multiplier > 0)
    assert np.mean(data.quality_multiplier) == pytest.approx(1.0, abs=0.05)


def test_disabling_the_feature_gives_a_neutral_multiplier():
    """The switch must produce all ones, so the likelihood needs no branch."""
    pytest.importorskip("midterms.data.votehub")
    import dataclasses

    from midterms import fundamentals as F
    from midterms.config import load_all
    from midterms.data import Roster, build_poll_table
    from midterms.data.votehub import latest_snapshot, load_snapshot
    from midterms.model.design import build_model_data

    snapshot = latest_snapshot()
    if snapshot is None:
        pytest.skip("no local poll snapshot")

    races, cfg = load_all()
    off = dataclasses.replace(
        cfg,
        polls=dataclasses.replace(
            cfg.polls,
            pollster_ratings=dataclasses.replace(
                cfg.polls.pollster_ratings, enabled=False
            ),
        ),
    )
    table = build_poll_table(load_snapshot(snapshot), races, off, Roster.load())
    data = build_model_data(table, races, off, F.compute(races, off))
    assert np.all(data.quality_multiplier == 1.0)


def test_config_and_module_agree_on_the_feature_being_live():
    cfg = ModelConfig.load()
    assert cfg.polls.pollster_ratings.enabled, (
        "ratings disabled in config; the tests above assert the enabled behaviour"
    )
