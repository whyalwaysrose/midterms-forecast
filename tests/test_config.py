"""Config integrity.

The seat-arithmetic tests are the most valuable in the suite. A miscounted
chamber baseline is the most damaging silent error a seat model can have: it
shifts every control probability without changing any individual race, so
nothing looks wrong on the dashboard.
"""

from __future__ import annotations

import pytest

from midterms.config import ConfigError, Control, ModelConfig, Race, RaceSet


def test_race_set_loads():
    races = RaceSet.load()
    assert races.chamber == "senate"
    assert races.cycle == 2026
    assert len(races.races) == 35


def test_seat_arithmetic_closes_to_one_hundred():
    races = RaceSet.load()
    control = races.control
    total = sum(control.seats_not_up.values()) + sum(control.seats_up.values())
    assert total == control.total_seats == 100


def test_seats_up_matches_race_incumbent_parties():
    races = RaceSet.load()
    for party in ("D", "R"):
        actual = sum(1 for r in races.races if r.incumbent_party == party)
        assert races.control.seats_up[party] == actual


def test_majority_thresholds_are_consistent_with_the_tiebreaker():
    """At 50-50 the tiebreaker party holds the chamber, so the two thresholds
    must not both be satisfiable."""
    races = RaceSet.load()
    c = races.control
    assert c.dem_seats_for_majority + c.rep_seats_for_majority == c.total_seats + 1
    assert c.tiebreaker_party in ("D", "R")


def test_race_ids_are_unique():
    races = RaceSet.load()
    ids = [r.id for r in races.races]
    assert len(set(ids)) == len(ids)


def test_every_race_has_plausible_presidential_shares():
    races = RaceSet.load()
    for race in races.races:
        assert 0.15 < race.pres_2024_dem_two_party < 0.85, race.id
        assert 0.15 < race.pres_2020_dem_two_party < 0.85, race.id


def test_votehub_subject_matches_feed_naming():
    races = RaceSet.load()
    georgia = races.by_id("senate-2026-GA")
    assert georgia.subject_for(2026) == "2026 Georgia"


def test_model_config_loads():
    cfg = ModelConfig.load()
    assert cfg.sampler.chains >= 2
    assert cfg.polls.population_reference == "lv"
    assert cfg.election_day_error.state_sd > 0


def test_fundamentals_weights_must_sum_to_one():
    from midterms.config import FundamentalsConfig

    with pytest.raises(ConfigError, match="must sum to 1"):
        FundamentalsConfig(
            weight_pres_2024=0.9,
            weight_pres_2020=0.5,
            lean_shrinkage=1.0,
            incumbency_bonus={"elected": 0.0, "appointed": 0.0, "open": 0.0},
            prior_sd=0.1,
        )


def _race(**overrides):
    base = dict(
        id="x",
        unit="XX",
        name="Example",
        special=False,
        incumbent_party="D",
        incumbent_status="elected",
        pres_2024_dem_two_party=0.5,
        pres_2020_dem_two_party=0.5,
        region="south",
    )
    base.update(overrides)
    return Race(**base)


def test_bad_party_is_rejected():
    with pytest.raises(ConfigError, match="incumbent_party"):
        Race.from_dict(
            {
                "id": "x", "unit": "XX", "name": "E", "special": False,
                "incumbent_party": "Q", "incumbent_status": "open",
                "pres_2024_dem_two_party": 0.5, "pres_2020_dem_two_party": 0.5,
                "region": "south",
            }
        )


def test_control_validation_catches_a_miscount():
    control = Control(
        total_seats=100,
        seats_not_up={"D": 34, "R": 31},
        seats_up={"D": 1, "R": 1},
        dem_seats_for_majority=51,
        rep_seats_for_majority=50,
        tiebreaker_party="R",
    )
    # 65 + 2 != 100
    with pytest.raises(ConfigError, match="does not close"):
        control.validate((_race(id="a"), _race(id="b", incumbent_party="R")))


def test_duplicate_race_ids_are_rejected():
    from datetime import date

    control = Control(
        total_seats=4,
        seats_not_up={"D": 1, "R": 1},
        seats_up={"D": 2, "R": 0},
        dem_seats_for_majority=3,
        rep_seats_for_majority=2,
        tiebreaker_party="R",
    )
    with pytest.raises(ConfigError, match="duplicate race ids"):
        RaceSet(
            cycle=2026,
            chamber="senate",
            election_date=date(2026, 11, 3),
            control=control,
            races=(_race(id="dup"), _race(id="dup")),
        )
