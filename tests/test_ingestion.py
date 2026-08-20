"""Poll normalisation, roster resolution, and design-matrix assembly.

These run against synthetic records rather than the live API so they are fast,
deterministic, and still exercise the paths that matter — especially the ones
that decide whether a poll is used at all.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from midterms.config import ModelConfig, RaceSet
from midterms.data.polls import NATIONAL_RACE_ID, build_poll_table
from midterms.data.roster import Roster
from midterms.data.votehub import is_primary_subject
from midterms.model.design import (
    POOLED_POLLSTER,
    build_model_data,
    build_time_grid,
)


@pytest.fixture(scope="module")
def setup():
    return RaceSet.load(), ModelConfig.load(), Roster.load()


def poll(**kw):
    """A VoteHub-shaped record with sensible defaults."""
    base = {
        "id": "p1",
        "poll_type": "us-senator",
        "subject": "2026 Georgia",
        "pollster": "Test Pollster",
        "sample_size": 800,
        "population": "lv",
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
        "answers": [{"choice": "Jon Ossoff", "pct": 50.0}, {"choice": "Mike Collins", "pct": 46.0}],
        "sponsors": [],
        "partisan": None,
        "internal": False,
        "url": "https://example.test",
    }
    base.update(kw)
    return base


AS_OF = date(2026, 8, 20)


# ------------------------------------------------------------------- roster


def test_roster_resolves_known_candidates(setup):
    _races, _cfg, roster = setup
    assert roster.resolve("senate-2026-GA", "Jon Ossoff") == "D"
    assert roster.resolve("senate-2026-GA", "Mike Collins") == "R"


def test_roster_is_case_and_whitespace_insensitive(setup):
    _races, _cfg, roster = setup
    assert roster.resolve("senate-2026-GA", "  jon   ossoff ") == "D"


def test_roster_normalises_known_feed_misspellings(setup):
    _races, _cfg, roster = setup
    # These typos really do appear in the VoteHub feed.
    assert roster.resolve("senate-2026-AK", "Dan Sullvian") == "R"
    assert roster.resolve("senate-2026-NC", "Thom Thillis") == "R"


def test_roster_maps_generic_placeholders(setup):
    _races, _cfg, roster = setup
    assert roster.resolve("senate-2026-ME", "Dem") == "D"
    assert roster.resolve("senate-2026-MN", "Rep") == "R"


def test_roster_fails_closed_on_unknown_names(setup):
    _races, _cfg, roster = setup
    assert roster.resolve("senate-2026-GA", "Someone Entirely New") == "other"


def test_unknown_names_reports_only_genuinely_unclassified(setup):
    _races, _cfg, roster = setup
    unknown = roster.unknown_names(
        "senate-2026-ID", ["Jim Risch", "Matt Loesby", "Brand New Person"]
    )
    # Loesby is explicitly listed under `other`, so it is classified.
    assert unknown == ["Brand New Person"]


def test_independent_is_mapped_and_flagged(setup):
    _races, _cfg, roster = setup
    assert roster.resolve("senate-2026-NE", "Dan Osborn") == "D"
    assert "senate-2026-NE" in roster.independent_notes


# -------------------------------------------------------------- subjects


def test_primary_subjects_are_detected():
    assert is_primary_subject("2026 Texas Democratic")
    assert is_primary_subject("2026 Illinois Republican")
    assert not is_primary_subject("2026 Texas")


# ------------------------------------------------------------ normalisation


def test_two_party_share_excludes_third_party(setup):
    races, cfg, roster = setup
    record = poll(
        answers=[
            {"choice": "Jon Ossoff", "pct": 48.0},
            {"choice": "Mike Collins", "pct": 32.0},
            {"choice": "Someone Else", "pct": 20.0},
        ]
    )
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert len(table.race_polls) == 1
    got = table.race_polls[0]
    assert got.two_party_dem == pytest.approx(48 / 80)
    assert got.other_pct == pytest.approx(20.0)


def test_field_date_is_the_midpoint_of_the_field_period(setup):
    races, cfg, roster = setup
    record = poll(start_date="2026-08-01", end_date="2026-08-05")
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert table.race_polls[0].field_date == date(2026, 8, 3)


def test_partisan_sponsor_sign(setup):
    races, cfg, roster = setup
    table = build_poll_table(
        {"us-senator": [
            poll(id="d", partisan="DEM"),
            poll(id="r", partisan="REP"),
            poll(id="n", partisan=None),
        ]},
        races, cfg, roster, as_of=AS_OF,
    )
    signs = {p.poll_id: p.partisan_sign for p in table.race_polls}
    assert signs == {"d": 1, "r": -1, "n": 0}


def test_poll_with_two_candidates_on_one_side_is_rejected(setup):
    """A multi-way hypothetical cannot be reduced to one two-party number."""
    races, cfg, roster = setup
    record = poll(
        subject="2026 Maine",
        answers=[
            {"choice": "Susan Collins", "pct": 42.0},
            {"choice": "Graham Platner", "pct": 30.0},
            {"choice": "Janet Mills", "pct": 22.0},
        ],
    )
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert len(table.race_polls) == 0
    assert table.rejections["no single D-vs-R matchup"] == 1


def test_primary_polls_are_dropped(setup):
    races, cfg, roster = setup
    record = poll(subject="2026 Texas Democratic")
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert len(table.race_polls) == 0
    assert table.rejections["primary subject"] == 1


def test_polls_after_as_of_are_excluded(setup):
    """This is what makes a backtest honest."""
    races, cfg, roster = setup
    record = poll(start_date="2026-08-18", end_date="2026-08-19")
    table = build_poll_table(
        {"us-senator": [record]}, races, cfg, roster, as_of=date(2026, 8, 1)
    )
    assert len(table.race_polls) == 0
    assert table.rejections["fielded after as_of"] == 1


def test_stale_polls_are_excluded(setup):
    races, cfg, roster = setup
    old = AS_OF - timedelta(days=cfg.polls.max_age_days + 30)
    record = poll(start_date=old.isoformat(), end_date=old.isoformat())
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert table.rejections["older than max_age_days"] == 1


def test_small_samples_are_excluded(setup):
    races, cfg, roster = setup
    record = poll(sample_size=cfg.polls.min_sample_size - 1)
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert table.rejections["sample too small"] == 1


def test_missing_sample_size_falls_back_to_the_default(setup):
    races, cfg, roster = setup
    record = poll(sample_size=None)
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert table.race_polls[0].sample_size == cfg.polls.default_sample_size


def test_unmapped_subject_is_recorded(setup):
    races, cfg, roster = setup
    record = poll(subject="2026 Atlantis")
    table = build_poll_table({"us-senator": [record]}, races, cfg, roster, as_of=AS_OF)
    assert any("unmapped subject" in reason for reason in table.rejections)


def test_generic_ballot_polls_land_on_the_national_race(setup):
    races, cfg, roster = setup
    record = poll(
        poll_type="generic-ballot", subject="2026",
        answers=[{"choice": "Dem", "pct": 48.0}, {"choice": "Rep", "pct": 45.0}],
    )
    table = build_poll_table({"generic-ballot": [record]}, races, cfg, roster, as_of=AS_OF)
    assert len(table.national) == 1
    assert table.national[0].race_id == NATIONAL_RACE_ID
    assert table.national[0].two_party_dem == pytest.approx(48 / 93)


# ------------------------------------------------------------------ design


def test_time_grid_ends_exactly_on_election_day():
    grid = build_time_grid(date(2025, 7, 1), date(2026, 11, 3), 7)
    assert grid[-1] == date(2026, 11, 3)
    assert grid[0] <= date(2025, 7, 1)
    # Uniform spacing.
    steps = {(grid[i + 1] - grid[i]).days for i in range(len(grid) - 1)}
    assert steps == {7}


def test_infrequent_pollsters_are_pooled(setup):
    races, cfg, roster = setup
    from midterms import fundamentals as F

    records = []
    # One frequent pollster (meets the threshold) and several one-off ones.
    for i in range(cfg.polls.min_polls_for_house_effect):
        records.append(poll(id=f"freq{i}", pollster="Frequent Pollster",
                            start_date=f"2026-07-0{i + 1}",
                            end_date=f"2026-07-0{i + 2}"))
    for i in range(3):
        records.append(poll(id=f"rare{i}", pollster=f"Rare Pollster {i}"))

    table = build_poll_table({"us-senator": records}, races, cfg, roster, as_of=AS_OF)
    fund = F.compute(races, cfg)
    data = build_model_data(table, races, cfg, fund)

    assert "Frequent Pollster" in data.pollster_names
    assert POOLED_POLLSTER in data.pollster_names
    assert "Rare Pollster 0" not in data.pollster_names
    # All three rare pollsters share the one pooled index.
    pooled_index = data.pollster_names.index(POOLED_POLLSTER)
    rare_positions = [i for i, p in enumerate(table.polls) if p.pollster.startswith("Rare")]
    assert all(data.pollster_idx[i] == pooled_index for i in rare_positions)


def test_design_arrays_are_aligned(setup):
    races, cfg, roster = setup
    from midterms import fundamentals as F

    records = [poll(id=f"p{i}", start_date="2026-07-01", end_date="2026-07-03")
               for i in range(4)]
    generic = [poll(id="g1", poll_type="generic-ballot", subject="2026",
                    answers=[{"choice": "Dem", "pct": 48.0}, {"choice": "Rep", "pct": 45.0}])]

    table = build_poll_table(
        {"us-senator": records, "generic-ballot": generic}, races, cfg, roster, as_of=AS_OF
    )
    fund = F.compute(races, cfg)
    data = build_model_data(table, races, cfg, fund)

    n = len(data.y)
    assert n == data.n_race_polls + data.n_national_polls
    for arr in (data.sampling_var, data.pollster_idx, data.population_idx, data.partisan_sign):
        assert len(arr) == n
    assert data.race_poll_race_idx.max() < data.n_races
    assert data.race_poll_time_idx.max() < data.n_grid
    assert (data.sampling_var > 0).all()
    assert np.isfinite(data.y).all()


def test_effective_sample_size_shrinks_with_undecideds(setup):
    from midterms.data.polls import NormalisedPoll
    from midterms.model.design import _effective_sample_size

    def make(dem, rep):
        return NormalisedPoll(
            poll_id="x", race_id="r", pollster="p",
            field_date=AS_OF, start_date=AS_OF, end_date=AS_OF,
            sample_size=1000, population="lv",
            dem_pct=dem, rep_pct=rep,
            two_party_dem=dem / (dem + rep), other_pct=100 - dem - rep,
            partisan_sign=0, sponsors=(), url="",
            dem_candidate="D", rep_candidate="R",
        )

    full = _effective_sample_size(make(50.0, 50.0))
    partial = _effective_sample_size(make(42.0, 40.0))
    assert full == pytest.approx(1000.0)
    assert partial == pytest.approx(820.0)
    assert partial < full
