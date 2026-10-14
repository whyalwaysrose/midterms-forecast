"""The forward calibration has to ask for its own chamber's data.

For three weeks it did not. `run_forward` built its poll table like this:

    build_poll_table(raw, races, cfg, Roster.load(), as_of=today)

and both omissions default to the Senate: `Roster.load()` reads
`candidates_senate_2026.yaml`, and `build_poll_table` reads `us-senator`. So the
House run scored Senate polls against Senate candidates, matched none of them to
a House subject, and recorded only the generic-ballot polls -- which need no
roster and no chamber. The accumulated House table held four rows, every one of
them `__national__`, while 35 district polls arrived and every daily workflow
reported success.

Fixing it took the House from 4 polls across 1 race to 45 across 36. These tests
are about the defaults that made it possible, not the arithmetic: the bug was
invisible precisely because nothing failed, so what has to be asserted is that
the chamber reaches the data layer at all.
"""

from __future__ import annotations

from datetime import date

import pytest

from midterms import forward
from midterms.data.polls import RACE_POLL_TYPE
from midterms.data.roster import ROSTER_FILE, Roster

# --- the roster accessor ----------------------------------------------------

def test_for_chamber_loads_a_different_roster_per_chamber():
    senate = Roster.for_chamber("senate")
    house = Roster.for_chamber("house")
    assert set(senate.by_race) != set(house.by_race)
    assert any(r.startswith("senate-2026-") for r in senate.by_race)
    assert any(r.startswith("house-2026-") for r in house.by_race)


def test_for_chamber_refuses_an_unknown_chamber():
    with pytest.raises(ValueError, match="unknown chamber"):
        Roster.for_chamber("assembly")


def test_the_senate_roster_cannot_classify_a_house_district_poll():
    """Why the wrong default produced silence instead of an error.

    The Senate roster has no entry for a House district, so a district poll's
    candidates are unknown, the poll is rejected as "no single D-vs-R matchup",
    and that is a perfectly ordinary rejection -- it is also what a genuine
    primary field looks like. Nothing distinguishes the two from outside.
    """
    senate = Roster.for_chamber("senate")
    house = Roster.for_chamber("house")
    district = next(r for r in house.by_race if house.by_race[r])
    names = list(house.by_race[district])

    assert senate.unknown_names(district, names) == tuple(names) or set(
        senate.unknown_names(district, names)
    ) == set(names)
    assert not house.unknown_names(district, names)


def test_every_chamber_has_a_roster_file():
    from midterms import paths

    assert set(ROSTER_FILE) == set(RACE_POLL_TYPE)
    for chamber, name in ROSTER_FILE.items():
        assert (paths.CONFIG_DIR / name).exists(), chamber


# --- the wiring that was wrong ----------------------------------------------

class _Recorder:
    """Captures what run_forward asks the data layer for."""

    def __init__(self):
        self.roster_chamber = None
        self.poll_type = None
        self.roster_passed = None


@pytest.fixture
def recorded(monkeypatch):
    seen = _Recorder()
    sentinel = object()

    def fake_for_chamber(chamber):
        seen.roster_chamber = chamber
        return sentinel

    def fake_build(raw, races, cfg, roster, **kwargs):
        seen.roster_passed = roster
        seen.poll_type = kwargs.get("race_poll_type")

        class _Table:
            polls = []

        return _Table()

    monkeypatch.setattr(Roster, "for_chamber", staticmethod(fake_for_chamber))
    monkeypatch.setattr(forward, "build_poll_table", fake_build)
    # Everything downstream of the table is irrelevant here.
    monkeypatch.setattr(forward, "score_against", lambda payload, polls: [])
    monkeypatch.setattr(forward, "append", lambda *a, **k: 0)
    monkeypatch.setattr(
        forward, "previous_run", lambda chamber, before: {"run_date": "2026-10-13"}
    )
    monkeypatch.setattr(forward, "latest_snapshot", lambda: "snapshot")
    monkeypatch.setattr(forward, "load_snapshot", lambda path: {})
    return seen, sentinel


@pytest.mark.parametrize("chamber", ["senate", "house"])
def test_run_forward_asks_for_its_own_chamber(chamber, recorded):
    """The regression. Neither argument may be left to its default."""
    seen, sentinel = recorded
    forward.run_forward(chamber=chamber, today=date(2026, 10, 14))

    assert seen.roster_chamber == chamber, "roster was not requested per chamber"
    assert seen.roster_passed is sentinel
    assert seen.poll_type == RACE_POLL_TYPE[chamber], (
        f"poll type was {seen.poll_type!r}, not {RACE_POLL_TYPE[chamber]!r}"
    )


def test_the_house_does_not_silently_get_senate_data(recorded):
    """Stated separately because this is the exact failure that happened."""
    seen, _ = recorded
    forward.run_forward(chamber="house", today=date(2026, 10, 14))
    assert seen.poll_type != "us-senator"
    assert seen.roster_chamber != "senate"


# --- what the recovered table has to look like ------------------------------

def test_the_accumulated_house_table_is_not_all_national():
    """The symptom, asserted on the committed record.

    Four rows all tagged `__national__` is what three weeks of a broken chamber
    argument looks like from the outside. If this file ever goes back to holding
    no district rows, something has defaulted to the Senate again.
    """
    import csv

    from midterms import paths

    path = paths.DATA_DIR / "calibration" / "forward_house.csv"
    if not path.exists():
        pytest.skip("no accumulated House calibration yet")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    district = [r for r in rows if r["race_id"] != forward.NATIONAL_RACE_ID]
    assert district, "every recorded House poll is national; the chamber is wrong"
    assert all(r["race_id"].startswith("house-2026-") for r in district)
