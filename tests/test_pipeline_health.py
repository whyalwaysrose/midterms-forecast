"""The pipeline health checks, tested against the failures they are named for.

Every other check in this project tests the model given its inputs. These test
that the inputs arrived, which is where the real defects have been. So each
test here reconstructs an actual past failure rather than inventing one -- a
check nobody can point at a real bug for is a check that gets tuned away the
first time it is inconvenient, and a threshold chosen without the failing case
in hand is a guess.

The reconstructions, with the numbers they actually had:

* the stale House roster: 27 of 65 districts with poll data unusable, 41%;
* `midterms fetch` with no `us-representative` in the snapshot;
* `forward.py` reading Senate polls for the House: zero race polls of the
  chamber's own type;
* a roster rebuild failing quietly for over a week.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import pytest

from midterms import health
from midterms.data.polls import ALL_POLL_TYPES

# --- stand-ins, kept honest -------------------------------------------------

@dataclass
class FakePoll:
    race_id: str


@dataclass
class FakeTable:
    polls: list = field(default_factory=list)
    rejections: dict = field(default_factory=dict)
    unknown_candidates: dict = field(default_factory=dict)

    @property
    def race_polls(self):
        return [p for p in self.polls if p.race_id != "__national__"]

    @property
    def national(self):
        return [p for p in self.polls if p.race_id == "__national__"]


@dataclass
class FakeRace:
    id: str
    pres_2024_dem_two_party: float = 0.5
    pres_2020_dem_two_party: float = 0.5


@dataclass
class FakeRaceSet:
    races: list


def test_the_fakes_match_the_real_objects():
    """The lesson from `p.id`.

    A predictive block was broken for five days because the real poll object
    had `poll_id` and the fake in its test had `id`, so the test agreed with
    the bug. Any attribute these stand-ins expose must exist on the real thing,
    or these tests are checking themselves.
    """
    from midterms.config import Race
    from midterms.data.polls import NormalisedPoll, PollTable

    real_poll = {f.name for f in NormalisedPoll.__dataclass_fields__.values()}
    assert {f.name for f in FakePoll.__dataclass_fields__.values()} <= real_poll

    real_table = set(PollTable.__dataclass_fields__) | {"race_polls", "national"}
    assert set(FakeTable.__dataclass_fields__) | {"race_polls", "national"} <= real_table

    real_race = set(Race.__dataclass_fields__)
    assert set(FakeRace.__dataclass_fields__) <= real_race


def _raw(**overrides):
    out = {t: [{"id": f"{t}-1"}] for t in ALL_POLL_TYPES}
    out.update(overrides)
    return out


def _healthy(chamber="house", n_races=60):
    polls = [FakePoll(f"{chamber}-2026-R{i:03d}") for i in range(n_races)]
    table = FakeTable(polls=polls + [FakePoll("__national__")])
    races = FakeRaceSet([
        FakeRace(f"{chamber}-2026-R{i:03d}", 0.52, 0.51) for i in range(n_races)
    ])
    return table, races


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


# --- the reconstructions ----------------------------------------------------

def test_a_snapshot_missing_a_poll_type_fails(tmp_path, monkeypatch):
    """`midterms fetch` omitted us-representative and exited 0."""
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    table, races = _healthy()
    raw = _raw(**{"us-representative": []})

    report = health.build("house", raw, table, races, today=date(2026, 10, 15))

    assert not report.ok
    assert not _check(report, "snapshot_complete").ok
    assert "us-representative" in _check(report, "snapshot_complete").detail


def test_a_chamber_reading_the_other_chambers_polls_fails(tmp_path, monkeypatch):
    """forward.py's Senate defaults: no polls of this chamber's own type."""
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    table = FakeTable(polls=[FakePoll("__national__")])
    races = FakeRaceSet([FakeRace("house-2026-R001")])

    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    assert not _check(report, "race_polls_used").ok


def test_the_stale_roster_fails_at_the_ratio_it_actually_had(tmp_path, monkeypatch):
    """27 of 65 districts unusable, which ran green for seven weeks."""
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    used = [FakePoll(f"house-2026-U{i:03d}") for i in range(38)]
    blocked = {f"house-2026-B{i:03d}": {"Someone New"} for i in range(27)}
    table = FakeTable(polls=used, unknown_candidates=blocked)
    # The full chamber, not just the polled part. The denominator has to be
    # races that HAVE poll data, because nine House districts in ten are never
    # polled at all: against all 435 the stale roster reads as 6% and the check
    # never fires on the chamber it matters most for.
    others = [FakeRace(f"house-2026-X{i:03d}") for i in range(370)]
    races = FakeRaceSet([FakeRace(p.race_id) for p in used]
                        + [FakeRace(r) for r in blocked] + others)
    assert len(races.races) == 435

    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))
    check = _check(report, "roster_reads_the_feed")

    assert not check.ok
    assert check.value["blocked"] == 27
    assert check.value["fraction"] == pytest.approx(27 / 65, abs=1e-3)


def test_legitimate_same_party_races_do_not_fail_it(tmp_path, monkeypatch):
    """CA-11 is Chan vs Wiener and CA-40 is Calvert vs Kim.

    Both are real general elections the two-party model cannot read, and both
    are correct refusals. They are invisible to this check because their names
    resolve -- which is the distinction the whole check rests on.
    """
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    table, races = _healthy(n_races=60)
    # Rejected, but with every name resolved: nothing lands in
    # unknown_candidates, so nothing counts as blocked.
    table.rejections = {"no single D-vs-R matchup": 17}

    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    assert _check(report, "roster_reads_the_feed").ok
    assert _check(report, "roster_reads_the_feed").value["blocked"] == 0


def test_a_sudden_collapse_in_coverage_fails(tmp_path, monkeypatch):
    """The shared signature: races quietly stop being used.

    Nothing ages out before this election -- max_age_days is 400 -- so a drop
    of this size is a fault whatever caused it. This is the check meant to
    catch the next bug rather than the last one.
    """
    path = tmp_path / "house.json"
    monkeypatch.setattr(health, "report_path", lambda c: path)
    path.write_text(json.dumps({"metrics": {"races_with_polls": 61}}), encoding="utf-8")

    table, races = _healthy(n_races=38)
    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    check = _check(report, "coverage_did_not_collapse")
    assert not check.ok
    assert check.value == {"before": 61, "after": 38}


def test_a_small_change_in_coverage_does_not_fail(tmp_path, monkeypatch):
    path = tmp_path / "house.json"
    monkeypatch.setattr(health, "report_path", lambda c: path)
    path.write_text(json.dumps({"metrics": {"races_with_polls": 61}}), encoding="utf-8")

    table, races = _healthy(n_races=58)
    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    assert _check(report, "coverage_did_not_collapse").ok


def test_a_race_with_no_lean_fails(tmp_path, monkeypatch):
    """OR-03 was 30 points wrong; this catches the absent, not the merely wrong."""
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    table, races = _healthy(n_races=5)
    races.races[2].pres_2024_dem_two_party = 0.0

    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    assert not _check(report, "every_race_has_a_lean").ok


def test_a_stale_roster_stamp_fails(tmp_path, monkeypatch):
    """The rebuild step is continue-on-error, so it can fail quietly."""
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    monkeypatch.setattr(health, "_roster_built", lambda c: date(2026, 10, 1))
    table, races = _healthy()

    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    check = _check(report, "roster_is_fresh")
    assert not check.ok
    assert check.value["age_days"] == 14


# --- the report itself ------------------------------------------------------

def test_a_healthy_run_passes_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    monkeypatch.setattr(health, "_roster_built", lambda c: date(2026, 10, 15))
    table, races = _healthy()

    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    assert report.ok, [c.detail for c in report.checks if not c.ok]


def test_the_report_round_trips_and_names_what_each_check_guards(tmp_path, monkeypatch):
    """The `guards` field is the maintenance contract.

    It records which real failure each invariant stands for, so that a future
    reader deciding whether to loosen a threshold can see what they would be
    giving up.
    """
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    table, races = _healthy()
    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    written = health.write(report)
    loaded = json.loads(written.read_text(encoding="utf-8"))

    assert loaded["chamber"] == "house"
    assert loaded["ok"] is True
    assert loaded["metrics"]["races_with_polls"] == 60
    for check in loaded["checks"]:
        assert check["guards"], f"{check['name']} does not say what it guards"


def test_render_shows_the_guard_only_for_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "report_path", lambda c: tmp_path / f"{c}.json")
    table = FakeTable(polls=[FakePoll("__national__")])
    races = FakeRaceSet([FakeRace("house-2026-R001")])
    report = health.build("house", _raw(), table, races, today=date(2026, 10, 15))

    text = health.render(report)
    assert "FAIL  race_polls_used" in text
    assert "guards against:" in text
