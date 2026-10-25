"""Dropping pollsters who fabricate, including ones 538 never got to ban.

`exclude_banned` reads the "Banned by 538" column of a vendored CSV, and 538
closed in March 2025. The list is frozen: it cannot ever contain a firm exposed
afterwards. Fabrication did not stop when the list did.

Wedgewood Polls admitted on 2026-10-20 that it invented its data, after
FiftyPlusOne compared respondent identifiers in its raw Mississippi Senate file
against the platform it claimed to have used. The New York Times removed all of
its polls and Decision Desk HQ removed seventeen. Six were in this model, in
Georgia, Kansas, Maine, Mississippi, Ohio and Texas -- two of them toss-ups,
both leaning Democratic.

So the ban list grew a second half that this project maintains. These tests
cover the part that is easy to get wrong: that a locally banned firm is really
dropped rather than down-weighted, that it is matched robustly enough to
survive a change of filing name, and that the rejection does not claim 538
banned someone 538 never rated.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from midterms.config import ConfigError, ModelConfig, RaceSet
from midterms.data.polls import build_poll_table
from midterms.data.roster import Roster

AS_OF = date(2026, 8, 20)


@pytest.fixture(scope="module")
def setup():
    return RaceSet.load(), ModelConfig.load(), Roster.load()


def poll(**kw):
    base = {
        "id": "p1",
        "poll_type": "us-senator",
        "subject": "2026 Georgia",
        "pollster": "Test Pollster",
        "sample_size": 800,
        "population": "lv",
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
        "answers": [
            {"choice": "Jon Ossoff", "pct": 50.0},
            {"choice": "Mike Collins", "pct": 46.0},
        ],
        "sponsors": [],
        "partisan": None,
        "internal": False,
        "url": "https://example.test",
    }
    base.update(kw)
    return base


def _with_local_ban(cfg, names):
    ratings = replace(cfg.polls.pollster_ratings, banned_locally=tuple(names))
    return replace(cfg, polls=replace(cfg.polls, pollster_ratings=ratings))


def _table(races, cfg, roster, records):
    return build_poll_table(
        {"us-senator": records}, races, cfg, roster, as_of=AS_OF
    )


def test_a_locally_banned_pollster_is_dropped(setup):
    races, cfg, roster = setup
    cfg = _with_local_ban(cfg, ["Wedgewood Polls"])

    table = _table(races, cfg, roster, [
        poll(id="a", pollster="Wedgewood Polls"),
        poll(id="b", pollster="Honest Research"),
    ])

    assert [p.poll_id for p in table.race_polls] == ["b"]


def test_it_is_dropped_not_down_weighted(setup):
    """A fabricated number cannot be weighted into usefulness.

    An invented figure looks precise rather than noisy, so widening its
    variance would make the model *more* confident in the direction the
    fabricator chose, not less.
    """
    races, cfg, roster = setup
    cfg = _with_local_ban(cfg, ["Wedgewood Polls"])

    table = _table(races, cfg, roster, [poll(id="a", pollster="Wedgewood Polls")])

    assert table.race_polls == []
    assert not any("Wedgewood" in p.pollster for p in table.polls)


def test_the_rejection_does_not_blame_538(setup):
    """538 never rated Wedgewood, and saying so in a log would be false.

    The two halves of the list are different kinds of claim: one is frozen
    history, the other is a judgement this project made and has to defend.
    """
    races, cfg, roster = setup
    cfg = _with_local_ban(cfg, ["Wedgewood Polls"])

    table = _table(races, cfg, roster, [poll(id="a", pollster="Wedgewood Polls")])

    reasons = list(table.rejections)
    assert "pollster banned by this project: Wedgewood Polls" in reasons
    assert not any("banned by 538: Wedgewood" in r for r in reasons)


@pytest.mark.parametrize("filed_as", [
    "Wedgewood Polls",
    "wedgewood polls",
    "  Wedgewood   Polls  ",
    "WEDGEWOOD POLLS",
])
def test_a_change_of_filing_name_does_not_un_ban(setup, filed_as):
    """Matched on the normalised form, like the ratings table.

    A firm that has already lied about its data is not owed the benefit of an
    exact string match, and re-filing under a differently-cased or spaced name
    is the cheapest possible evasion.
    """
    races, cfg, roster = setup
    cfg = _with_local_ban(cfg, ["Wedgewood Polls"])

    table = _table(races, cfg, roster, [poll(id="a", pollster=filed_as)])

    assert table.race_polls == []


def test_an_unlisted_pollster_survives(setup):
    races, cfg, roster = setup
    cfg = _with_local_ban(cfg, ["Wedgewood Polls"])

    table = _table(races, cfg, roster, [poll(id="a", pollster="Wedgwood Associates")])

    assert [p.poll_id for p in table.race_polls] == ["a"]


def test_an_empty_local_list_changes_nothing(setup):
    """The default. Nobody is banned by this project unless named."""
    races, cfg, roster = setup
    assert ModelConfig.load().polls.pollster_ratings.banned_locally == (
        "Wedgewood Polls",
    ), "the committed config should carry exactly the documented ban"

    cleared = _with_local_ban(cfg, [])
    table = _table(races, cleared, roster, [poll(id="a", pollster="Wedgewood Polls")])

    assert [p.poll_id for p in table.race_polls] == ["a"]


def test_the_538_list_still_applies(setup):
    """Big Data Poll is banned by 538 and must stay banned."""
    races, cfg, roster = setup
    cfg = _with_local_ban(cfg, [])

    table = _table(races, cfg, roster, [poll(id="a", pollster="Big Data Poll")])

    assert table.race_polls == []
    assert any("banned by 538" in r for r in table.rejections)


def test_turning_exclusion_off_keeps_everyone(setup):
    races, cfg, roster = setup
    ratings = replace(
        cfg.polls.pollster_ratings,
        exclude_banned=False,
        banned_locally=("Wedgewood Polls",),
    )
    cfg = replace(cfg, polls=replace(cfg.polls, pollster_ratings=ratings))

    table = _table(races, cfg, roster, [poll(id="a", pollster="Wedgewood Polls")])

    assert [p.poll_id for p in table.race_polls] == ["a"]


def test_a_string_instead_of_a_list_is_refused():
    """`banned_locally: Wedgewood Polls` would iterate as characters."""
    from midterms.config import _polls_config

    base = ModelConfig.load().raw["polls"]
    bad = {**base, "pollster_ratings": {
        **base["pollster_ratings"], "banned_locally": "Wedgewood Polls",
    }}
    with pytest.raises(ConfigError, match="must be a list"):
        _polls_config(bad)


def test_no_banned_pollster_remains_in_the_calibration_record():
    """The record scores the model against observations.

    A fabricated poll there is not a prediction the model got wrong; it is a
    number nobody collected, and leaving it in would quietly corrupt the
    coverage and PIT figures the Testing tab reports.
    """
    import csv

    from midterms import paths

    banned = {
        n.casefold()
        for n in ModelConfig.load().polls.pollster_ratings.banned_locally
    }
    for chamber in ("senate", "house"):
        path = paths.DATA_DIR / "calibration" / f"forward_{chamber}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        offenders = [r for r in rows if r["pollster"].casefold() in banned]
        assert not offenders, (
            f"{len(offenders)} rows in {path.name} come from a banned pollster"
        )
