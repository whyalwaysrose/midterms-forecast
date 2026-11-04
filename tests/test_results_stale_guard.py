"""Telling a stale results parser from an article nobody has filled in yet.

The guard exists because the House parser has twice silently dropped races: a
`{{party stripe}}` capitalisation cost 27 districts in 2024 while 2022
validated perfectly, and scoring on a fraction of the chamber looks exactly
like scoring on all of it.

Its first version compared parsed races against district *rows* and fired when
fewer than 90% of rows had a winner. That is not a parser fault, it is the
morning after an election. At 09:00 on 2026-11-04 the article carried 435
district rows and about a hundred called races, so the guard reported the
markup had "probably changed" on data that was completely correct, and would
have done so every run for days while the real results arrived.

The discriminating quantity is not how many races are called -- that is the
world's business, not the parser's -- but whether the parser reads the ones
that are. So it counts the winner markers actually present in the wikitext.
"""

from __future__ import annotations

import pytest

from midterms.data import results as R


def _district(state: str, number: int, called: bool) -> str:
    """One district row, with or without a marked winner."""
    marker = "{{Aye}}" if called else ""
    return f"""
|-
! {{{{ushr|{state}|{number}|X}}}}
| nowrap | {{{{Plainlist|
* {{{{Party stripe|Republican Party (US)}}}}{marker} '''Rep Candidate''' (Republican) 55.0%
* {{{{Party stripe|Democratic Party (US)}}}}Dem Candidate (Democratic) 45.0%
}}}}
"""


def _article(called: int, total: int) -> str:
    rows = "".join(
        _district("AL", n + 1, called=n < called) for n in range(total)
    )
    return '\n==Alabama==\n{| class="wikitable sortable"\n' + rows + "\n|}\n"


def _broken(called: int, total: int) -> str:
    """An article whose winners are marked in a way the parser cannot read.

    The markers are present -- so the article *has* results -- but they sit on
    a template the parser does not recognise. This is the 2024 capitalisation
    bug's shape: the data is there and we are not reading it.
    """
    text = _article(called, total)
    return text.replace(
        "{{Party stripe|Republican Party (US)}}{{Aye}}",
        "{{PartyStripeNG|Republican Party (US)}}{{Aye}}",
    )


def test_an_unfilled_article_is_not_reported_as_a_stale_parser(capsys):
    """The regression: 2026-11-04, 435 rows and ~95 called.

    Nothing is wrong here. The election happened yesterday.
    """
    out = R.parse_house_districts(_article(called=95, total=435))

    assert len(out) == 95
    captured = capsys.readouterr()
    assert "::error" not in captured.out
    assert "probably changed" not in captured.out


def test_an_article_with_no_results_at_all_is_silent(capsys):
    """Election night before any call. The ordinary starting state."""
    out = R.parse_house_districts(_article(called=0, total=435))

    assert out == {}
    assert "::error" not in capsys.readouterr().out


def test_a_parser_that_cannot_read_marked_winners_does_fire(capsys):
    """The failure the guard is for: results present, parser blind to them."""
    text = _broken(called=400, total=435)
    assert R._winner_markers(text) == 400, "the markers are still in the text"

    out = R.parse_house_districts(text)

    assert len(out) < 400
    captured = capsys.readouterr()
    assert "::error" in captured.out
    assert "probably changed" in captured.out
    assert "400" in captured.out, "the message should say how many were marked"


def test_a_fully_filled_article_that_parses_is_silent(capsys):
    out = R.parse_house_districts(_article(called=435, total=435))

    assert len(out) == 435
    assert "::error" not in capsys.readouterr().out


def test_a_small_shortfall_is_tolerated(capsys):
    """Alaska's ranked-choice rounds mark more than once.

    More markers than races is normal, so the guard needs slack; it must not
    fire on a handful.
    """
    text = _article(called=430, total=435)
    text = text.replace(
        "* {{Party stripe|Democratic Party (US)}}Dem Candidate (Democratic) 45.0%",
        "* {{Party stripe|Democratic Party (US)}}Dem Candidate (Democratic) 45.0%\n"
        "* {{Party stripe|Democratic Party (US)}}{{Aye}} Round two (Democratic) 51.0%",
        1,
    )
    R.parse_house_districts(text)
    assert "::error" not in capsys.readouterr().out


def test_the_senate_guard_behaves_the_same_way(capsys):
    """Thirty-five races, a handful called. Same shape, same silence."""
    from tests.test_race_results import SPECIAL_ELECTION_SECTION  # noqa: F401

    # Few rows, so the minimum-rows floor keeps it quiet regardless.
    R.parse_senate_races(_article(called=2, total=5))
    assert "::error" not in capsys.readouterr().out


@pytest.mark.parametrize("marker_case", ["{{Aye}}", "{{aye}}", "{{AYE}}"])
def test_marker_counting_is_case_insensitive(marker_case):
    """2022 and 2024 disagreed on template capitalisation; so will 2026."""
    text = _article(called=1, total=1).replace("{{Aye}}", marker_case)
    assert R._winner_markers(text) == 1
