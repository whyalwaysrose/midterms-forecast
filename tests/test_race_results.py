"""Reading 435 district results and 35 Senate results off Wikipedia.

METHODOLOGY 8c pre-registered the per-race calibration as the evidence and the
chamber outcome as an anecdote. Until this parser existed, only the anecdote
could be computed.

Every fixture below is a real markup variation found by running the parser
against 2018, 2022 and 2024 and counting what came back. None of them raises;
each one silently drops races, which on election night would mean grading a
fraction of the chamber and never knowing. In order of how much they cost:

* case: 2024 writes {{party stripe}} where 2022 writes {{Party stripe}} -- 27
  districts, and 2022 still validated perfectly, which is the worst way for a
  parser to be wrong;
* cell attributes: `! rowspan=2 | {{ushr|WV|2|X}}` where a seat had two
  incumbents -- eleven states;
* no bullet: an unopposed candidate is written straight into the cell -- twelve
  safe seats, the ones most likely to be unopposed;
* no percentage at all: the winner is marked before the shares are typed in,
  which is the normal state of a race on election night.
"""

from __future__ import annotations

from midterms.data import results as R

# --- fixtures, each a real variation ----------------------------------------

TWO_CANDIDATES = """
==Alabama==
{| class="wikitable sortable"
|-
! {{ushr|AL|2|X}}
| {{Shading PVI|R|17}}
| [[Barry Moore (American politician)|Barry Moore]]
| nowrap | {{Plainlist|
* {{Party stripe|Republican Party (US)}}{{Aye}} '''[[Barry Moore (American politician)|Barry Moore]]''' (Republican) 69.1%
* {{Party stripe|Democratic Party (US)}}Phyllis Harvey-Hall (Democratic) 29.2%
}}
|}
"""

LOWERCASE_TEMPLATES = """
==Delaware==
|-
! {{ushr|DE|AL|X}}
| nowrap | {{Plainlist}}
* {{party stripe|Democratic Party (US)}}{{aye}} '''[[Sarah McBride]]''' (Democratic) 57.6%
* {{party stripe|Republican Party (US)}}John Whalen (Republican) 41.1%
"""

CELL_ATTRIBUTES = """
==West Virginia==
|-
! rowspan=2 | {{ushr|WV|2|X}}
| [[David McKinley]]<br />{{small|Redistricted from the {{ushr|WV|1|C}}}}
| nowrap | {{Plainlist|
* {{Party stripe|Republican Party (US)}}{{Aye}} '''[[Alex Mooney]]''' (Republican) 65.9%
* {{Party stripe|Democratic Party (US)}}Barry Wendell (Democratic) 34.1%
}}
"""

UNOPPOSED_INLINE = """
==Texas==
|-
! {{ushr|TX|11|X}}
| [[August Pfluger]]
| nowrap | {{Party stripe|Republican Party (US)}}{{Aye}} '''[[August Pfluger]]''' (Republican) 100%
"""

CALLED_WITHOUT_SHARES = """
==Alabama==
|-
! {{ushr|AL|7|X}}
| nowrap | {{Plainlist |
* {{Party stripe|Democratic Party (US)}}{{Aye}} '''{{Sortname|Terri|Sewell}}''' (Democratic)
* ''Unopposed''
}}
"""

MINNESOTA_DFL = """
==Minnesota==
|-
! {{ushr|MN|2|X}}
| nowrap | {{Plainlist|
* {{Party stripe|Minnesota Democratic-Farmer-Labor Party}}{{Aye}} '''[[Angie Craig]]''' (DFL) 51.0%
* {{Party stripe|Republican Party (US)}}Tyler Kistner (Republican) 45.7%
}}
"""

SAME_PARTY_RUNOFF = """
==California==
|-
! {{ushr|CA|34|X}}
| nowrap | {{Plainlist|
* {{Party stripe|Democratic Party (US)}}{{Aye}} '''[[Jimmy Gomez]]''' (Democratic) 51.4%
* {{Party stripe|Democratic Party (US)}}David Kim (Democratic) 48.6%
}}
"""

NOT_CALLED_YET = """
==Arizona==
|-
! {{ushr|AZ|1|X}}
| {{sortname|David|Schweikert}}
| Incumbent running.
"""

SPECIAL_ELECTION_SECTION = """
==Special elections==
|-
! {{ushr|NY|19|X}}
| nowrap | {{Plainlist|
* {{Party stripe|Democratic Party (US)}}{{Aye}} '''[[Pat Ryan]]''' (Democratic) 51.1%
* {{Party stripe|Republican Party (US)}}Marc Molinaro (Republican) 48.9%
}}

==New York==
|-
! {{ushr|NY|19|X}}
| nowrap | {{Plainlist|
* {{Party stripe|Republican Party (US)}}{{Aye}} '''[[Marc Molinaro]]''' (Republican) 50.8%
* {{Party stripe|Democratic Party (US)}}[[Josh Riley]] (Democratic) 49.2%
}}
"""


def only(wikitext: str) -> R.RaceOutcome:
    parsed = R.parse_house_districts(wikitext)
    assert len(parsed) == 1, f"expected one district, got {sorted(parsed)}"
    return next(iter(parsed.values()))


# --- the ordinary case ------------------------------------------------------


def test_a_two_candidate_race_yields_winner_and_margin():
    """The disambiguator in "[[Barry Moore (American politician)]]" must not be
    read as the party, which is why the party is chosen by what it says rather
    than by being the first parenthesis on the line."""
    outcome = only(TWO_CANDIDATES)
    assert outcome.race_id == "house-2026-AL-02"
    assert outcome.winner_party == "R" and not outcome.dem_won
    assert outcome.dem_pct == 29.2 and outcome.rep_pct == 69.1
    assert outcome.dem_margin < 0


# --- the variations that each cost races ------------------------------------


def test_lowercase_templates_are_recognised():
    """2024 writes {{party stripe}} and {{aye}}; 2022 writes them capitalised.

    This is the one that matters most, because 2022 validated perfectly with a
    case-sensitive match while 2024 quietly lost 27 districts.
    """
    outcome = only(LOWERCASE_TEMPLATES)
    assert outcome.race_id == "house-2026-DE-01"      # at-large is numbered 01
    assert outcome.dem_won and outcome.dem_pct == 57.6


def test_a_district_cell_may_carry_table_attributes():
    outcome = only(CELL_ATTRIBUTES)
    assert outcome.race_id == "house-2026-WV-02"
    assert outcome.winner_party == "R"


def test_an_in_cell_district_reference_is_not_a_row():
    """`{{ushr|WV|1|C}}` says an incumbent was redistricted from elsewhere. It
    carries a C where a row marker carries an X, which is what separates 435
    real rows from 783 uses of the template."""
    assert "WV|1|C" in CELL_ATTRIBUTES
    assert "house-2026-WV-01" not in R.parse_house_districts(CELL_ATTRIBUTES)


def test_an_unopposed_candidate_written_without_a_bullet_is_found():
    """With one candidate there is no {{plainlist}}; the name goes straight in
    the cell. Missing this lost twelve seats, all of them safe."""
    outcome = only(UNOPPOSED_INLINE)
    assert outcome.race_id == "house-2026-TX-11"
    assert outcome.winner_party == "R" and outcome.rep_pct == 100.0
    assert outcome.dem_pct is None


def test_a_race_called_before_its_shares_are_published_still_scores_a_winner():
    """The ordinary state of a race on election night: a call, no numbers yet.

    The winner is what the win probability is scored against, so it must be
    recorded; the margin stays unknown rather than being invented.
    """
    outcome = only(CALLED_WITHOUT_SHARES)
    assert outcome.dem_won
    assert outcome.dem_pct is None and outcome.dem_margin is None


def test_the_dfl_is_the_democratic_party():
    """Minnesota's ballot line is the Democratic-Farmer-Labor party. Without
    this its whole delegation lands in "other"."""
    assert only(MINNESOTA_DFL).dem_won


# --- what must NOT be scored ------------------------------------------------


def test_a_same_party_runoff_has_a_winner_but_no_two_party_margin():
    """California, Washington and Louisiana can run two Democrats against each
    other. A Democrat won, so the win probability is scored -- but there is no
    two-party margin, and inventing +-100 would corrupt interval coverage with
    races the model never tried to place."""
    outcome = only(SAME_PARTY_RUNOFF)
    assert outcome.dem_won
    assert outcome.rep_pct is None
    assert outcome.dem_margin is None


def test_an_uncalled_race_is_absent_rather_than_guessed():
    assert R.parse_house_districts(NOT_CALLED_YET) == {}


def test_a_concurrent_special_election_does_not_displace_the_general():
    """NY-19 was on the ballot twice in 2022: a special for the old seat and the
    general for the new one. Only the general is what the forecast predicted,
    and it is the one under the state heading."""
    parsed = R.parse_house_districts(SPECIAL_ELECTION_SECTION)
    assert set(parsed) == {"house-2026-NY-19"}
    assert parsed["house-2026-NY-19"].winner_party == "R"   # the general


# --- the guard --------------------------------------------------------------


def test_a_full_set_of_rows_that_mostly_fail_to_parse_is_reported(capsys):
    """The failure that would otherwise be silent.

    Finding 435 rows and parsing 400 means the markup moved. Finding 435 and
    parsing none means election night. Only the first is a fault, and it has to
    announce itself rather than return a plausible number.
    """
    good = """
==Alabama==
|-
! {{ushr|AL|1|X}}
| nowrap | {{Party stripe|Republican Party (US)}}{{Aye}} '''A''' (Republican) 60.0%
"""
    broken = "\n".join(
        f"|-\n! {{{{ushr|AL|{n}|X}}}}\n| nowrap | nothing parseable here"
        for n in range(2, 440)
    )
    R.parse_house_districts(good + broken)
    assert "::error" in capsys.readouterr().out


def test_no_rows_at_all_is_quiet():
    """Before the election there is nothing to say, every single day."""
    assert R.parse_house_districts("==Alabama==\nnothing here\n") == {}


# --- senate -----------------------------------------------------------------

SENATE_SUMMARY = """
===Elections leading to the next Congress===
{| class="wikitable"
|-
! [[2026 United States Senate election in Georgia|Georgia]]
| {{Sortname|Jon|Ossoff}}
| nowrap | {{Plainlist|
* {{Party stripe|Democratic Party (US)}}{{Aye}} '''[[Jon Ossoff]]''' (Democratic) 51.4%
* {{Party stripe|Republican Party (US)}}A Challenger (Republican) 48.6%
}}
|}

===Candidates===
! [[2026 United States Senate election in Georgia|Georgia]]
* {{Party stripe|Democratic Party (US)}}{{Aye}} '''Someone Else''' (Democratic) 99.0%
"""


def test_senate_rows_are_read_only_from_the_race_summary():
    """The state link that identifies a row also appears all through the
    candidate and primary tables. Matching it everywhere returned 118 rows for
    35 races, and would have scored a primary as though it were the election.
    """
    parsed = R.parse_senate_races(SENATE_SUMMARY)
    assert set(parsed) == {"senate-2026-GA"}
    assert parsed["senate-2026-GA"].dem_pct == 51.4      # not the 99.0 below
