"""Where the 2026 result comes from, without anyone having to type it in.

Scoring needs the outcome, and the outcome has to arrive on its own or the
scorer is a thing somebody has to remember to feed. Every machine-readable
results feed worth having is a paid product -- AP and Decision Desk both sell
theirs -- so this reads the chamber totals off Wikipedia's election articles,
which are free, edited within hours, and already structured.

WHY THE INFOBOX AND NOT THE RESULTS TABLES
------------------------------------------
The prose tables differ per chamber and per cycle: in 2022 the Senate's summary
lived under "Summary results" and the House's under "Results", with different
column layouts. The infobox is the one part with a schema -- ``party1``,
``seats1``, ``seats_after1``, ``majority_seats`` -- and both 2026 articles
already carry it with the seat fields present and empty, waiting for the night.

Even so, this is scraping a wiki, so it is written to fail loudly rather than
quietly:

* every parse is validated against the chamber size, which is the check that
  catches nearly everything -- a mis-parse almost never sums to exactly 435;
* an empty seat field returns "not yet", not zero;
* party names are matched on both spellings Wikipedia uses across cycles
  ("Democratic Party (US)" and "Democratic Party (United States)").

INDEPENDENTS ARE NOT SILENTLY ASSIGNED
--------------------------------------
The forecast counts the Democratic *caucus*, with Nebraska's Dan Osborn on the
Democratic side, because chamber arithmetic forces a choice. Wikipedia counts
parties, and on election night nobody will yet know who caucuses with whom.

So independents are returned as their own number and never folded in here. The
caller decides, the same way ``compare_to_markets`` reports the Senate both
ways, and the scorer prints both. Guessing here would put a silent assumption
underneath the one measurement this project only gets to make once.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"

PAGES = {
    "senate": "2026 United States Senate elections",
    "house": "2026 United States House of Representatives elections",
}

#: Total seats in each chamber, and the check every parse has to pass.
CHAMBER_SIZE = {"senate": 100, "house": 435}

#: Wikipedia has used both spellings across cycles.
PARTY_ALIASES = {
    "democratic party (us)": "D",
    "democratic party (united states)": "D",
    "republican party (us)": "R",
    "republican party (united states)": "R",
    "independent": "I",
    "independent politician": "I",
}

USER_AGENT = (
    "midterms-forecast/1.0 (https://github.com/whyalwaysrose/midterms-forecast) "
    "election-result-scoring"
)


@dataclass(frozen=True)
class ChamberSeats:
    """Seats by party after the election, as Wikipedia reports them."""

    chamber: str
    dem: int
    rep: int
    ind: int
    majority_seats: int
    source: str
    field: str          # which infobox field this came from

    @property
    def total(self) -> int:
        return self.dem + self.rep + self.ind


def _fetch_lead_wikitext(page: str, timeout: float = 30.0) -> str:
    """The lead section's wikitext, which is where the infobox lives.

    Uses `requests` like the rest of the project's fetching, so it picks up the
    same certificate bundle -- the standard library's urllib trusts the system
    store, which on this machine is stale enough to reject Wikipedia outright.
    """
    response = requests.get(
        WIKI_API,
        params={
            "action": "parse", "page": page, "prop": "wikitext",
            "section": "0", "format": "json", "formatversion": "2",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Wikipedia: {payload['error'].get('info', 'unknown error')}")
    return payload["parse"]["wikitext"]


def _fetch_full_wikitext(page: str, timeout: float = 60.0) -> str:
    """The whole article. The per-state results tables are spread across fifty
    sections, so one request for everything beats fifty for the pieces."""
    response = requests.get(
        WIKI_API,
        params={"action": "parse", "page": page, "prop": "wikitext",
                "format": "json", "formatversion": "2"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Wikipedia: {payload['error'].get('info', 'unknown error')}")
    return payload["parse"]["wikitext"]


def _clean(value: str) -> str:
    """Strip the wiki furniture an infobox value collects.

    Seat counts arrive as things like ``'''222'''``, ``213 <!--Vacancies-->``
    and ``45{{efn|name=Independents}}``.
    """
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = value.replace("'''", "").replace("''", "")
    value = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", value)
    return value.strip()


def _seat_count(value: str) -> int | None:
    """The number at the front of an infobox seat field, or None.

    Stripping templates wholesale does not survive contact with these fields.
    2022's Senate carried ``seats_after1 = 49{{efn|name=Sinema|...<ref></ref>}}``
    -- a footnote long enough to explain a party switch, containing its own
    markup -- and anything short of a real wikitext parser leaves some of it
    behind, at which point an otherwise good ``49`` is discarded as "not a
    number".

    A seat count is the integer at the start; everything after it is prose.
    Reading it that way is robust to whatever the footnote contains, and the
    chamber-size check is what catches a genuine misread.
    """
    match = re.match(r"\s*(\d+)", _clean(value))
    return int(match.group(1)) if match else None


def parse_seats(wikitext: str, chamber: str, source: str) -> ChamberSeats | None:
    """Pull seats-by-party out of an election infobox, or return None.

    Returns None whenever the answer is not yet knowable -- the usual case
    before election night, when the fields exist but are empty.
    """
    fields: dict[str, str] = {}
    for match in re.finditer(r"^\s*\|\s*([a-z_0-9]+)\s*=(.*)$", wikitext, re.M | re.I):
        fields.setdefault(match.group(1).lower(), match.group(2))

    parties: dict[int, str] = {}
    for key, raw in fields.items():
        index = re.fullmatch(r"party(\d+)", key)
        if index:
            code = PARTY_ALIASES.get(_clean(raw).lower())
            if code:
                parties[int(index.group(1))] = code

    if not parties:
        log.warning("%s: no recognised party fields in the infobox", chamber)
        return None

    majority_seats = _seat_count(fields.get("majority_seats", "")) or 0

    # `seats_after` is the chamber total and `seats` is seats won in this
    # election. For the House every seat is up so they agree; for the Senate
    # only `seats_after` can sum to 100. Try both and keep whichever validates.
    for field in ("seats_after", "seats"):
        counts = {"D": 0, "R": 0, "I": 0}
        seen = False
        for index, code in parties.items():
            raw = fields.get(f"{field}{index}", "")
            if not raw.strip():
                continue
            count = _seat_count(raw)
            if count is None:
                log.warning("%s: %s%d starts with no number: %r",
                            chamber, field, index, _clean(raw)[:60])
                continue
            counts[code] += count
            seen = True
        if not seen:
            continue

        seats = ChamberSeats(
            chamber=chamber, dem=counts["D"], rep=counts["R"], ind=counts["I"],
            majority_seats=majority_seats, source=source, field=field,
        )
        expected = CHAMBER_SIZE[chamber]
        if seats.total == expected:
            return seats
        log.warning(
            "%s: %s fields sum to %d, not %d -- rejected",
            chamber, field, seats.total, expected,
        )

    return None


def fetch_chamber_seats(chamber: str, page: str | None = None) -> ChamberSeats | None:
    """Read a chamber's final seat counts from Wikipedia, or None if not yet in.

    Never raises on a missing result -- an election that has not happened is the
    normal case for most of this file's life. It does raise if the network or
    the API itself fails, because that is a real fault and should be visible.
    """
    page = page or PAGES[chamber]
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page.replace(' ', '_'))}"
    wikitext = _fetch_lead_wikitext(page)
    seats = parse_seats(wikitext, chamber, url)
    if seats is None:
        log.info("%s: no usable seat totals yet at %s", chamber, url)
    return seats


# ---------------------------------------------------------------------------
# Per-race results
#
# The chamber totals above settle who won. They cannot touch the question
# METHODOLOGY 8c pre-registered as the one that carries evidence: did the races
# called at 70% come true 70% of the time? That needs an outcome per race, and
# for the House that means 435 of them.
#
# Wikipedia's election article has them, in the per-state sections, as a table
# row per district. The row is identifiable because it opens a header cell with
# the {{ushr}} template, and the winner is marked with {{Aye}}:
#
#     ! {{ushr|NY|19|X}}
#     ...
#     * {{Party stripe|Republican Party (US)}}{{Aye}} '''[[Marc Molinaro]]''' (Republican) 50.8%
#     * {{Party stripe|Democratic Party (US)}}[[Josh Riley]] (Democratic) 49.2%
#
# Three things that are not obvious, each found by running this against 2022
# and counting what came back:
#
# * A district cell can carry table attributes -- `! rowspan=2 | {{ushr|WV|2|X}}`
#   when redistricting gave a seat two incumbents. Matching only `! {{ushr`
#   silently lost eleven states' worth of districts.
# * {{ushr}} also appears *inside* cells, to say an incumbent was redistricted
#   from somewhere else. Those carry a `C` suffix where row markers carry `X`,
#   which is what separates 435 real rows from 783 template uses.
# * Special elections held on the same day appear in their own section, giving
#   a second row for the same district. Only rows under a state heading are the
#   general election, which is what the forecast predicted.
# ---------------------------------------------------------------------------

#: Full state names as Wikipedia headings them, to postal codes. Only these
#: fifty are parsed: territory delegates are not among the 435 and must not be
#: counted, and "Special elections" is not a state.
STATE_HEADINGS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

HOUSE_SEATS = 435

#: A level-2 heading, which is how the article separates states from each other
#: and from the special-elections section.
_HEADING = re.compile(r"^==\s*([^=\n]+?)\s*==\s*$", re.M)

#: A district row. The optional group swallows cell attributes such as
#: `rowspan=2 |`; the trailing `X` is what distinguishes a row marker from an
#: in-cell cross-reference.
_DISTRICT_ROW = re.compile(
    r"^!(?:[^|{}\n]*\|)?\s*\{\{ushr\|([A-Z]{2})\|([0-9]+|AL)\|X\}\}", re.M
)

#: Any short parenthesised group on a candidate line. The party is one of
#: these, but so is the disambiguation in "[[Barry Moore (American politician)]]",
#: which is why the party is chosen by what it says rather than by position.
_PAREN = re.compile(r"\(([^()]{1,40})\)")

#: A vote share, where one has been published yet.
_PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")


@dataclass(frozen=True)
class RaceOutcome:
    """One race, as Wikipedia reports it."""

    race_id: str
    winner_party: str           # "D", "R" or "O" for anyone else
    dem_pct: float | None
    rep_pct: float | None

    @property
    def dem_won(self) -> bool:
        return self.winner_party == "D"

    @property
    def dem_margin(self) -> float | None:
        """Two-party Democratic margin in points, or None.

        None whenever the contest was not one Democrat against one Republican:
        an uncontested seat, or the same-party run-offs California, Washington
        and Louisiana produce. The model forecasts a two-party margin, so there
        is nothing to compare against in those races -- but the *winner* is
        still known, so the win probability is still scored. Inventing a +-100
        margin would corrupt interval coverage with races the model never tried
        to place.
        """
        if self.dem_pct is None or self.rep_pct is None:
            return None
        total = self.dem_pct + self.rep_pct
        if total <= 0:
            return None
        return 100.0 * (self.dem_pct - self.rep_pct) / total


def _party_code(name: str) -> str:
    """Map a ballot party label to D, R, or O.

    Two states do not call their Democratic party "Democratic" on the ballot:
    Minnesota's is the Democratic-Farmer-Labor party and North Dakota's the
    Democratic-NPL. Wikipedia prints "(DFL)" and "(D-NPL)", and without these a
    whole state's delegation lands in "other" -- which is how four Minnesota
    seats first came back unattributed.
    """
    lowered = name.strip().lower()
    if lowered.startswith(("democratic", "dfl", "d-npl")):
        return "D"
    if lowered.startswith("republican"):
        return "R"
    return "O"


def _parse_district_block(block: str) -> tuple[str, float | None, float | None] | None:
    """Winner and the two major-party shares from one district's table row.

    Returns None when no winner is marked, which is the ordinary state of a
    race that has been held but not yet called.
    """
    winner = None
    best: dict[str, float] = {}
    for line in block.splitlines():
        # Candidate lines are bulleted in a {{plainlist}} when there are
        # several, but an unopposed candidate is written straight into the cell
        # with no bullet at all -- which silently lost twelve safe seats,
        # exactly the races most likely to be unopposed. What every candidate
        # line does carry, in both forms, is a party stripe.
        #
        # Matched case-insensitively: 2022 writes {{Party stripe}} and 2024
        # {{party stripe}}, sometimes both within one state. Case-sensitivity
        # here cost 27 districts in 2024 while leaving 2022 perfect, which is
        # the worst possible way for this to be wrong -- one cycle validates
        # clean and the next quietly loses six percent of the chamber.
        if "{{party stripe|" not in line.lower():
            continue
        groups = list(_PAREN.finditer(line))
        if not groups:
            continue
        # The party is the parenthesised group that names a party -- not
        # simply the first one, which in "[[Barry Moore (American politician)]]"
        # is a disambiguator.
        party = next((g for g in groups if _party_code(g.group(1)) != "O"), groups[-1])
        code = _party_code(party.group(1))

        # A share may not exist yet. An unopposed candidate is often written
        # "(Democratic)" followed by ''Unopposed'', and on election night a
        # race can be called hours before its percentages are typed in. Either
        # way the winner is known, which is what the win probability is scored
        # against; the margin simply stays unknown.
        share = _PERCENT.search(line[party.end():])
        if share:
            pct = float(share.group(1))
            # Several candidates can share a party in a top-two run-off, and
            # Alaska lists two rounds; the leading figure is the party's.
            best[code] = max(best.get(code, 0.0), pct)
        # Case-insensitive, and the FIRST marked line wins. Both matter for
        # Alaska, whose ranked-choice block lists a first round marked {{mby}}
        # and then a final round marked {{aye}} -- lowercase. Taking the max
        # share per party across both rounds lands on the final-round numbers,
        # since a surviving candidate's share only grows as others are
        # eliminated.
        if "{{aye}}" in line.lower() and winner is None:
            winner = code

    if winner is None:
        return None
    return winner, best.get("D"), best.get("R")


def parse_house_districts(wikitext: str) -> dict[str, RaceOutcome]:
    """Every called House district in the article, keyed by race id.

    Counts rows as well as outcomes, because the dangerous failure here is not
    an exception -- it is a format change that drops a slice of the chamber and
    leaves a plausible number behind. Finding 435 rows and parsing 300 of them
    means the layout moved; finding 435 and parsing 120 on election night means
    most races are simply not called yet. The two look identical from the
    outcome count alone, so both numbers are reported.
    """
    out: dict[str, RaceOutcome] = {}
    rows_seen = 0

    headings = list(_HEADING.finditer(wikitext))
    for index, heading in enumerate(headings):
        state = STATE_HEADINGS.get(heading.group(1))
        if state is None:
            continue                      # territory, or "Special elections"
        end = headings[index + 1].start() if index + 1 < len(headings) else len(wikitext)
        section = wikitext[heading.end():end]

        rows = list(_DISTRICT_ROW.finditer(section))
        for position, row in enumerate(rows):
            if row.group(1) != state:
                continue                  # a stray reference to another state
            number = row.group(2)
            # Wikipedia writes at-large seats "AL"; this project numbers them 01.
            district = "01" if number == "AL" else f"{int(number):02d}"
            race_id = f"house-2026-{state}-{district}"

            stop = rows[position + 1].start() if position + 1 < len(rows) else len(section)
            rows_seen += 1
            parsed = _parse_district_block(section[row.end():stop])
            if parsed is None:
                continue
            winner, dem, rep = parsed
            out.setdefault(race_id, RaceOutcome(race_id, winner, dem, rep))

    log.info("house: %d district rows, %d with a called winner", rows_seen, len(out))
    if rows_seen >= HOUSE_SEATS and 0 < len(out) < rows_seen * 0.9:
        # Some called, most not parsed. Election night produces few-or-none;
        # a layout change produces most-but-not-all, which is this.
        message = (
            f"house results: {rows_seen} district rows but only {len(out)} parsed "
            "-- the article's candidate markup has probably changed. Scoring on "
            "this would quietly grade a fraction of the chamber."
        )
        log.error(message)
        print(f"::error title=House results parser may be stale::{message}")
    return out


#: Level-3 headings, which is how the Senate article separates its race
#: summary into the specials held early and the regular class elections.
_SUBHEADING = re.compile(r"^===\s*([^=\n]+?)\s*===\s*$", re.M)

#: The two subsections that between them hold every race on the ballot. Both
#: are needed: 2026 runs 33 regular Class 2 seats plus the Florida and Ohio
#: specials, and taking only the first section would quietly drop two races
#: without changing anything that looks wrong.
_SENATE_SECTIONS = (
    "elections leading to the next congress",
    "special elections during the preceding congress",
)

#: A Senate row opens with a link to that state's own election article.
_SENATE_ROW = re.compile(
    r"^!\s*(?:[^|{}\n]*\|)?\s*\[\[\d{4} United States Senate "
    r"(?:special )?election in ([A-Za-z ]+?)\s*\|",
    re.M,
)


def parse_senate_races(wikitext: str) -> dict[str, RaceOutcome]:
    """Every called Senate race in the article, keyed by race id.

    Scoped to the race-summary subsections. The state link that identifies a
    row also appears throughout the candidate and primary tables further up the
    article -- matching it everywhere returns 118 rows for 35 races.
    """
    out: dict[str, RaceOutcome] = {}
    rows_seen = 0

    subs = list(_SUBHEADING.finditer(wikitext))
    for index, sub in enumerate(subs):
        if sub.group(1).strip().lower() not in _SENATE_SECTIONS:
            continue
        end = subs[index + 1].start() if index + 1 < len(subs) else len(wikitext)
        section = wikitext[sub.end():end]

        rows = list(_SENATE_ROW.finditer(section))
        for position, row in enumerate(rows):
            state = STATE_HEADINGS.get(row.group(1).strip())
            if state is None:
                continue
            rows_seen += 1
            stop = rows[position + 1].start() if position + 1 < len(rows) else len(section)
            parsed = _parse_district_block(section[row.end():stop])
            if parsed is None:
                continue
            winner, dem, rep = parsed
            race_id = f"senate-2026-{state}"
            out.setdefault(race_id, RaceOutcome(race_id, winner, dem, rep))

    log.info("senate: %d race rows, %d with a called winner", rows_seen, len(out))
    if rows_seen >= 30 and 0 < len(out) < rows_seen * 0.9:
        message = (
            f"senate results: {rows_seen} race rows but only {len(out)} parsed "
            "-- the article's candidate markup has probably changed."
        )
        log.error(message)
        print(f"::error title=Senate results parser may be stale::{message}")
    return out


def fetch_senate_races(page: str | None = None) -> dict[str, RaceOutcome]:
    """Per-race Senate results from Wikipedia, empty until they exist."""
    page = page or PAGES["senate"]
    return parse_senate_races(_fetch_full_wikitext(page))


def fetch_race_results(chamber: str, page: str | None = None) -> dict[str, RaceOutcome]:
    """Per-race results for either chamber."""
    if chamber == "house":
        return fetch_house_districts(page)
    return fetch_senate_races(page)


def fetch_house_districts(page: str | None = None) -> dict[str, RaceOutcome]:
    """Per-district House results from Wikipedia, empty until they exist."""
    page = page or PAGES["house"]
    wikitext = _fetch_full_wikitext(page)
    results = parse_house_districts(wikitext)
    log.info("house: %d districts called at %s", len(results), page)
    return results
