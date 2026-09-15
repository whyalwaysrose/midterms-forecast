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
