"""What a prediction market thinks, alongside what the model thinks.

Shown for contrast, never as an input. Feeding market prices into the model
would be circular -- traders read forecasts, including this kind of one -- and
would destroy the calibration the error scales were fitted for. Nothing in this
module is imported by anything under ``model/``, and that should stay true.

**Probabilities, not prices.** A prediction-market contract that settles at $1
if an outcome happens trades at the market's implied probability, so the price
*is* the number worth showing. Presenting it as a percentage next to the model's
own percentage makes it a comparison of forecasts. No dollar figures, no order
books, and nothing linking a reader to a trade.

**Chamber level only.** Per-state markets exist, but a state-by-state
scoreboard invites a defect: Nebraska's market prices the Democratic Party at
near zero while the model has Dan Osborn -- an independent counted on the
Democratic side for control arithmetic -- around two thirds. Both are right
about different questions, and displaying them side by side would show a
69-point discrepancy that is purely definitional.

**Failure is not an outage.** The forecast must publish whether or not this
works, so every entry point returns partial data rather than raising, and the
dashboard treats missing markets as a normal state. Polymarket is also
DNS-blocked in some jurisdictions -- France's regulator null-routes it -- which
is another reason the fetch happens once on the runner and readers get the
numbers from our own origin.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
TIMEOUT_SECONDS = 20

SOURCE_NAME = "Polymarket"
SOURCE_URL = "https://polymarket.com"


@dataclass
class MarketOutcome:
    """One outcome of one event, with the market's implied probability.

    Mutable on purpose: :func:`attach_history` fills the series in place after
    the outcomes are built, because the history endpoint is keyed on a token id
    that is only known once the market has been parsed.
    """

    label: str
    probability: float
    #: CLOB token for the "Yes" side, which is what the price-history endpoint
    #: is keyed on. Kept so a snapshot can be turned into a time series later;
    #: not shown to anyone.
    token_id: str = ""
    #: Daily implied probability over time, oldest first, as (ISO date, p).
    #: Empty when history has not been fetched or the endpoint had none.
    history: list = field(default_factory=list)


@dataclass
class MarketEvent:
    """A group of mutually exclusive outcomes traders are pricing."""

    slug: str
    title: str
    #: Total traded, used only to tell a reader how seriously to take it. A
    #: $20k market and a $10M market deserve different weight.
    volume: float
    outcomes: list[MarketOutcome] = field(default_factory=list)

    def probability_of(self, label: str) -> float | None:
        for outcome in self.outcomes:
            if outcome.label.strip().lower() == label.strip().lower():
                return outcome.probability
        return None


#: The events worth showing, and the shape each is expected to have.
#:
#: Slugs rather than a search: Gamma silently ignores an unrecognised query
#: parameter, so `search=senate` returns the whole site's top markets looking
#: exactly like a filtered result. Asking by slug either finds the event or
#: does not.
WANTED = (
    ("balance-of-power-2026-midterms", "Control of Congress"),
    ("which-party-will-win-the-senate-in-2026", "Senate control"),
    ("republican-senate-seats-after-the-2026-midterm-elections-927", "Senate seats"),
)


def _get(url: str) -> object | None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "midterms-forecast (github.com/whyalwaysrose)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        log.warning("markets: %s -> %s", url, exc)
        return None


def _maybe_json(value: object) -> object:
    """Gamma returns some list fields as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _outcomes(event: dict) -> list[MarketOutcome]:
    """Every "Yes" leg of an event, as an implied probability.

    A grouped event carries one binary market per outcome, each with a Yes and a
    No leg. The Yes price is that outcome's probability; the No leg is its
    complement and would double-count.
    """
    found: list[MarketOutcome] = []
    for market in event.get("markets") or []:
        label = str(market.get("groupItemTitle") or market.get("question") or "").strip()
        names = _maybe_json(market.get("outcomes"))
        prices = _maybe_json(market.get("outcomePrices"))
        if not (isinstance(names, list) and isinstance(prices, list) and label):
            continue
        for name, price in zip(names, prices, strict=False):
            if str(name).strip().lower() != "yes":
                continue
            try:
                probability = float(price)
            except (TypeError, ValueError):
                continue
            if 0.0 <= probability <= 1.0:
                # clobTokenIds is [yes, no] in the same order as outcomes.
                tokens = _maybe_json(market.get("clobTokenIds"))
                token = ""
                if isinstance(tokens, list) and tokens:
                    token = str(tokens[0])
                found.append(
                    MarketOutcome(label=label, probability=probability, token_id=token)
                )
    found.sort(key=lambda o: -o.probability)
    return found


def fetch() -> dict[str, MarketEvent]:
    """The chamber-level events, keyed by slug. Missing ones are simply absent."""
    events: dict[str, MarketEvent] = {}
    for slug, _ in WANTED:
        payload = _get(f"{GAMMA}/events?slug={urllib.parse.quote(slug)}")
        if not isinstance(payload, list) or not payload:
            log.warning("markets: no event for %s", slug)
            continue
        raw = payload[0]
        try:
            volume = float(raw.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        outcomes = _outcomes(raw)
        if not outcomes:
            log.warning("markets: %s returned no priced outcomes", slug)
            continue
        events[slug] = MarketEvent(
            slug=slug,
            title=str(raw.get("title") or slug),
            volume=volume,
            outcomes=outcomes,
        )
        log.info("markets: %s -> %d outcomes, $%.0f volume", slug, len(outcomes), volume)
    return events


# --------------------------------------------------------------- snapshots


def snapshot_path(run_date: date | None = None) -> Path:
    from ..paths import RAW_DIR

    stamp = (run_date or date.today()).isoformat()
    return RAW_DIR / f"markets-{stamp}.json"


def write_snapshot(events: dict[str, MarketEvent], run_date: date | None = None) -> Path:
    """Persist a fetch, so the site keeps working when the API does not.

    Also the only way this can be developed against on a machine where
    Polymarket is blocked.
    """
    path = snapshot_path(run_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "events": {slug: asdict(event) for slug, event in events.items()},
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="")
    log.info("markets: wrote %s (%d events)", path, len(events))
    return path


def latest_snapshot() -> Path | None:
    from ..paths import RAW_DIR

    if not RAW_DIR.is_dir():
        return None
    found = sorted(RAW_DIR.glob("markets-*.json"))
    return found[-1] if found else None


def load_snapshot(path: Path) -> tuple[dict[str, MarketEvent], str]:
    """Read a snapshot back. Returns the events and when they were fetched."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = {
        slug: MarketEvent(
            slug=body["slug"],
            title=body["title"],
            volume=body["volume"],
            outcomes=[
                MarketOutcome(
                    label=o["label"],
                    probability=o["probability"],
                    token_id=o.get("token_id", ""),
                    # Tuples survive a JSON round trip as lists; the chart does
                    # not care, but comparing snapshots would.
                    history=[tuple(p) for p in o.get("history") or []],
                )
                for o in body["outcomes"]
            ],
        )
        for slug, body in (raw.get("events") or {}).items()
    }
    return events, str(raw.get("fetched_at", ""))


# ------------------------------------------------------------ price history

CLOB = "https://clob.polymarket.com"

#: Daily points. The endpoint takes fidelity in minutes; 1440 is one a day,
#: which is all a chart spanning months can show and keeps the payload small.
HISTORY_FIDELITY_MINUTES = 1440


def fetch_history(token_id: str) -> list[tuple[str, float]]:
    """Daily implied probability for one outcome, oldest first.

    Returns an empty list rather than raising on any failure. A market with no
    trading history is normal, not an error, and the dashboard has to render
    either way.
    """
    if not token_id:
        return []
    payload = _get(
        f"{CLOB}/prices-history?market={urllib.parse.quote(token_id)}"
        f"&interval=max&fidelity={HISTORY_FIDELITY_MINUTES}"
    )
    if not isinstance(payload, dict):
        return []

    points: list[tuple[str, float]] = []
    for point in payload.get("history") or []:
        try:
            when = datetime.fromtimestamp(int(point["t"]), tz=UTC).date()
            price = float(point["p"])
        except (KeyError, TypeError, ValueError, OSError, OverflowError):
            continue
        if 0.0 <= price <= 1.0:
            points.append((when.isoformat(), round(price, 4)))

    # One point per day, keeping the last reading of each -- the endpoint can
    # return several inside a day and a chart only needs the close.
    by_day: dict[str, float] = {}
    for when, price in points:
        by_day[when] = price
    return sorted(by_day.items())


#: Events whose card is a time series. The seat-count market is deliberately
#: absent: it is a distribution over a single election, so eleven lines of
#: history would be spaghetti where one bar chart is legible.
CHARTED_OVER_TIME = frozenset({
    "balance-of-power-2026-midterms",
    "which-party-will-win-the-senate-in-2026",
})

#: Most points to keep per series. Thirteen months of daily readings is 400
#: points, and the snapshot is committed on every run -- at 200 KB a day that is
#: 14 MB by election day, for detail no chart a few hundred pixels wide can
#: show. 120 keeps roughly three-day resolution over the whole period.
MAX_HISTORY_POINTS = 120


def thin(series: list, limit: int = MAX_HISTORY_POINTS) -> list:
    """Evenly sample a series down to ``limit`` points, always keeping the last.

    The final point is today's price, which the card prints as a number, so it
    must survive: dropping it would leave the line ending somewhere other than
    the figure beside it.
    """
    if len(series) <= limit:
        return list(series)
    step = (len(series) - 1) / (limit - 1)
    picked = {round(i * step) for i in range(limit)}
    picked.add(len(series) - 1)
    return [series[i] for i in sorted(picked)]


def attach_history(events: dict[str, MarketEvent], limit: int = 4) -> None:
    """Fill in the time series for outcomes that will be charted, in place.

    ``limit`` caps how many outcomes per event get a request. Each one is a
    separate call, and an outcome priced near zero contributes a flat line along
    the axis -- ink without information.
    """
    for slug, event in events.items():
        if slug not in CHARTED_OVER_TIME:
            log.info("markets: %s is charted as a distribution, skipping history", slug)
            continue
        wanted = sorted(event.outcomes, key=lambda o: -o.probability)[:limit]
        filled = points = 0
        for outcome in wanted:
            series = thin(fetch_history(outcome.token_id))
            if series:
                outcome.history[:] = series
                filled += 1
                points += len(series)
        log.info(
            "markets: %s -> history for %d of %d outcomes (%d points)",
            slug, filled, len(event.outcomes), points,
        )
