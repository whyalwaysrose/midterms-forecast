"""What does Polymarket actually offer for the 2026 Senate, and can we reach it?

Run this on a GitHub runner, not locally. Polymarket is DNS-blocked on some
connections -- France's gambling regulator null-routes every polymarket.com
domain to localhost and serves an anj.fr block page -- so a developer machine
may be unable to see the API at all while the production path is fine. The
runner is US-hosted and reaches it normally; nothing here circumvents anything,
it is simply the machine that would do the fetching in production.

The first version passed `search=senate` and got back the highest-volume events
on the whole site -- the 2028 presidential nomination, the next Prime Minister
of Ethiopia, whether Jesus Christ returns before 2027. Gamma ignores an
unrecognised query parameter rather than erroring, so an unfiltered list came
back looking like a filtered one. This version asks by slug, which is exact, and
enumerates by tag, which is bounded, instead of trusting a search field.

Usage (on a runner):
    python scripts/probe_polymarket.py
"""

from __future__ import annotations

import json
import sys
import re
import urllib.error
import urllib.parse
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"
TIMEOUT = 25

#: Event slugs known from public reporting. Exact lookups, so no guessing.
KNOWN_SLUGS = (
    "which-party-will-win-the-senate-in-2026",
    "balance-of-power-2026-midterms",
    "will-democrats-win-all-core-four-senate-races",
)


def fetch(path: str) -> tuple[int, object]:
    url = f"{GAMMA}{path}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "midterms-forecast/probe"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body[:400]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:300]
    except OSError as exc:
        return 0, f"unreachable: {exc}"


def maybe_json(value):
    """Gamma returns some list fields as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def show_market(market: dict, indent: str = "      ") -> None:
    outcomes = maybe_json(market.get("outcomes"))
    prices = maybe_json(market.get("outcomePrices"))
    pairs = ""
    if isinstance(outcomes, list) and isinstance(prices, list):
        pairs = "   ".join(
            f"{o} {float(p) * 100:.1f}%"
            for o, p in zip(outcomes, prices, strict=False)
        )
    try:
        volume = f"${float(market.get('volumeNum') or market.get('volume') or 0):,.0f}"
    except (TypeError, ValueError):
        volume = "?"
    print(f"{indent}{str(market.get('question', '?'))[:58]:58s} {pairs:26s} vol {volume}")


def main() -> int:
    print("=" * 78)
    print("  1. Reachability and response shape")
    print("=" * 78)
    status, payload = fetch("/markets?limit=1")
    print(f"  GET /markets?limit=1 -> {status}")
    if status != 200:
        print(f"  {payload}\n  Cannot reach Polymarket from here either. Stop.")
        return 1

    print()
    print("=" * 78)
    print("  2. Known events, looked up by exact slug")
    print("=" * 78)
    for slug in KNOWN_SLUGS:
        status, payload = fetch(f"/events?slug={urllib.parse.quote(slug)}")
        if status != 200 or not isinstance(payload, list) or not payload:
            print(f"\n  {slug} -> {status} (no event)")
            continue
        event = payload[0]
        markets = event.get("markets") or []
        try:
            volume = f"${float(event.get('volume') or 0):,.0f}"
        except (TypeError, ValueError):
            volume = "?"
        print(f"\n  {event.get('title')}")
        print(f"    slug {slug}   volume {volume}   {len(markets)} markets")
        for market in markets[:8]:
            show_market(market)

    print()
    print("=" * 78)
    print("  3. Enumerating by tag, to find per-state Senate markets")
    print("=" * 78)
    found: dict[str, dict] = {}
    for tag in ("elections", "politics", "midterms", "us-politics"):
        for offset in (0, 100, 200, 300):
            status, payload = fetch(
                f"/events?closed=false&limit=100&offset={offset}"
                f"&tag_slug={tag}&order=volume&ascending=false"
            )
            if status != 200 or not isinstance(payload, list) or not payload:
                break
            for event in payload:
                found[str(event.get("id"))] = event
    print(f"  {len(found)} open events across those tags")

    senate = {
        i: e for i, e in found.items()
        if "senate" in str(e.get("title", "")).lower()
        or "senate" in str(e.get("slug", "")).lower()
    }
    print(f"  {len(senate)} mention the Senate\n")
    for event in sorted(senate.values(), key=lambda e: -float(e.get("volume") or 0)):
        try:
            volume = f"${float(event.get('volume') or 0):,.0f}"
        except (TypeError, ValueError):
            volume = "?"
        print(f"  {volume:>14}  {str(event.get('title'))[:62]:62s} {event.get('slug')}")

    print()
    print("=" * 78)
    print("  4. Per-state coverage of the 2026 map")
    print("=" * 78)
    states = [
        "Georgia", "Michigan", "North Carolina", "New Hampshire", "Maine",
        "Ohio", "Texas", "Iowa", "Alaska", "Minnesota", "Nebraska", "Virginia",
        "Kentucky", "Louisiana", "Kansas", "Colorado", "Illinois", "New Mexico",
    ]
    covered = 0
    for state in states:
        hits = [
            e for e in senate.values()
            if state.lower() in str(e.get("title", "")).lower()
            or state.lower().replace(" ", "-") in str(e.get("slug", "")).lower()
        ]
        if hits:
            covered += 1
            print(f"    {state:16s} {hits[0].get('slug')}")
        else:
            print(f"    {state:16s} -")
    print(f"\n  {covered} of {len(states)} checked states have a Senate market.")
    # Emit the prices as JSON so they can be diffed against the model offline.
    # The model says 71% for Democratic control; the market says 49.5%. A gap
    # that size is either a real disagreement worth understanding or a defect,
    # and the way to tell is whether it is spread across every race or
    # concentrated in a few.
    print()
    print("=" * 78)
    print("  5. Machine-readable dump (Democratic win probability per race)")
    print("=" * 78)

    dump: dict[str, object] = {}
    for event in senate.values():
        slug = str(event.get("slug", ""))
        match = re.fullmatch(r"([a-z-]+)-senate-election-winner", slug)
        if not match:
            continue
        state = match.group(1).replace("-", " ").title()
        best = None
        for market in event.get("markets") or []:
            label = str(market.get("groupItemTitle") or market.get("question") or "")
            outcomes = maybe_json(market.get("outcomes"))
            prices = maybe_json(market.get("outcomePrices"))
            if not (isinstance(outcomes, list) and isinstance(prices, list)):
                continue
            try:
                volume = float(market.get("volumeNum") or market.get("volume") or 0)
            except (TypeError, ValueError):
                volume = 0.0
            for outcome, price in zip(outcomes, prices, strict=False):
                name = str(outcome).strip().lower()
                is_dem = name.startswith("democrat") or (
                    name == "yes" and "democrat" in label.lower()
                )
                if is_dem and (best is None or volume > best[1]):
                    best = (float(price), volume, label)
        if best:
            dump[state] = {
                "dem": round(best[0], 4),
                "volume": round(best[1]),
                "market": best[2][:48],
            }

    print(json.dumps(dump, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
