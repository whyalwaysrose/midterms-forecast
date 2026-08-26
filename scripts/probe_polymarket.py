"""What does Polymarket actually offer for the 2026 Senate, and can we reach it?

Run this on a GitHub runner, not locally. Polymarket is DNS-blocked on some
connections -- France's gambling regulator null-routes every polymarket.com
domain to localhost and serves an anj.fr block page -- so a developer machine
may be unable to see the API at all while the production path is fine. The
runner is US-hosted and reaches it normally; nothing here circumvents anything,
it is simply the machine that will do the fetching in production.

Prints three things:

1. whether the API is reachable, and what the response shape is
2. which 2026 Senate markets exist, chamber-level and state-level, since that
   decides whether a per-state view has anything to put in it
3. whatever the API says about terms, so the licensing question can be settled
   from the source rather than from a search summary

Usage:
    python scripts/probe_polymarket.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"
TIMEOUT = 25


def fetch(url: str) -> tuple[int, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "midterms-forecast/probe"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8", "replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body[:500]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:300]
    except OSError as exc:
        return 0, f"unreachable: {exc}"


def brief(market: dict) -> str:
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            pass
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            pass
    pairs = ""
    if isinstance(outcomes, list) and isinstance(prices, list):
        pairs = "  ".join(
            f"{o}={float(p):.3f}" for o, p in zip(outcomes, prices, strict=False)
        )
    volume = market.get("volumeNum") or market.get("volume") or 0
    try:
        volume = f"${float(volume):,.0f}"
    except (TypeError, ValueError):
        volume = str(volume)
    return f"{market.get('question', '?')[:72]:72s} {pairs}  vol {volume}"


def main() -> int:
    print("=" * 78)
    print("  1. Reachability")
    print("=" * 78)
    status, payload = fetch(f"{GAMMA}/markets?limit=1")
    print(f"  GET /markets?limit=1 -> {status}")
    if status != 200:
        print(f"  {payload}")
        print("\n  The runner cannot reach Polymarket either. Stop here.")
        return 1
    if isinstance(payload, list) and payload:
        print(f"  response is a list; first market has {len(payload[0])} fields")
        print(f"  fields: {', '.join(sorted(payload[0])[:18])}")

    print()
    print("=" * 78)
    print("  2. What 2026 Senate markets exist?")
    print("=" * 78)

    seen: dict[str, dict] = {}
    for term in ("senate", "senate 2026", "midterms"):
        status, payload = fetch(
            f"{GAMMA}/events?closed=false&limit=60&order=volume"
            f"&ascending=false&search={urllib.parse.quote(term)}"
        )
        if status != 200 or not isinstance(payload, list):
            print(f"  search {term!r} -> {status}")
            continue
        for event in payload:
            seen[str(event.get("id"))] = event

    print(f"  {len(seen)} distinct open events matched\n")
    for event in sorted(
        seen.values(), key=lambda e: -float(e.get("volume") or 0)
    )[:25]:
        title = str(event.get("title", "?"))[:70]
        volume = float(event.get("volume") or 0)
        markets = event.get("markets") or []
        print(f"  ${volume:>12,.0f}  {title:70s}  ({len(markets)} markets)")

    print()
    print("=" * 78)
    print("  3. A chamber-level market in detail")
    print("=" * 78)
    for event in seen.values():
        title = str(event.get("title", "")).lower()
        if "senate" in title and ("which party" in title or "control" in title):
            print(f"  {event.get('title')}")
            print(f"  slug: {event.get('slug')}")
            for market in (event.get("markets") or [])[:6]:
                print(f"    {brief(market)}")
            break
    else:
        print("  no obvious chamber-control event in the search results")

    print()
    print("=" * 78)
    print("  4. Per-state coverage")
    print("=" * 78)
    states = [
        "Georgia", "Michigan", "North Carolina", "New Hampshire", "Maine",
        "Ohio", "Texas", "Iowa", "Alaska", "Minnesota", "Nevada", "Virginia",
    ]
    for state in states:
        hits = [
            e for e in seen.values()
            if state.lower() in str(e.get("title", "")).lower()
            and "senate" in str(e.get("title", "")).lower()
        ]
        mark = f"{len(hits)} market(s)" if hits else "-"
        print(f"    {state:16s} {mark}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
