"""Can we display Polymarket's implied probabilities, and what are they?

Run this on a GitHub runner. Polymarket is DNS-blocked on some connections --
France's gambling regulator null-routes every polymarket.com domain and serves
an anj.fr block page -- so a developer machine may be unable to see the API or
the terms at all while the production path is fine. The runner is US-hosted and
reaches both normally.

Two jobs:

1. **Settle the licensing question**, which gates everything else. This project
   rejected the NYT and Silver Bulletin as sources purely on licensing, so the
   same standard applies here, and the terms cannot be read from the dev
   machine. Fetches the actual documents rather than trusting a search summary.

2. **Dump the chamber-level markets** we would display: Senate control, the
   Balance of Power sweep combinations, and the Republican-seat-count
   distribution. Not per-state, and not prices -- what a reader would see is an
   implied probability, which is what a prediction-market price already is.

Usage (on a runner):
    python scripts/probe_polymarket.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"
TIMEOUT = 25

#: The chamber-level events worth showing. Per-state markets exist but are not
#: what this is for.
EVENT_SLUGS = (
    "which-party-will-win-the-senate-in-2026",
    "balance-of-power-2026-midterms",
    "will-democrats-win-all-core-four-senate-races",
    "republican-senate-seats-after-the-2026-midterm-elections-927",
)

#: Documents that might carry data-reuse terms.
TERMS_URLS = (
    "https://polymarket.com/tos",
    "https://polymarket.com/terms-of-service",
    "https://docs.polymarket.com/",
    "https://docs.polymarket.com/developers/gamma-markets-api/overview",
)

#: Phrases worth surfacing verbatim from whatever the terms say.
TERMS_PATTERNS = (
    r"[^.]*redistribut[^.]*\.",
    r"[^.]*scrap[^.]*\.",
    r"[^.]*\bAPI\b[^.]*\.",
    r"[^.]*commercial[^.]*\.",
    r"[^.]*attribut[^.]*\.",
)


def get(url: str) -> tuple[int, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; midterms-forecast)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except OSError as exc:
        return 0, f"unreachable: {exc}"


def maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|svg)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#x27;", "'")
    return re.sub(r"\s+", " ", text).strip()


def terms() -> None:
    print("=" * 78)
    print("  1. What do the terms say about reusing the data?")
    print("=" * 78)
    for url in TERMS_URLS:
        status, body = get(url)
        text = strip_html(body) if status == 200 else body
        print(f"\n  [{status}] {url}  ({len(text)} chars of text)")
        if status != 200 or len(text) < 200:
            continue
        seen: set[str] = set()
        for pattern in TERMS_PATTERNS:
            for match in re.findall(pattern, text, re.IGNORECASE):
                sentence = match.strip()
                if 40 < len(sentence) < 400 and sentence not in seen:
                    seen.add(sentence)
                    print(f"      - {sentence}")
        if not seen:
            print("      (nothing mentioning reuse, redistribution or the API)")


def markets() -> None:
    print()
    print("=" * 78)
    print("  2. Chamber-level implied probabilities")
    print("=" * 78)
    dump: dict[str, dict] = {}
    for slug in EVENT_SLUGS:
        status, body = get(f"{GAMMA}/events?slug={urllib.parse.quote(slug)}")
        if status != 200:
            print(f"\n  {slug} -> {status}")
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            print(f"\n  {slug} -> not JSON")
            continue
        if not payload:
            print(f"\n  {slug} -> no event")
            continue

        event = payload[0]
        try:
            volume = float(event.get("volume") or 0)
        except (TypeError, ValueError):
            volume = 0.0
        print(f"\n  {event.get('title')}   (volume ${volume:,.0f})")

        outcomes: dict[str, float] = {}
        for market in event.get("markets") or []:
            label = str(market.get("groupItemTitle") or market.get("question") or "?")
            names = maybe_json(market.get("outcomes"))
            prices = maybe_json(market.get("outcomePrices"))
            if not (isinstance(names, list) and isinstance(prices, list)):
                continue
            for name, price in zip(names, prices, strict=False):
                if str(name).strip().lower() == "yes":
                    try:
                        outcomes[label] = float(price)
                    except (TypeError, ValueError):
                        pass
        for label, probability in sorted(outcomes.items(), key=lambda kv: -kv[1]):
            if probability > 0.0005:
                print(f"      {label[:58]:58s} {probability * 100:5.1f}%")
        dump[slug] = {
            "title": event.get("title"),
            "volume": round(volume),
            "outcomes": {k: round(v, 4) for k, v in outcomes.items()},
        }

    print()
    print("=" * 78)
    print("  3. Machine-readable")
    print("=" * 78)
    print(json.dumps(dump, indent=1, sort_keys=True))


def main() -> int:
    status, _ = get(f"{GAMMA}/markets?limit=1")
    print(f"reachability: GET /markets -> {status}\n")
    if status != 200:
        print("Cannot reach Polymarket from here. Stop.")
        return 1
    terms()
    markets()
    return 0


if __name__ == "__main__":
    sys.exit(main())
