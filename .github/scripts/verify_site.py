"""Last check before the site goes live.

Deployment is the one step with no undo: a broken page is served to whoever
opens the link, and the previous version is gone. These are the failures that
would actually reach a reader, checked against the built artifact rather than
the source tree.

Deliberately not a substitute for the test suite. Everything here is a property
of the *assembled directory* that unit tests cannot see: that every asset the
page references exists, that the data matches the schema the page expects, and
that the stamping step ran.

Usage:
    python .github/scripts/verify_site.py site
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ASSET_REF = re.compile(r'(?:src|href)="((?:js|css)/[^"]+)"')


def fail(message: str) -> None:
    print(f"  FAIL  {message}")


def main(argv: list[str]) -> int:
    site = Path(argv[1] if len(argv) > 1 else "site")
    problems: list[str] = []

    index = site / "index.html"
    if not index.is_file():
        fail(f"{index} does not exist")
        return 1
    html = index.read_text(encoding="utf-8")

    # --- every referenced asset resolves ---------------------------------
    referenced = ASSET_REF.findall(html)
    if not referenced:
        problems.append("index.html references no js/ or css/ assets at all")
    for ref in referenced:
        path, _, version = ref.partition("?v=")
        if not (site / path).is_file():
            problems.append(f"index.html references {path}, which is not in the artifact")
        elif not version:
            problems.append(
                f"{path} is unstamped; the stamp-assets step did not run, so a "
                "cached copy could be paired with fresh data"
            )

    # --- the page and the data agree on the schema -----------------------
    forecast_path = site / "data" / "forecast.json"
    forecast = None
    if not forecast_path.is_file():
        problems.append("site/data/forecast.json is missing; the page would show an error")
    else:
        try:
            forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"forecast.json is not valid JSON: {exc}")
            forecast = None

        app = site / "js" / "app.js"
        if forecast is not None and app.is_file():
            match = re.search(
                r"const SCHEMA_VERSION\s*=\s*(\d+)", app.read_text(encoding="utf-8")
            )
            if not match:
                problems.append("app.js does not declare SCHEMA_VERSION")
            elif int(match.group(1)) != forecast.get("schema_version"):
                problems.append(
                    f"app.js expects schema {match.group(1)} but forecast.json is "
                    f"{forecast.get('schema_version')} - every visitor would see an error"
                )

        if forecast is not None:
            races = forecast.get("races") or []
            if not races:
                problems.append("forecast.json contains no races")
            chamber = forecast.get("chamber_forecast") or {}
            probability = chamber.get("dem_control_prob")
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                problems.append(f"dem_control_prob is not a probability: {probability!r}")

    # --- supporting data the page loads unconditionally ------------------
    for name in ("history.json", "commentary.json", "us-states.json"):
        if not (site / "data" / name).is_file():
            problems.append(f"site/data/{name} is missing")

    # --- the House, if it is present at all ------------------------------
    #
    # Not required. The House was added part-way through the cycle and its run
    # is allowed to fail without blocking the Senate, so a deploy carrying only
    # the Senate is a valid deploy and must not be refused.
    #
    # But a *half-present* House is not: the page hides the chamber switcher
    # unless both the forecast and the layout load, and a forecast whose
    # districts have no squares to sit in would be a blank map behind a control
    # that looks like it works. If one is here, both must be.
    house = site / "data" / "forecast_house.json"
    layout = site / "data" / "us-districts.json"
    if house.is_file() != layout.is_file():
        present, absent = (
            ("forecast_house.json", "us-districts.json")
            if house.is_file()
            else ("us-districts.json", "forecast_house.json")
        )
        problems.append(
            f"site/data/{present} is present but {absent} is not; the House tab "
            "needs both, and would render an empty map"
        )
    elif house.is_file():
        try:
            payload = json.loads(house.read_text(encoding="utf-8"))
            tiles = json.loads(layout.read_text(encoding="utf-8")).get("tiles") or []
        except json.JSONDecodeError as exc:
            problems.append(f"House data is not valid JSON: {exc}")
        else:
            # Every district in the forecast must have a square, or it silently
            # vanishes from the map while still counting toward the majority --
            # the one failure this picture could hide from a reader.
            squares = {t.get("district") for t in tiles}
            missing = sorted(
                r["unit"] for r in payload.get("races", [])
                if r.get("unit") not in squares
            )
            if missing:
                problems.append(
                    f"{len(missing)} districts have no square on the cartogram "
                    f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''})"
                )
            if forecast and payload.get("schema_version") != forecast.get("schema_version"):
                problems.append(
                    "forecast_house.json and forecast.json declare different "
                    "schema versions; the page would silently drop the House"
                )

    if problems:
        print(f"Site verification FAILED ({len(problems)} problems):\n")
        for problem in problems:
            fail(problem)
        print("\nRefusing to deploy.")
        return 1

    print(
        f"Site verification passed: {len(referenced)} assets all present and "
        f"stamped, schema matches, data files complete."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
