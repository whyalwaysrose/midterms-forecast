"""The last gate before a deploy, tested against assembled directories.

`verify_site.py` is the only thing standing between a broken build and a page
served to whoever opens the link. It had no tests of its own, which is an odd
place to leave untested: every other check in this suite protects something
that can be fixed after the fact.

The House checks in particular need pinning, because they have to fail in one
direction and pass in the other. A deploy carrying only the Senate is valid --
the House run is allowed to fail without blocking it -- so refusing that would
take the whole dashboard down to protect a tab. A deploy carrying half a House
is not valid, and letting that through renders a working-looking control over
an empty map.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / ".github" / "scripts" / "verify_site.py"
REAL_LAYOUT = REPO / "site" / "data" / "us-districts.json"


def run(site: Path) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(VERIFIER), str(site)],
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A minimal artifact that passes: stamped assets, Senate data, no House."""
    root = tmp_path / "site"
    (root / "data").mkdir(parents=True)
    (root / "js").mkdir()
    (root / "css").mkdir()

    (root / "index.html").write_text(
        '<link rel="stylesheet" href="css/style.css?v=abc">'
        '<script src="js/app.js?v=abc"></script>',
        encoding="utf-8",
    )
    (root / "css" / "style.css").write_text("/* */", encoding="utf-8")
    (root / "js" / "app.js").write_text("const SCHEMA_VERSION = 5;", encoding="utf-8")
    for name in ("history.json", "commentary.json", "us-states.json"):
        (root / "data" / name).write_text("{}", encoding="utf-8")
    (root / "data" / "forecast.json").write_text(
        json.dumps({
            "schema_version": 5,
            "races": [{"unit": "GA"}],
            "chamber_forecast": {"dem_control_prob": 0.5},
        }),
        encoding="utf-8",
    )
    return root


def all_districts() -> list[str]:
    layout = json.loads(REAL_LAYOUT.read_text(encoding="utf-8"))
    return [tile["district"] for tile in layout["tiles"]]


def add_house(site: Path, units: list[str], schema: int = 5) -> None:
    (site / "data" / "forecast_house.json").write_text(
        json.dumps({
            "schema_version": schema,
            "races": [{"unit": u} for u in units],
            "chamber_forecast": {"dem_control_prob": 0.5},
        }),
        encoding="utf-8",
    )
    shutil.copy(REAL_LAYOUT, site / "data" / "us-districts.json")


def test_a_senate_only_deploy_is_valid(site):
    """The House is optional, and refusing without it would be the worse bug.

    A House run that overruns its hour is expected and survivable. Blocking the
    deploy over it would take down the Senate forecast too, which is exactly
    the coupling the workflow's continue-on-error exists to avoid.
    """
    code, out = run(site)
    assert code == 0, out


def test_a_complete_house_is_valid(site):
    add_house(site, all_districts())
    code, out = run(site)
    assert code == 0, out


def test_a_district_with_no_square_is_refused(site):
    """The one failure this picture could hide.

    A district absent from the cartogram does not error -- it simply is not
    drawn, while still counting toward the 218 the headline reports. The map
    and the arithmetic would disagree with nothing to say so.
    """
    add_house(site, [*all_districts(), "ZZ-99"])
    code, out = run(site)
    assert code == 1
    assert "no square on the cartogram" in out


def test_a_layout_with_no_forecast_is_refused(site):
    """This is not hypothetical: it is the tree that was committed once.

    The page hides the chamber switcher unless both files load, so half a House
    renders a control that looks live over a blank map -- and the missing half
    looks identical to a deploy that simply had not run the House yet.
    """
    shutil.copy(REAL_LAYOUT, site / "data" / "us-districts.json")
    code, out = run(site)
    assert code == 1
    assert "us-districts.json is present but forecast_house.json is not" in out


def test_a_forecast_with_no_layout_is_refused(site):
    add_house(site, all_districts())
    (site / "data" / "us-districts.json").unlink()
    code, out = run(site)
    assert code == 1
    assert "forecast_house.json is present but us-districts.json is not" in out


def test_the_two_chambers_must_agree_on_the_schema(site):
    """Otherwise the page drops the House without saying anything.

    `loadHouse` returns early on a schema mismatch and leaves the switcher
    hidden, which is safe but silent -- the deploy would look successful and
    half the dashboard would be missing.
    """
    add_house(site, all_districts(), schema=4)
    code, out = run(site)
    assert code == 1
    assert "different" in out and "schema" in out


def test_unstamped_assets_are_refused(site):
    """Pre-existing behaviour, pinned while we are here."""
    (site / "index.html").write_text(
        '<script src="js/app.js"></script>', encoding="utf-8"
    )
    code, out = run(site)
    assert code == 1
    assert "unstamped" in out


def test_a_missing_forecast_does_not_crash_the_house_check(site):
    """The House block reads `forecast` for its schema comparison.

    If forecast.json is missing, that name is never bound in the earlier branch
    -- so an unguarded read would raise NameError and the verifier would exit
    through a traceback instead of reporting the real problem.
    """
    (site / "data" / "forecast.json").unlink()
    add_house(site, all_districts())
    code, out = run(site)
    assert code == 1
    assert "Traceback" not in out
    assert "forecast.json is missing" in out
