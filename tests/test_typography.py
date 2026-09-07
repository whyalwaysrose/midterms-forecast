"""Type, contrast, measure, and the sentences at the top of each chart.

All four are presentation, and all four fail silently. Nothing throws when a
colour drops below the readable threshold in one theme, when a paragraph runs to
175 characters a line, or when a headline rounds a tied race into a lead. The
page keeps working and simply stops communicating.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "site" / "css" / "style.css").read_text(encoding="utf-8")
JS = (ROOT / "site" / "js" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "site" / "index.html").read_text(encoding="utf-8")


def test_type_scale_and_contrast():
    """Runs the real checker, so its thresholds cannot drift from the CSS."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_typography.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- measure ----------------------------------------------------------------


def test_running_text_is_capped_and_data_is_not():
    """Prose gets a measure; charts, maps and tables keep the full width.

    Butterick puts comfortable line length at 45-90 characters. Measured on the
    live page before this cap, 99 of 146 text blocks were over 90 and the worst
    ran to 175 -- and it was the methodology prose, the longest reading on the
    page, that suffered most.

    The distinction is what gets read line after line. Narrowing a table or an
    axis would cost information for nothing.
    """
    assert re.search(r"--measure:\s*\d+ch", CSS), "the measure token must be declared"

    capped = re.search(r"([^}]+)\{\s*max-width:\s*var\(--measure\)", CSS)
    assert capped, "no rule applies --measure"
    selectors = capped.group(1)
    for prose in (".card-sub", ".commentary p", ".method-panel p", ".primer-item p"):
        assert prose in selectors, f"{prose} is running text and should be capped"

    # The negative lookahead matters: `.chart` is a prefix of `.chart-note`,
    # which is a caption and *should* be capped. Without it this asserts the
    # opposite of what it means on any selector that shares a prefix.
    for data in (".race-list", ".chart", "table.polls", ".us-map"):
        assert not re.search(
            rf"{re.escape(data)}(?![\w-])[^{{]*\{{[^}}]*max-width:\s*var\(--measure\)", CSS
        ), f"{data} is scanned, not read -- it should keep the full width"


# --- chart headlines --------------------------------------------------------
#
# Cheap static guards rather than a browser harness, matching the rest of the
# front-end tests: there is no JS runtime here or in CI. Each one pins a way the
# headline could state something the forecast does not support.


def test_every_chart_headline_is_written_from_the_data():
    """A heading left as its placeholder would ship an em dash to readers."""
    for element_id in (
        "seat-chart-title", "history-title", "national-title",
        "senate-map-title", "house-map-title",
    ):
        assert f'id="{element_id}"' in HTML, f"#{element_id} is missing from the page"
        assert f"$('{element_id}').textContent" in JS, (
            f"#{element_id} is never written from the forecast, so it would keep "
            f"the placeholder the HTML ships with"
        )


def test_a_close_generic_ballot_is_not_reported_as_a_lead():
    """The interval decides the verb, not the point estimate.

    A median of D+0.3 whose 90% interval spans zero is not a lead, and a
    headline saying "Democrats lead the generic ballot" would be claiming one.
    """
    body = re.search(r"function nationalHeadline\((.*?)\n}", JS, re.S)
    assert body, "nationalHeadline is missing"
    assert "dem_margin_p05" in body.group(1) and "dem_margin_p95" in body.group(1), (
        "nationalHeadline must consult the interval, not just the median, "
        "before it says either party leads"
    )


def test_a_flat_forecast_is_allowed_to_say_nothing_happened():
    """Most days this model does not move, and the headline must be able to
    say so. Manufacturing a direction out of a few tenths of a point would
    misrepresent the model's stability as news."""
    body = re.search(r"function historyHeadline\((.*?)\n}", JS, re.S)
    assert body, "historyHeadline is missing"
    assert "held steady" in body.group(1), (
        "historyHeadline has no flat branch, so it will always claim a direction"
    )


def test_the_chart_keeps_its_name_as_a_kicker():
    """This is a dashboard, not an article.

    Replacing "Distribution of Senate seats" with a finding and nothing else
    would leave a reader scanning for a particular panel with no label to scan
    for. The name moves to the kicker; it does not disappear.
    """
    assert ".chart-kicker" in CSS, "the kicker has no styling"
    assert HTML.count('class="chart-kicker"') >= 5, (
        "every data-driven heading needs the chart's name above it"
    )
