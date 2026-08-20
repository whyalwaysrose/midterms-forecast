"""Static guards on the front end.

These are deliberately cheap greps, not a browser harness. They pin the two
failure modes that are invisible in Python tests but break the page for every
visitor, both of which actually happened during development.
"""

from __future__ import annotations

import re

from midterms import outputs, paths

SITE = paths.SITE_DIR
CSS = (SITE / "css" / "style.css").read_text(encoding="utf-8")
JS = (SITE / "js" / "app.js").read_text(encoding="utf-8")
HTML = (SITE / "index.html").read_text(encoding="utf-8")


def test_hidden_attribute_cannot_be_overridden_by_a_class_rule():
    """Regression: `.drawer { display: flex }` beat the UA `[hidden]` rule.

    The race drawer stayed painted as a fixed, full-viewport, z-index 50 overlay
    even while its `hidden` attribute was set, so the entire dashboard was
    unclickable. A class selector is author-origin and silently outranks the
    browser's own `[hidden] { display: none }`, so the page must assert it back.
    """
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", CSS), (
        "style.css must contain `[hidden] { display: none !important; }` — "
        "without it any class rule that sets `display` will keep a hidden "
        "element on screen"
    )


def test_every_element_using_hidden_is_covered_by_that_rule():
    """The guard only works if it is global, not scoped to one class."""
    hidden_elements = re.findall(r"<[^>]*\shidden\s*>", HTML)
    assert hidden_elements, "expected the page to use the hidden attribute"
    # A scoped variant such as `.drawer[hidden]` would leave the others exposed.
    assert re.search(r"^\[hidden\]", CSS, re.M), "the [hidden] rule must be unscoped"


def test_charts_are_measured_only_while_visible():
    """Regression: charts sized themselves against a container of width 0.

    `chartWidth()` falls back to an unscaled default when `clientWidth` is 0,
    which happens whenever the container is still `hidden`. Both reveals must
    therefore happen before the corresponding render call.
    """
    def call_site(pattern: str) -> int:
        """Offset of a call, ignoring the function's own definition."""
        match = re.search(pattern, JS, re.M)
        assert match, f"no call site matching {pattern!r}"
        return match.start()

    main_reveal = JS.index("$('main').hidden = false")
    first_chart = call_site(r"^\s+renderSeatChart\(forecast\);")
    assert main_reveal < first_chart, (
        "main must be revealed before renderSeatChart, or every chart on the "
        "page renders at the fallback width and its axis text is downscaled"
    )

    drawer_reveal = JS.index("$('drawer').hidden = false")
    drawer_chart = call_site(r"^\s+renderTrajectory\(\$\('drawer-chart'\)")
    assert drawer_reveal < drawer_chart, (
        "the drawer must be revealed before its trajectory chart is drawn"
    )


def test_schema_version_matches_between_python_and_javascript():
    """The page refuses to render mismatched data, so keep the two in step."""
    match = re.search(r"const SCHEMA_VERSION\s*=\s*(\d+)", JS)
    assert match, "app.js must declare SCHEMA_VERSION"
    assert int(match.group(1)) == outputs.SCHEMA_VERSION


def test_wide_tables_scroll_inside_their_own_container():
    """A table must never make the page body scroll sideways."""
    assert ".table-scroll" in CSS and "overflow-x: auto" in CSS
    assert 'class="table-scroll"' in JS
