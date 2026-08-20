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
MAP_JS = (SITE / "js" / "map.js").read_text(encoding="utf-8")
CHARTS_JS = (SITE / "js" / "charts.js").read_text(encoding="utf-8")
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


def test_map_script_is_loaded_after_app():
    """map.js calls helpers defined in app.js, so order is load-bearing."""
    app_tag = HTML.index('src="js/app.js"')
    map_tag = HTML.index('src="js/map.js"')
    assert app_tag < map_tag


def test_map_focus_and_key_handlers_are_bound_per_element():
    """Regression: delegation does not work for SVG children.

    Focus and keydown events raised on an SVG <path> do not reliably bubble to
    an HTML ancestor. A delegated listener on the map container never fired,
    even though document.activeElement was correctly the focused path, which
    silently removed keyboard access to all 35 races.
    """
    assert "target.addEventListener('focus'" in MAP_JS
    assert "target.addEventListener('keydown'" in MAP_JS
    assert "host.addEventListener('focusin'" not in MAP_JS
    assert "host.addEventListener('keydown'" not in MAP_JS


def test_tossup_carries_a_texture_not_only_a_colour():
    """It sits between the two hues, so it is the step most easily misread."""
    assert "hatch-tossup" in MAP_JS
    assert "hatch-tossup" in CSS


def test_rating_scale_has_five_steps():
    """Seven was measured and rejected: adjacent pairs fell under the
    perceptual separation floor near the neutral midpoint."""
    steps = re.findall(r"key: '([a-z-]+)',\s*label:", MAP_JS)
    assert steps == ["safe-r", "lean-r", "tossup", "lean-d", "safe-d"]


def test_every_rating_step_is_defined_in_all_three_theme_blocks():
    """A token defined only under one theme renders as nothing in the others."""
    for step in ("safe-r", "lean-r", "tossup", "lean-d", "safe-d", "no-race"):
        assert CSS.count(f"--{step}:") == 3, f"--{step} must be defined 3 times"


def test_map_furniture_degrades_with_available_width():
    """Inline labels scale with the map, so they must drop out before they
    become unreadable; chips are fixed-size and survive further down."""
    assert "const showInlineLabels = width >= 820" in MAP_JS
    assert "const showChips = width >= 640" in MAP_JS
    assert 'id="map-hint"' in HTML


def _js_function_body(source: str, name: str) -> str:
    """Source of one top-level JS function, up to its closing brace."""
    start = source.index(f"function {name}")
    end = source.index(chr(10) + "}" + chr(10), start)
    return source[start:end]


def test_scripts_load_in_dependency_order():
    """charts.js and map.js both call helpers defined in app.js."""
    order = [HTML.index(f'src="js/{name}.js"') for name in ("app", "charts", "map")]
    assert order == sorted(order)


def test_history_axis_is_scaled_to_the_data():
    """Regression: the forecast-over-time chart was pinned to 0-100%.

    Every run has sat inside a ten-point band, so a fixed axis rendered the line
    as a flat squiggle using a tenth of the panel. It now scales to the data —
    but must always keep 50% in the domain, since crossing it is the one change
    that alters the story.
    """
    assert "paddedDomain(probs" in CHARTS_JS
    assert "include: 0.5" in CHARTS_JS
    # The old fixed-domain scale must be gone.
    assert "const p = i / 4;" not in CHARTS_JS


def test_axis_ticks_land_on_round_numbers():
    assert "function niceTicks" in CHARTS_JS
    for chart in ("renderSeatChart", "renderHistoryChart", "renderTrajectory"):
        body = _js_function_body(CHARTS_JS, chart)
        assert "niceTicks(" in body, f"{chart} must use niceTicks for its axis"


def test_every_chart_has_a_hover_layer():
    """An SVG chart is interactive by default; each one ships a tooltip."""
    for chart in ("renderSeatChart", "renderHistoryChart", "renderTrajectory"):
        body = _js_function_body(CHARTS_JS, chart)
        assert "chartTip(host)" in body, f"{chart} needs a tooltip"
        assert "pointermove" in body, f"{chart} needs a hover handler"
        assert "pointerleave" in body, f"{chart} must clear its hover state"


def test_hover_dimming_is_gated_to_hover_devices():
    """:hover latches after a tap on touch, which would strand the chart dimmed."""
    assert "@media (hover: hover)" in CSS


def test_chart_host_anchors_its_tooltip():
    """The tooltip is absolutely positioned inside the chart host."""
    assert re.search(r"\.chart\s*\{[^}]*position:\s*relative", CSS)
