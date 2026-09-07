"""Invariants of the candidate-share chart that cannot be seen by eye.

These are text assertions against charts.js because there is no JS runtime in
the suite. They are deliberately narrow: each one pins a specific bug that
shipped, so a rewrite that keeps the behaviour keeps the test passing and a
rewrite that reintroduces the bug does not.
"""

from __future__ import annotations

import re

from midterms import paths

CHARTS = (paths.SITE_DIR / "js" / "charts.js").read_text(encoding="utf-8")


def candidate_chart() -> str:
    """Source of renderCandidateChart, up to the next top-level function."""
    start = CHARTS.index("function renderCandidateChart")
    rest = CHARTS[start + 1 :]
    end = rest.find("\nfunction ")
    return rest if end == -1 else rest[:end]


def test_the_tie_line_is_drawn_from_the_domain_not_from_a_tick():
    """50% must appear whenever it is in view, whatever the tick ladder does.

    It used to be whichever tick happened to equal 50, so it vanished when the
    ladder stepped over it. Wyoming and West Virginia span 20-80%, the step
    comes out as 20, and the ticks are 40 and 60 -- the one line saying who is
    ahead was missing from exactly the two charts furthest from a tie.
    """
    body = candidate_chart()
    assert re.search(r"if \(lo < 50 && hi > 50\)", body), (
        "the 50% line must be drawn from the domain, independently of niceTicks"
    )


def test_the_tick_loop_neither_draws_nor_suppresses_the_tie_line():
    body = candidate_chart()
    loop = re.search(r"for \(const tick of niceTicks\(.*?\n  \}", body, re.S)
    assert loop, "could not find the tick loop"
    loop = loop.group(0)
    assert "continue" in loop, "the tick loop must skip 50, or it is drawn twice"
    assert "threshold-line" not in loop, (
        "the tick loop must not be what decides whether 50 is drawn"
    )


def test_shares_are_plotted_as_a_mirrored_pair():
    """Two-party shares sum to 100, so the lines must be built from one value.

    Deriving the Republican line independently would let rounding or a changed
    accessor put both candidates above 50 at once.
    """
    body = candidate_chart()
    assert "100 - demShare(d.p50)" in body, (
        "the Republican line must be 100 minus the Democratic share"
    )


def test_polls_before_the_grid_starts_are_dropped_not_clamped():
    """A poll older than the chart must not pile up on the left-hand edge."""
    body = candidate_chart()
    assert re.search(r"if \(Date\.parse\(poll\.date\) < t0\) continue", body), (
        "polls before the first grid point must be skipped, not drawn at x=padL"
    )


# --- direct labelling -------------------------------------------------------
#
# The two series carry their own names at the point each line ends, replacing
# the swatch legend that used to sit above the chart. What can go wrong is not
# visible in a diff: a label drawn outside the plot, two labels stacked on the
# same pixel in a tied race, or a full name where only a surname fits.


def test_the_right_margin_is_sized_to_the_labels():
    """A label drawn past the viewBox is simply not on the chart.

    padR was a constant 12px when the lines were unlabelled. A name placed at
    the end of a line needs room that the geometry has to know about before it
    lays anything out, which is why the candidate names are resolved above the
    padding rather than beside the tooltip where they used to live.
    """
    source = candidate_chart()
    assert re.search(r"padR\s*=\s*[\d\s+]*labelGutter\(", source), (
        "the right margin no longer accounts for the label width, so a long "
        "surname will be drawn outside the chart"
    )
    names = source.index("const demLabel")
    padding = source.index("const padR")
    assert names < padding, (
        "the labels must be resolved before the padding that is sized to fit them"
    )


def test_labels_are_separated_before_they_are_drawn():
    """The two lines are mirrors about 50%, so a toss-up puts them on one pixel.

    Alaska sits at D+1.0 today and the two labels come out 15px apart. At a
    true tie they would be identical, and a stack of two names in two colours
    is worse than no label at all.
    """
    source = candidate_chart()
    assert "separateLabels(" in source, (
        "candidate labels are placed without a collision guard; a tied race "
        "will stack them"
    )


def test_the_separation_guard_moves_the_pair_rather_than_clamping_each_end():
    """Clamping independently undoes the separation it just applied.

    Two labels pushed past the same edge both land on that edge -- back on one
    pixel, which is the state the guard exists to prevent. The pair has to
    translate as a block.
    """
    start = CHARTS.index("function separateLabels")
    rest = CHARTS[start:]
    body = rest[: rest.index("\n}")]
    assert "lo += top - hi" in body and "hi -= lo - bottom" in body, (
        "separateLabels clamps each end independently, which re-stacks any pair "
        "that overflows the same edge"
    )


def test_only_surnames_are_drawn_on_the_lines():
    """"Shelley Moore Capito" is a paragraph at the end of a 210px chart."""
    assert "function surname" in CHARTS
    assert "surname(demName)" in candidate_chart(), (
        "the full candidate name is being drawn on the line"
    )


def test_the_seat_chart_labels_only_the_sides_it_draws():
    """A key to a colour that never appears is noise.

    The House can produce a distribution entirely on one side of the majority
    line. Printing "Republican majority" over an all-blue histogram would be
    labelling something that is not there.
    """
    start = CHARTS.index("function renderSeatChart")
    rest = CHARTS[start + 1 :]
    body = rest[: rest.find("\nfunction ")]
    assert "bars.some((d) => d.seats < threshold)" in body, (
        "the seat chart labels the Republican side unconditionally"
    )
    assert "bars.some((d) => d.seats >= threshold)" in body, (
        "the seat chart labels the Democratic side unconditionally"
    )


def test_labels_that_sit_over_the_data_carry_a_halo():
    """Direct labelling puts text wherever the data happens to be.

    A series label lands on whatever its line ends over -- a gridline, the
    uncertainty band, the other line. The majority label on the seat histogram
    sits directly over the tallest bars in the distribution by construction.
    Both need to be readable against all of it.
    """
    css = (paths.SITE_DIR / "css" / "style.css").read_text(encoding="utf-8")
    for rule in (".series-label", ".threshold-text"):
        block = re.search(rf"{re.escape(rule)}\s*\{{(.*?)\}}", css, re.S)
        assert block, f"{rule} is not defined"
        assert "paint-order: stroke" in block.group(1), (
            f"{rule} has no halo, so it will be unreadable wherever it lands "
            f"on top of the data"
        )
