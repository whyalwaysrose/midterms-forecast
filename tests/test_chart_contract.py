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
