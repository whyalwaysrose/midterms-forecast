"""Wording guards for the generated commentary.

The commentary is the part of the project a human actually reads, so phrasing
bugs matter as much as numerical ones. These pin the cases that read wrong.
"""

from __future__ import annotations

from midterms import commentary as C
from tests.test_commentary import make_payload, make_poll, make_race


def _headline(prev_prob, curr_prob, polls_prev, polls_curr):
    prev = make_payload("2026-08-19", prev_prob, [make_race(prob=prev_prob, polls=polls_prev)])
    curr = make_payload("2026-08-20", curr_prob, [make_race(prob=curr_prob, polls=polls_curr)])
    return C.generate(curr, prev).headline


def test_downward_move_does_not_read_as_a_positive_change():
    """Regression: the headline once read 'are down +0.7 pts'."""
    headline = _headline(0.60, 0.55, [make_poll("a")], [make_poll("a"), make_poll("b")])
    assert "down" in headline
    assert "down +" not in headline
    assert "up " not in headline


def test_upward_move_reads_as_up():
    headline = _headline(0.55, 0.60, [make_poll("a")], [make_poll("a"), make_poll("b")])
    assert "up " in headline
    assert "down" not in headline


def test_no_direction_word_is_paired_with_a_signed_magnitude():
    for prev_p, curr_p in [(0.60, 0.55), (0.55, 0.60), (0.50, 0.72)]:
        headline = _headline(prev_p, curr_p, [make_poll("a")], [make_poll("a"), make_poll("b")])
        assert "up +" not in headline
        assert "down +" not in headline
        assert "up -" not in headline
        assert "down -" not in headline


def test_flat_probability_with_new_polls_reads_as_unchanged():
    headline = _headline(0.60, 0.60, [make_poll("a")], [make_poll("a"), make_poll("b")])
    assert "unchanged at" in headline
    assert "1 new poll" in headline
    assert "1 new polls" not in headline


def test_new_poll_count_is_pluralised():
    many = [make_poll("a"), make_poll("b"), make_poll("c")]
    headline = _headline(0.55, 0.60, [make_poll("a")], many)
    assert "2 new polls" in headline
