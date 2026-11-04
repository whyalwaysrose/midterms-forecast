"""The arithmetic behind the post-mortem's load-bearing number.

METHODOLOGY 8c-i rests on one figure: the four North Carolina districts the
model missed were correlated at 0.94, so the chance of all four falling
together was about 19% rather than the 0.7% independence implies. That single
number is the difference between "one event the model priced at one-in-five"
and "four independent failures", and it is the basis for not shrinking the
kernel in response.

A quietly wrong copula or threshold would make that claim unfounded while
still printing a plausible percentage, so the maths is pinned here against
cases with known answers.
"""

from __future__ import annotations

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import postmortem_2026 as P  # noqa: E402,I001


def _tail(z: float) -> float:
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


@pytest.mark.parametrize("p", [0.01, 0.05, 0.21, 0.295, 0.5, 0.8, 0.99])
def test_the_threshold_inverts_the_normal_tail(p):
    """`_z_for(p)` must be the point with exactly p of the mass above it."""
    assert _tail(P._z_for(p)) == pytest.approx(p, abs=1e-6)


def test_a_half_maps_to_zero():
    assert P._z_for(0.5) == pytest.approx(0.0, abs=1e-6)


def test_independence_is_the_product_of_the_marginals():
    rng = np.random.default_rng(0)
    marginals = [0.29, 0.21, 0.31, 0.35]
    _together, independent = P.joint_flip_probability(marginals, 0.0, rng)
    assert independent == pytest.approx(0.29 * 0.21 * 0.31 * 0.35)


def test_at_zero_correlation_the_simulation_matches_independence():
    """The copula must reduce to the product when nothing is shared."""
    rng = np.random.default_rng(1)
    marginals = [0.3, 0.3, 0.3]
    together, independent = P.joint_flip_probability(marginals, 0.0, rng)
    assert together == pytest.approx(independent, abs=0.004)


def test_at_perfect_correlation_they_all_move_as_one():
    """With rho = 1 the joint probability is the smallest marginal.

    If every race shares one draw, they flip together exactly when the most
    demanding of them does.
    """
    rng = np.random.default_rng(2)
    together, _ = P.joint_flip_probability([0.3, 0.4, 0.5], 1.0, rng)
    assert together == pytest.approx(0.30, abs=0.005)


def test_correlation_raises_the_joint_probability_monotonically():
    rng = np.random.default_rng(3)
    marginals = [0.29, 0.21, 0.31, 0.35]
    values = [
        P.joint_flip_probability(marginals, rho, rng)[0]
        for rho in (0.0, 0.3, 0.6, 0.94)
    ]
    assert values == sorted(values)
    assert values[-1] > 20 * values[0], "correlation should dominate here"


def test_the_north_carolina_figure_reproduces():
    """The number METHODOLOGY 8c-i quotes, to the tenth it is quoted at."""
    rng = np.random.default_rng(20261104)
    together, independent = P.joint_flip_probability(
        [0.29, 0.21, 0.31, 0.35], 0.940, rng
    )
    assert together == pytest.approx(0.19, abs=0.02)
    assert independent == pytest.approx(0.007, abs=0.001)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.94, 0.99])
def test_the_copula_preserves_each_race_s_own_probability(rho):
    """The constraint that makes the joint number mean anything.

    A copula is only a statement about dependence; each race must still flip
    with exactly the probability the model gave it. Get the factor loadings
    wrong -- `rho` and `1 - rho` instead of their square roots, say -- and the
    latent variance stops being one, every marginal drifts, and the joint
    figure is computed for races that are not the ones being discussed. At high
    correlation the joint answer still looks about right, which is what makes
    it worth pinning separately.

    Exercised through the script's own function rather than a copy of its
    maths here: with a single race, "the probability they all flip" is just
    that race's probability, whatever the correlation. A test that rebuilt the
    copula would only agree with itself -- the shape of the `p.id` fixture that
    agreed with the predictive-block bug for five days.
    """
    for want in (0.29, 0.21, 0.31, 0.35):
        rng = np.random.default_rng(11)
        got, _ = P.joint_flip_probability([want], rho, rng)
        assert got == pytest.approx(want, abs=0.005), (
            f"a lone race at rho={rho} flipped {got:.3f} of the time, not {want}"
        )


def test_a_cluster_needs_more_than_a_pair():
    """Two races going the same way is a coincidence, not a pattern."""
    assert P.MIN_CLUSTER >= 3


def test_state_is_read_from_either_chamber_s_race_id():
    assert P._state_of("house-2026-NC-09") == "NC"
    assert P._state_of("senate-2026-GA") == "GA"
    assert P._state_of("house-2026-AK-01") == "AK"
