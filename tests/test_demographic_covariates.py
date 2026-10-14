"""The demographic-covariate measurement, and the data it rests on.

METHODOLOGY 9.1 spent months saying demographic covariates were the highest-value
available improvement to the correlation kernel. They were measured on
2026-10-14 and they are not: education lowers agreement with the measured error
ranking because it is +0.79 collinear with the 2024 lean the kernel already
carries, and the one specification that helps does so by +0.005 inside a
bootstrap interval of +/-0.11.

A negative result is only worth the file it is written in if it stays
reproducible, so these tests pin the parts that could rot quietly: the vendored
CSV covering every racing state, the residualisation actually removing what it
claims to, and the district-level counts that make the House half unconfirmable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fit_demographic_covariates as D  # noqa: E402,I001


def test_the_vendored_csv_covers_every_state_racing_in_2026():
    """A gap here would silently drop a state from the covariate.

    The script refuses to score with gaps, so this is the test that keeps that
    refusal from being the way anyone finds out.
    """
    from midterms.config import load_all

    demo = D.load_demographics()
    races, _cfg = load_all()
    units = {race.unit for race in races.races}
    missing = sorted(units - set(demo))
    assert not missing, f"no demographics for {missing}"


def test_the_vendored_values_are_plausible_shares():
    demo = D.load_demographics()
    assert len(demo) >= 51, "expected 50 states plus DC at least"
    for state, row in demo.items():
        assert 0.0 < row["education"] < 100.0, state
        assert 0.0 < row["urban"] <= 100.0, state
    # Anchors against published figures, so a silently shifted column or a
    # changed table shell is caught rather than averaged away. West Virginia is
    # the least degree-holding state and DC the most, by a wide margin.
    assert demo["WV"]["education"] < 26.0
    assert demo["DC"]["education"] > 55.0
    assert demo["DC"]["urban"] == pytest.approx(100.0, abs=0.5)


def test_residualising_removes_the_controls_explanation():
    """The specification that makes the experiment well-posed.

    Appending a covariate raw mostly re-weights an axis the kernel has. The
    residual is the part that is new, and it must be orthogonal to what it was
    regressed on or it is not.
    """
    rng = np.random.default_rng(0)
    control = rng.normal(size=60)
    other = rng.normal(size=60)
    # Mostly explained by `control`, with a little of its own.
    target = 3.0 * control + 0.4 * rng.normal(size=60)

    residual = D.residualise_against(target, control, other)

    assert abs(float(np.corrcoef(residual, control)[0, 1])) < 1e-9
    assert abs(float(np.corrcoef(residual, other)[0, 1])) < 1e-9
    # And it kept strictly less variance than it started with, since the
    # controls genuinely explained some.
    assert np.var(residual) < np.var(target)


def test_residualising_against_nothing_only_centres():
    target = np.array([1.0, 2.0, 3.0, 10.0])
    residual = D.residualise_against(target)
    assert residual == pytest.approx(target - target.mean())


def test_no_district_pair_recurs_often_enough_to_measure():
    """Finding 5, which is the whole reason the House half is unanswerable.

    If this ever fails because a count went up, that is good news and the
    district-level measurement should be attempted -- so the assertion is
    written to say so rather than to pin zero forever.
    """
    counts = D.district_pair_counts()
    assert set(counts) == {"2002-2010", "2012-2020", "2022-"}

    total_district_cycles = sum(dc for dc, _ in counts.values())
    total_recurring = sum(pairs for _, pairs in counts.values())

    # House district polling is thin: ~120 district-cycles across 21 years.
    assert total_district_cycles < 200, (
        f"{total_district_cycles} district-cycles is more than this dataset had; "
        "re-check whether a district-level kernel fit is now possible"
    )
    assert total_recurring == 0, (
        f"{total_recurring} district pairs now share four cycles. The House "
        "kernel may be measurable after all -- see METHODOLOGY weakness 1."
    )


def test_the_education_table_columns_are_the_ba_plus_lines():
    """B15003 lines 22-25 are bachelor's, master's, professional, doctorate.

    Hardcoded line numbers against a Census table shell are exactly the kind of
    thing that silently becomes wrong, and the national share is the cheap check:
    the 2023 ACS 5-year figure for 25+ is about 35%.
    """
    assert D.BA_PLUS == (22, 23, 24, 25)

    demo = D.load_demographics()
    # Not a population-weighted national figure, but the unweighted mean across
    # states should still land in the same neighbourhood rather than, say, 8% or
    # 80% -- which is what picking up the wrong columns would give.
    mean_share = sum(r["education"] for r in demo.values()) / len(demo)
    assert 28.0 < mean_share < 40.0, mean_share


def test_the_byte_range_still_reaches_the_state_block():
    """The 400 KB assumption behind not downloading 92 MB.

    Asserted on the committed constant rather than over the network, so the
    suite stays offline; the script itself fails loudly if the range stops
    containing 51 states.
    """
    assert D.EDU_RANGE.startswith("bytes=0-")
    end = int(D.EDU_RANGE.split("-")[1])
    assert end >= 400_000, "state rows were found within the first 400 KB"


def test_the_vendored_csv_is_small_enough_to_belong_in_the_repo():
    size = (REPO / "data" / "history" / "state_demographics.csv").stat().st_size
    assert size < 8_000, f"{size} bytes; derive it rather than vendor it"
