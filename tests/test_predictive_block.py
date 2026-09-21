"""The block that lets a later run score polls this one never saw.

Nothing else in the suite builds a ForecastRun, so without this the predictive
block would first execute on a CI runner in the middle of the daily forecast --
the one place a mistake is most expensive and least visible.

The posterior here is synthetic. It cannot prove the block matches the shape of
a real trace, and the daily run is what actually establishes that. What it does
prove is that the arithmetic, the coordinate lookups and the reference-population
detection are right, and that a broken block degrades to an omission instead of
taking the forecast down with it.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from midterms.outputs import ForecastRun  # noqa: E402

POLLSTERS = ("Marist", "Quinnipiac", "__pooled__")
POPULATIONS = ("lv", "rv", "a")
REFERENCE = "rv"           # pinned at exactly zero, as the model constructs it


def synthetic_posterior(seed: int = 0):
    """A posterior with the variables and coordinates the block reads."""
    rng = np.random.default_rng(seed)
    chains, draws = 2, 250

    house = rng.normal(0.0, 0.02, (chains, draws, len(POLLSTERS)))
    population = rng.normal(0.01, 0.01, (chains, draws, len(POPULATIONS)))
    population[:, :, POPULATIONS.index(REFERENCE)] = 0.0     # the reference

    return xr.Dataset(
        {
            "house_effect": (("chain", "draw", "pollster"), house),
            "population_effect": (("chain", "draw", "population"), population),
            "partisan_effect": (("chain", "draw"),
                                np.abs(rng.normal(0.01, 0.003, (chains, draws)))),
            "sigma_excess": (("chain", "draw"),
                             np.abs(rng.normal(0.03, 0.005, (chains, draws)))),
        },
        coords={
            "chain": list(range(chains)),
            "draw": list(range(draws)),
            "pollster": list(POLLSTERS),
            "population": list(POPULATIONS),
        },
    )


class _Idata:
    def __init__(self, posterior):
        self.posterior = posterior


def _real_poll(poll_id: str):
    """A genuine NormalisedPoll, not a stand-in.

    This used to be a hand-written fake with `self.id = poll_id`. The block read
    `p.id`, the fake supplied `.id`, and the test passed -- while the real class
    has `poll_id` and no `id` at all. So every production run from 2026-09-16
    raised AttributeError inside the block, the wrapper swallowed it as designed,
    and five days of forward-calibration evidence went uncollected with every
    workflow reporting success.

    A fake that agrees with the code under test proves nothing about the code
    under test. Building the real dataclass means the shape can only come from
    one place.
    """
    from datetime import date

    from midterms.data.polls import NormalisedPoll

    return NormalisedPoll(
        poll_id=poll_id, race_id="__national__", pollster="Test",
        field_date=date(2026, 9, 15), start_date=date(2026, 9, 13),
        end_date=date(2026, 9, 15), sample_size=1000, population="lv",
        dem_pct=48.0, rep_pct=46.0, two_party_dem=48.0 / 94.0, other_pct=6.0,
        partisan_sign=0, sponsors=(), url="", dem_candidate="", rep_candidate="",
    )


class _Table:
    national = (_real_poll("nat-2"), _real_poll("nat-1"))


def make_run(posterior=None, cfg=None):
    """A ForecastRun carrying only what the predictive block touches."""
    from midterms.config import ModelConfig

    run = ForecastRun.__new__(ForecastRun)
    run.idata = _Idata(posterior if posterior is not None else synthetic_posterior())
    run.table = _Table()
    run.cfg = cfg or ModelConfig.load(chamber="senate")
    return run


def test_the_block_carries_what_scoring_a_future_poll_needs():
    """A predictive built from the latent trajectory alone is too narrow.

    Everything between the latent state and an observed number has to come with
    it -- house effect, population effect, partisan lean, excess noise -- or a
    later run will judge this model overconfident when it is not.
    """
    block = make_run()._predictive_block()
    assert block is not None

    for key in ("house_effect", "population_effect", "partisan_effect",
                "sigma_excess", "design_effect", "student_t_nu",
                "match_student_t_variance", "reference_population",
                "national_poll_ids"):
        assert key in block, f"{key} is missing; a later run cannot rebuild the predictive"

    assert set(block["house_effect"]) == set(POLLSTERS)
    assert set(block["population_effect"]) == set(POPULATIONS)
    for entry in block["house_effect"].values():
        assert {"mean", "sd"} == set(entry)
        assert entry["sd"] > 0


def test_the_likelihood_settings_travel_with_it():
    """Scored against the distribution this model used, not today's config.

    `design_effect` and `student_t_nu` have both been retuned during this cycle.
    A forward score that read the current config would silently grade an old
    forecast against a likelihood it never had.
    """
    block = make_run()._predictive_block()
    from midterms.config import ModelConfig

    cfg = ModelConfig.load(chamber="senate")
    assert block["design_effect"] == cfg.polls.design_effect
    assert block["student_t_nu"] == cfg.polls.student_t_nu
    assert block["match_student_t_variance"] == cfg.polls.match_student_t_variance


def test_the_reference_population_identifies_itself():
    """It is the one pinned at exactly zero, which is how the model builds it.

    Reading it from a stored index would be one more thing to keep in step;
    reading it from the values cannot drift.
    """
    assert make_run()._predictive_block()["reference_population"] == REFERENCE


def test_national_poll_ids_are_listed_and_sorted():
    """Race polls are identified per race by `all_poll_ids`. Generic-ballot
    polls have no race to hang off, so without these they could never be
    recognised as ones a later run had not seen."""
    assert make_run()._predictive_block()["national_poll_ids"] == ["nat-1", "nat-2"]


def test_a_broken_block_costs_the_diagnostic_and_not_the_forecast():
    """The whole point of the try/except, stated as a test.

    This block exists to answer a calibration question. Publishing no forecast
    at all because it could not be assembled would be a far worse outcome than
    losing a day of calibration data, and the failure is visible in the next
    run's forward score rather than silent.
    """
    broken = synthetic_posterior().drop_vars("sigma_excess")
    assert make_run(posterior=broken)._predictive_block() is None
