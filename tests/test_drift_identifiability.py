"""Race drift must not carry a second national swing.

`theta = alpha + lambda * eta + eps`. If `eps` is free to move every race the
same way at once, that common component is a national swing -- and the model
already has one, `eta`, fitted to hundreds of generic-ballot polls. The two then
trade off along a ridge: the likelihood cannot tell them apart, so the sampler
explores the ridge instead of the posterior, and whichever term the data happens
to grab first sets the level.

The consequences were not subtle. Ten states redrew their congressional maps for
2026, moving the House fundamentals thirteen seats toward the Republicans, and
the published forecast did not move at all -- `eps` absorbed the change. Beneath
that, 38 polled districts running 2.6 points more Democratic than their own
fundamentals were setting the level for all 397 unpolled ones, which is a
selected sample doing a job the generic ballot already does from 448 polls.

So `eps` is sum-to-zero across races at every step, exactly as house effects are
sum-to-zero across pollsters and for exactly the same reason. These tests pin
that, because nothing about it raises if it regresses: the model still samples,
still converges, and quietly answers a slightly different question.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymc")

from midterms.config import load_all  # noqa: E402
from midterms.model.hierarchical import build_model  # noqa: E402


@pytest.fixture(scope="module")
def prior_draws():
    """Draws of `theta` and its parts from the prior, on the real Senate design.

    The prior is enough and much cheaper than fitting: the constraint is a
    property of how the model is written, not of what the data says.
    """
    import pymc as pm

    from midterms import fundamentals as F
    from midterms.data import Roster, build_poll_table
    from midterms.data.votehub import latest_snapshot, load_snapshot
    from midterms.model.design import build_model_data

    snapshot = latest_snapshot()
    if snapshot is None:
        pytest.skip("no local poll snapshot to build a design from")

    races, cfg = load_all(chamber="senate")
    table = build_poll_table(load_snapshot(snapshot), races, cfg, Roster.load())
    if not table.race_polls:
        pytest.skip("snapshot holds no Senate polls")

    data = build_model_data(table, races, cfg, F.compute(races, cfg))
    with build_model(data, cfg):
        idata = pm.sample_prior_predictive(draws=200, random_seed=3)
    return idata.prior, data


def _eps(prior, data):
    """Recover eps = theta - alpha - lambda*eta, in (draw, race, grid)."""
    theta = prior["theta"].to_numpy()[0]
    alpha = prior["alpha"].to_numpy()[0]
    eta = prior["eta"].to_numpy()[0]
    lam = prior["elasticity"].to_numpy()[0]
    return theta - alpha[:, :, None] - lam[:, :, None] * eta[:, None, :]


def test_race_drift_sums_to_zero_across_races(prior_draws):
    """The constraint itself, measured on the reconstructed drift."""
    prior, data = prior_draws
    eps = _eps(prior, data)
    across_races = eps.sum(axis=1)          # (draw, grid)

    # Scale the tolerance to how large the drift actually is, so this cannot be
    # passed by a model whose drift is simply tiny.
    typical = np.abs(eps).mean()
    assert typical > 1e-6, "drift is ~0 everywhere; the test would prove nothing"
    assert np.abs(across_races).max() < 1e-6 * max(1.0, eps.shape[1]), (
        f"race drift has a common component: max |sum over races| = "
        f"{np.abs(across_races).max():.2e} against a typical |eps| of {typical:.2e}"
    )


def test_drift_still_moves_individual_races(prior_draws):
    """The constraint must not have flattened the thing it constrains.

    Sum-to-zero removes the *common* component and nothing else. Races must
    still drift, and still drift by different amounts, or the correlated-movement
    machinery has been silently disabled rather than corrected.
    """
    prior, data = prior_draws
    eps = _eps(prior, data)
    final = eps[:, :, -1]                    # drift accumulated to election day

    assert np.abs(final).mean() > 1e-4, "races no longer drift at all"
    assert final.std(axis=1).mean() > 1e-4, "races all drift identically"


def test_drift_still_correlates_races(prior_draws):
    """Correlated movement is the point of the term and must survive.

    Under the constraint the average pairwise correlation is necessarily
    slightly negative -- with n races summing to zero it is about -1/(n-1) --
    so this checks that *similar* races still move together more than the
    average pair does, which is the behaviour unpolled races rely on.
    """
    prior, data = prior_draws
    eps = _eps(prior, data)
    final = eps[:, :, -1]
    n = final.shape[1]

    corr = np.corrcoef(final.T)
    off = corr[~np.eye(n, dtype=bool)]
    forced = -1.0 / (n - 1)

    assert off.max() > 0.2, (
        "no pair of races moves together; the correlation structure is gone"
    )
    assert off.mean() == pytest.approx(forced, abs=0.15), (
        f"mean pairwise correlation {off.mean():.3f} is far from the {forced:.3f} "
        f"the constraint forces -- the drift may not be constrained as intended"
    )
