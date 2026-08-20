"""The hierarchical Bayesian dynamic model.

STRUCTURE
---------
Everything is on the logit scale of the Democratic two-party share.

A single national environment evolves as a Gaussian random walk::

    eta_t = eta_{t-1} + N(0, sigma_eta^2)

Each race is that environment, geared by its own elasticity, plus a baseline
and its own idiosyncratic drift::

    theta_{r,t} = alpha_r + lambda_r * eta_t + eps_{r,t}

    alpha_r  ~ N(fundamentals_r, tau^2)      structural prior (tied environment)
    lambda_r ~ TruncNormal(1, .25)           how hard the race swings with the nation
    eps_{r,t}                                race-specific random walk, eps_{r,0} = 0

Polls observe that latent state through a biased, noisy instrument::

    logit(y_i) ~ StudentT(nu, theta_{r_i,t_i} + h_{p_i} + g_{pop_i} + rho*s_i,
                          sqrt(v_i + sigma_excess^2))

IDENTIFIABILITY
---------------
Three choices keep this from being a badly-posed problem:

1. House effects are **sum-to-zero** (``ZeroSumNormal``). Without the
   constraint you could add a constant to every house effect and subtract it
   from the latent state and get the identical likelihood, so the sampler would
   wander along that ridge forever.
2. The population effect for the reference screen (likely voters) is pinned to
   exactly zero, so the remaining screens are estimated *relative* to it.
3. The fundamentals prior recenters each state against the nation, so the
   national swing is represented once — by ``eta`` — and not also inside
   ``alpha``.

Every random walk is written **non-centred** (a cumulative sum of standard
normals scaled by a separately-sampled SD). Centred random walks produce the
classic funnel geometry and sample badly; this formulation does not.
"""

from __future__ import annotations

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..config import ModelConfig
from .design import ModelData

log = logging.getLogger(__name__)


def build_model(data: ModelData, cfg: ModelConfig) -> pm.Model:
    """Construct the PyMC model for a set of races."""
    grid_days = cfg.grid_days
    # Per-day innovation SDs are supplied in config; a grid step spans
    # `grid_days` days, and random-walk variance is additive in time.
    national_step_scale = cfg.national_environment.rw_sd_per_day_prior * np.sqrt(grid_days)
    race_step_scale = cfg.race.rw_sd_per_day_prior * np.sqrt(grid_days)

    coords = {
        "race": list(data.race_ids),
        "grid": [d.isoformat() for d in data.grid_dates],
        "grid_step": list(range(data.n_steps)),
        "pollster": list(data.pollster_names),
        "population_nonref": [
            data.populations[i] for i in data.nonreference_population_indices
        ],
        "obs": list(range(len(data.y))),
    }

    n_races = data.n_races
    nonref = data.nonreference_population_indices

    with pm.Model(coords=coords) as model:
        # ------------------------------------------------------------------
        # National environment: eta_t, a non-centred Gaussian random walk.
        # eta = 0 corresponds to a tied national generic ballot.
        # ------------------------------------------------------------------
        sigma_eta = pm.HalfNormal("sigma_eta", sigma=national_step_scale)
        eta_start = pm.Normal("eta_start", 0.0, cfg.national_environment.initial_sd)
        eta_innovations = pm.Normal("eta_innovations", 0.0, 1.0, dims="grid_step")
        eta = pm.Deterministic(
            "eta",
            eta_start
            + pt.concatenate([pt.zeros(1), pt.cumsum(eta_innovations) * sigma_eta]),
            dims="grid",
        )

        # ------------------------------------------------------------------
        # Race baselines under a tied national environment.
        # ------------------------------------------------------------------
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="race")
        alpha = pm.Deterministic(
            "alpha",
            pt.as_tensor_variable(data.fundamentals_prior_mean)
            + cfg.fundamentals.prior_sd * z_alpha,
            dims="race",
        )

        # ------------------------------------------------------------------
        # Elasticity: how much of the national swing each race absorbs.
        # Partially pooled around 1 and kept positive so the sign of the
        # national environment cannot flip for an individual race.
        # ------------------------------------------------------------------
        elasticity = pm.TruncatedNormal(
            "elasticity",
            mu=cfg.race.elasticity_mean,
            sigma=cfg.race.elasticity_sd,
            lower=cfg.race.elasticity_lower,
            dims="race",
        )

        # ------------------------------------------------------------------
        # Race-specific drift: a random walk pinned to zero at the grid start,
        # so it captures *movement* while alpha captures *level*.
        # ------------------------------------------------------------------
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=race_step_scale)
        eps_innovations = pm.Normal(
            "eps_innovations", 0.0, 1.0, dims=("race", "grid_step")
        )
        eps = pt.concatenate(
            [
                pt.zeros((n_races, 1)),
                pt.cumsum(eps_innovations, axis=1) * sigma_eps,
            ],
            axis=1,
        )

        theta = pm.Deterministic(
            "theta",
            alpha[:, None] + elasticity[:, None] * eta[None, :] + eps,
            dims=("race", "grid"),
        )

        # ------------------------------------------------------------------
        # Measurement model.
        # ------------------------------------------------------------------
        sigma_house = pm.HalfNormal("sigma_house", sigma=cfg.polls.house_effect_sd_prior)
        # Sum-to-zero across pollsters: see IDENTIFIABILITY in the module docstring.
        #
        # Non-centred. Writing this as ZeroSumNormal(sigma=sigma_house) directly
        # would couple each pollster's effect to the shared scale and produce a
        # funnel: most of the 100+ pollsters have only one or two polls, so their
        # effects are prior-dominated, and the sampler then has to explore a
        # narrowing neck as sigma_house shrinks. Scaling a unit-scale zero-sum
        # variable keeps the sum-to-zero constraint (scaling preserves it) while
        # decoupling the geometry.
        house_effect_raw = pm.ZeroSumNormal("house_effect_raw", sigma=1.0, dims="pollster")
        house_effect = pm.Deterministic(
            "house_effect", sigma_house * house_effect_raw, dims="pollster"
        )

        population_effect_nonref = pm.Normal(
            "population_effect_nonref",
            0.0,
            cfg.polls.population_effect_sd_prior,
            dims="population_nonref",
        )
        population_effect = pt.set_subtensor(
            pt.zeros(len(data.populations))[nonref], population_effect_nonref
        )
        pm.Deterministic("population_effect", population_effect)

        # Partisan sponsors favour their own side, so the coefficient is
        # constrained positive and multiplied by a signed indicator.
        partisan_effect = pm.HalfNormal(
            "partisan_effect", sigma=cfg.polls.partisan_effect_prior
        )

        sigma_excess = pm.HalfNormal("sigma_excess", sigma=cfg.polls.excess_sd_prior)

        # Latent value each poll is measuring. Race polls come first in the
        # observation ordering, then national generic-ballot polls.
        latent_parts = []
        if data.n_race_polls:
            latent_parts.append(
                theta[data.race_poll_race_idx, data.race_poll_time_idx]
            )
        if data.n_national_polls:
            latent_parts.append(eta[data.national_poll_time_idx])
        latent = pt.concatenate(latent_parts) if len(latent_parts) > 1 else latent_parts[0]

        mu = (
            latent
            + house_effect[data.pollster_idx]
            + population_effect[data.population_idx]
            + partisan_effect * pt.as_tensor_variable(data.partisan_sign)
        )

        sigma_obs = pt.sqrt(
            pt.as_tensor_variable(data.sampling_var) + sigma_excess**2
        )

        pm.StudentT(
            "poll_obs",
            nu=cfg.polls.student_t_nu,
            mu=mu,
            sigma=sigma_obs,
            observed=data.y,
            dims="obs",
        )

    return model


def sample(model: pm.Model, cfg: ModelConfig, progressbar: bool = True):
    """Draw from the posterior, preferring the Numba (nutpie) NUTS backend.

    On Windows the default PyTensor C backend needs a compiler that is usually
    absent; nutpie compiles through Numba instead. If it is unavailable we fall
    back to PyMC's own NUTS rather than failing the run.
    """
    sampler_kwargs = dict(
        draws=cfg.sampler.draws,
        tune=cfg.sampler.tune,
        chains=cfg.sampler.chains,
        target_accept=cfg.sampler.target_accept,
        random_seed=cfg.sampler.seed,
        progressbar=progressbar,
    )

    backend = cfg.sampler.backend
    if backend == "nutpie":
        try:
            import nutpie  # noqa: F401
        except ImportError:
            log.warning("nutpie not installed; falling back to PyMC NUTS")
            backend = "pymc"

    with model:
        if backend == "nutpie":
            try:
                return pm.sample(nuts_sampler="nutpie", **sampler_kwargs)
            except Exception as exc:  # pragma: no cover - environment dependent
                log.warning("nutpie sampling failed (%s); falling back to PyMC NUTS", exc)
        return pm.sample(**sampler_kwargs)


def convergence_report(idata) -> dict[str, float]:
    """Headline convergence diagnostics.

    A forecast whose sampler did not converge is not a forecast, so these are
    written into every run's output and surfaced on the dashboard.
    """
    import arviz as az

    # Deterministics are functions of the sampled parameters; diagnosing them
    # adds noise without adding information.
    var_names = [
        "sigma_eta",
        "eta_start",
        "eta_innovations",
        "z_alpha",
        "elasticity",
        "sigma_eps",
        "eps_innovations",
        "sigma_house",
        "house_effect_raw",
        "population_effect_nonref",
        "partisan_effect",
        "sigma_excess",
    ]
    present = [v for v in var_names if v in idata.posterior]
    summary = az.summary(idata, var_names=present, round_to=None)

    divergences = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0

    return {
        "max_r_hat": float(np.nanmax(summary["r_hat"].to_numpy())),
        "min_ess_bulk": float(np.nanmin(summary["ess_bulk"].to_numpy())),
        "min_ess_tail": float(np.nanmin(summary["ess_tail"].to_numpy())),
        "divergences": divergences,
        "n_draws": int(idata.posterior.sizes["draw"] * idata.posterior.sizes["chain"]),
    }
