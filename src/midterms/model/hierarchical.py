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

        approval_cfg = cfg.national_environment.approval
        use_approval = approval_cfg.enabled and data.n_approval_polls > 0

        if use_approval:
            # Presidential approval as a second indicator of the same national
            # mood. The two series share *movement*, not level: their
            # innovations are correlated by a coefficient the model estimates,
            # while approval keeps its own intercept.
            #
            # Level is deliberately not shared. Turning "net approval -8" into a
            # generic-ballot margin needs a historical fit across many midterms,
            # and the approval series available here starts in 2018 — two
            # midterms, which cannot identify that relationship. Correlated
            # movement is a far weaker claim and one the data can actually
            # support.
            corr_raw = pm.Normal("approval_corr_raw", 0.0, 1.0)
            approval_corr = pm.Deterministic(
                "approval_corr", pt.tanh(corr_raw * approval_cfg.correlation_prior_sd)
            )
            approval_innovations_raw = pm.Normal(
                "approval_innovations_raw", 0.0, 1.0, dims="grid_step"
            )
            # Cholesky of [[1, rho], [rho, 1]] applied to the pair, so the
            # approval series keeps unit-variance innovations.
            approval_innovations = (
                approval_corr * eta_innovations
                + pt.sqrt(1.0 - approval_corr**2) * approval_innovations_raw
            )
            sigma_approval = pm.HalfNormal(
                "sigma_approval",
                sigma=approval_cfg.rw_sd_per_day_prior * np.sqrt(grid_days),
            )
            approval_start = pm.Normal(
                "approval_start", 0.0, approval_cfg.initial_sd
            )
            approval_level = pm.Deterministic(
                "approval_level",
                approval_start
                + pt.concatenate(
                    [pt.zeros(1), pt.cumsum(approval_innovations) * sigma_approval]
                ),
                dims="grid",
            )

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
        # Correlate drift across politically similar races. L is the Cholesky of
        # a correlation matrix, so each race's marginal innovation variance is
        # still 1 (its rows have unit norm) — this changes how races move
        # *together*, not how far any one of them moves.
        #
        # The payoff is concentrated in the races with no polling: under
        # independence they could only follow the national environment, whereas
        # now a poll in a similar state carries information to them.
        correlated_innovations = pt.dot(
            pt.as_tensor_variable(data.movement_chol), eps_innovations
        )
        eps = pt.concatenate(
            [
                pt.zeros((n_races, 1)),
                pt.cumsum(correlated_innovations, axis=1) * sigma_eps,
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
        if use_approval:
            latent_parts.append(approval_level[data.approval_poll_time_idx])
        latent = pt.concatenate(latent_parts) if len(latent_parts) > 1 else latent_parts[0]

        # The observation vector is built in design.py and the latent vector
        # here; they are assembled independently and must line up exactly.
        # Without this check a mismatch surfaces as an Elemwise shape error
        # several frames deep inside PyTensor, naming neither cause nor cure.
        expected = data.n_race_polls + data.n_national_polls + (
            data.n_approval_polls if use_approval else 0
        )
        if expected != len(data.y):
            raise ValueError(
                f"design/model mismatch: model expects {expected} observations "
                f"({data.n_race_polls} race + {data.n_national_polls} generic + "
                f"{data.n_approval_polls if use_approval else 0} approval) but the "
                f"design supplies {len(data.y)}. The approval switch is probably "
                f"set differently in build_model_data and build_model."
            )

        mu = (
            latent
            + house_effect[data.pollster_idx]
            + population_effect[data.population_idx]
            + partisan_effect * pt.as_tensor_variable(data.partisan_sign)
        )

        sigma_obs = pt.sqrt(
            pt.as_tensor_variable(data.sampling_var) + sigma_excess**2
        )

        # A Student-t's standard deviation is its scale times sqrt(nu/(nu-2)),
        # so passing the intended SD in as `sigma` silently inflates every
        # poll's error — by 41% at nu=4. Dividing it out keeps the heavy tails,
        # which are the reason for using t at all, while making the likelihood's
        # actual spread the one calibration asked for.
        nu = cfg.polls.student_t_nu
        if cfg.polls.match_student_t_variance:
            sigma_obs = sigma_obs / np.sqrt(nu / (nu - 2.0))

        pm.StudentT(
            "poll_obs",
            nu=nu,
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
        "approval_corr",
        "sigma_approval",
        "approval_start",
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
