"""Calibration checking.

The honest problem with backtesting an election model in August is that the
election has not happened, so there are no outcomes to score against. What we
*can* score, right now and continuously, is whether the model predicts **polls
it has not seen**.

The procedure:

    1. Refit using only polls fielded up to ``today - holdout_days``.
    2. Form the posterior predictive distribution for each poll fielded after
       that cutoff — including house effects, screen adjustments and the
       measurement noise the model believes in.
    3. Ask how often reality landed inside the intervals.

If the 80% intervals contain 80% of held-out polls, the model's uncertainty is
roughly right. If they contain 95%, it is underconfident; 60%, overconfident.
The PIT (probability integral transform) values sharpen this: for a
well-calibrated model they are uniform on [0, 1], and their deviation from
uniformity says *how* the model is wrong, not just that it is.

This is a necessary condition for calibration, not a sufficient one — a model
can predict polls well and still be wrong about the election if the polls
themselves are biased. That residual risk is what ``election_day_error`` in the
config exists to represent, and it can only be validated against real results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from . import fundamentals as F
from .config import ModelConfig, RaceSet, load_all
from .data.polls import NATIONAL_RACE_ID, NormalisedPoll, build_poll_table
from .data.roster import Roster
from .data.votehub import VoteHubClient, latest_snapshot, load_snapshot
from .model.design import build_model_data
from .model.hierarchical import build_model, sample

log = logging.getLogger(__name__)

CREDIBLE_LEVELS = (0.5, 0.8, 0.9, 0.95)


@dataclass
class CalibrationResult:
    n_holdout: int
    cutoff: date
    coverage: dict[float, float]
    pit: np.ndarray
    mean_abs_error_margin: float
    median_abs_error_margin: float
    ks_statistic: float

    def report(self) -> str:
        lines = [
            f"Held-out polls: {self.n_holdout} (fielded after {self.cutoff})",
            f"Mean |error| vs poll margin:   {self.mean_abs_error_margin:.2f} pts",
            f"Median |error| vs poll margin: {self.median_abs_error_margin:.2f} pts",
            "",
            "Interval coverage (target = nominal):",
        ]
        for level in CREDIBLE_LEVELS:
            actual = self.coverage[level]
            verdict = (
                "well calibrated" if abs(actual - level) <= 0.07
                else "UNDERCONFIDENT" if actual > level
                else "OVERCONFIDENT"
            )
            lines.append(f"  {level:.0%} interval -> {actual:.1%}   {verdict}")
        lines.append("")
        lines.append(f"PIT uniformity (KS statistic): {self.ks_statistic:.3f}")
        lines.append(
            "  (below ~0.10 is good for this sample size; large values mean the "
            "predictive distribution is the wrong shape, not merely the wrong width)"
        )
        return "\n".join(lines)


def _posterior_predictive_for_poll(
    poll: NormalisedPoll,
    idata,
    data,
    cfg: ModelConfig,
    race_index: dict[str, int],
    grid: tuple[date, ...],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Draws from the predictive distribution of ``logit(y)`` for one poll."""
    posterior = idata.posterior

    offset = (poll.field_date - grid[0]).days / cfg.grid_days
    time_idx = int(np.clip(round(offset), 0, len(grid) - 1))

    if poll.race_id == NATIONAL_RACE_ID:
        latent = posterior["eta"].isel(grid=time_idx).to_numpy().ravel()
    else:
        idx = race_index.get(poll.race_id)
        if idx is None:
            return None
        latent = posterior["theta"].isel(race=idx, grid=time_idx).to_numpy().ravel()

    # House effect: use the pollster's own if the training window earned it one,
    # otherwise the pooled bucket, otherwise zero. A held-out poll from an
    # unseen pollster must not borrow someone else's bias.
    from .model.design import POOLED_POLLSTER

    names = list(data.pollster_names)
    key = poll.pollster if poll.pollster in names else POOLED_POLLSTER
    if key in names:
        house = posterior["house_effect"].isel(pollster=names.index(key)).to_numpy().ravel()
    else:
        house = np.zeros_like(latent)

    pops = list(data.populations)
    pop_idx = pops.index(poll.population) if poll.population in pops else data.reference_population_index
    pop_effect = posterior["population_effect"].to_numpy().reshape(-1, len(pops))[:, pop_idx]

    partisan = posterior["partisan_effect"].to_numpy().ravel() * poll.partisan_sign
    sigma_excess = posterior["sigma_excess"].to_numpy().ravel()

    from .model.design import _sampling_variance

    sampling_var = _sampling_variance(poll, cfg.polls.design_effect)
    sigma = np.sqrt(sampling_var + sigma_excess**2)
    # Mirror the likelihood exactly, variance correction included, or the
    # coverage numbers describe a distribution the model never used.
    if cfg.polls.match_student_t_variance:
        sigma = sigma / np.sqrt(cfg.polls.student_t_nu / (cfg.polls.student_t_nu - 2.0))

    mu = latent + house + pop_effect + partisan

    # Match the likelihood: Student-t, not Gaussian. The generator is supplied
    # by the caller and advanced across polls — re-seeding it here would give
    # every poll the identical set of t draws, correlating the Monte Carlo error
    # across the whole holdout set and making the coverage estimate noisier.
    nu = cfg.polls.student_t_nu
    t_draws = rng.standard_t(nu, size=mu.shape)
    return mu + sigma * t_draws


def calibrate(
    holdout_days: int = 30,
    races: RaceSet | None = None,
    cfg: ModelConfig | None = None,
    as_of: date | None = None,
    offline: bool = True,
) -> CalibrationResult:
    """Fit on polls before a cutoff and score the polls after it."""
    if races is None or cfg is None:
        races, cfg = load_all()
    roster = Roster.load()
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=holdout_days)

    snapshot = latest_snapshot()
    if snapshot is None or not offline:
        raw = VoteHubClient().fetch_and_snapshot(["us-senator", "generic-ballot", "approval"])
    else:
        raw = load_snapshot(snapshot)

    train_table = build_poll_table(raw, races, cfg, roster, as_of=cutoff)
    full_table = build_poll_table(raw, races, cfg, roster, as_of=as_of)

    train_ids = {p.poll_id for p in train_table.polls}
    holdout = [p for p in full_table.polls if p.poll_id not in train_ids]
    if not holdout:
        raise ValueError(
            f"no polls fielded in the last {holdout_days} days; nothing to score"
        )

    log.info(
        "training on %d polls, scoring %d held out after %s",
        len(train_table.polls), len(holdout), cutoff,
    )

    fund = F.compute(races, cfg)
    data = build_model_data(train_table, races, cfg, fund)
    model = build_model(data, cfg)
    idata = sample(model, cfg, progressbar=False)

    race_index = {rid: i for i, rid in enumerate(data.race_ids)}
    grid = data.grid_dates

    pit_values: list[float] = []
    abs_errors: list[float] = []
    inside = {level: 0 for level in CREDIBLE_LEVELS}
    scored = 0
    rng = np.random.default_rng(cfg.sampler.seed + 17)

    for poll in holdout:
        draws = _posterior_predictive_for_poll(
            poll, idata, data, cfg, race_index, grid, rng
        )
        if draws is None:
            continue
        observed = F.logit(poll.two_party_dem)

        pit_values.append(float(np.mean(draws <= observed)))
        abs_errors.append(
            abs(F.logit_to_margin(np.median(draws)) - poll.margin)
        )
        for level in CREDIBLE_LEVELS:
            lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
            if lo <= observed <= hi:
                inside[level] += 1
        scored += 1

    pit = np.asarray(pit_values)
    # One-sample KS statistic against Uniform(0, 1).
    ordered = np.sort(pit)
    n = len(ordered)
    empirical = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(empirical - ordered))) if n else float("nan")

    return CalibrationResult(
        n_holdout=scored,
        cutoff=cutoff,
        coverage={level: inside[level] / scored for level in CREDIBLE_LEVELS},
        pit=pit,
        mean_abs_error_margin=float(np.mean(abs_errors)),
        median_abs_error_margin=float(np.median(abs_errors)),
        ks_statistic=ks,
    )


def run_backtest(holdout_days: int = 30, verbose: bool = False) -> int:
    """CLI entry point. Returns a process exit code."""
    try:
        result = calibrate(holdout_days=holdout_days)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    print()
    print("=" * 72)
    print("  CALIBRATION: predicting held-out polls")
    print("=" * 72)
    print(result.report())
    print("=" * 72)
    return 0
