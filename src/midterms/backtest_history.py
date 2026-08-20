"""Score the forecast against elections that actually happened.

``backtest.py`` asks whether the model predicts *polls* it has not seen. That is
a necessary check and a weak one: a model can predict polls beautifully and
still be wrong about the election, because polls themselves are biased. This
module asks the question that matters — **do the win probabilities come true at
the rate they claim?** — using 2010-2022 Senate races, where the answer is known.

WHAT IS AND IS NOT TESTED
-------------------------
The point estimate here is a recency-weighted poll average, not the full PyMC
posterior. That is deliberate. The quantity under test is the **error model**:
the national and state scales in ``election_day_error``, which set how wide the
seat distribution is and therefore how confident the headline number is allowed
to be. Those scales, not the aggregation, are what a reliability curve can
falsify — and they are the numbers that were asserted from the literature until
``midterms calibrate`` fitted them.

The full model's aggregation is better than a weighted average (house effects,
partial pooling, a latent trend), so treat the point-estimate error reported
here as an upper bound on the real model's.

TWO LEVELS
----------
Per race, correlation is irrelevant — only the marginal error width decides
whether a 70% call comes true 70% of the time. Across a chamber it is
everything, because correlated misses do not average out. Both are scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import POINTS_PER_LOGIT, load_history, senate_polls

log = logging.getLogger(__name__)

#: Recency half-life for the poll average, in days. Polls a fortnight apart
#: carry meaningfully different information this far from an election.
RECENCY_HALFLIFE_DAYS = 21.0


@dataclass
class BacktestResult:
    n_races: int
    n_cycles: int
    brier: float
    brier_skill_vs_naive: float
    log_score: float
    reliability: pd.DataFrame
    coverage: dict[float, float]
    point_estimate_mae: float
    seat_z_scores: list[float] = field(default_factory=list)
    label: str = ""

    def report(self) -> str:
        lines = [
            f"[{self.label}]  {self.n_races} races across {self.n_cycles} cycles",
            f"  point-estimate MAE (poll average vs result) : {self.point_estimate_mae:5.2f} pts",
            f"  Brier score (lower is better)               : {self.brier:.4f}",
            f"  skill vs always-predicting-the-favourite    : {self.brier_skill_vs_naive:+.1%}",
            f"  mean log score                              : {self.log_score:.4f}",
            "",
            "  Interval coverage of the actual margin:",
        ]
        for level, hit in sorted(self.coverage.items()):
            gap = hit - level
            verdict = (
                "well calibrated" if abs(gap) <= 0.06
                else "TOO WIDE" if gap > 0
                else "TOO NARROW"
            )
            lines.append(f"    {level:.0%} -> {hit:5.1%}   {verdict}")
        lines.append("")
        lines.append("  Reliability (predicted vs actual win rate):")
        for _, row in self.reliability.iterrows():
            lines.append(
                f"    predicted {row['bin']:>9s}  n={int(row['n']):3d}  "
                f"mean p {row['mean_pred']:.2f}  actual {row['actual']:.2f}"
            )
        if self.seat_z_scores:
            z = np.array(self.seat_z_scores)
            lines.append("")
            lines.append(
                f"  Chamber-level: seat-total z-scores across cycles, "
                f"mean {z.mean():+.2f}, sd {z.std(ddof=1):.2f} (want ~0 and ~1)"
            )
        return "\n".join(lines)


def _weighted_poll_average(group: pd.DataFrame) -> float:
    """Recency- and precision-weighted mean poll margin for one race."""
    recency = np.exp(
        -(group["time_to_election"] - group["time_to_election"].min())
        / RECENCY_HALFLIFE_DAYS
    )
    size = group["samplesize"].fillna(600).clip(lower=100)
    weight = recency * np.sqrt(size)
    return float(np.average(group["margin_poll"], weights=weight))


def build_race_table(days_window=(45, 120), min_cycle=2010) -> pd.DataFrame:
    """One row per historical race: poll-based estimate and actual result."""
    sen = senate_polls(load_history(), days_window, min_cycle)
    rows = []
    for (cycle, race_id), group in sen.groupby(["cycle", "race_id"]):
        rows.append(
            {
                "cycle": cycle,
                "race_id": race_id,
                "location": group["location"].iloc[0],
                "n_polls": len(group),
                "predicted": _weighted_poll_average(group),
                "actual": float(group["margin_actual"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _simulate(
    races: pd.DataFrame,
    national_sd_pts: float,
    state_sd_pts: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulated final margins, shape (n_sims, n_races).

    The national term is one draw shared by every race in a simulation, which is
    what makes chamber-level outcomes correlated; the state term is independent
    per race. Correlation *between* similar states is not modelled here, since
    the historical table has no covariates attached — so the seat-level check
    below is, if anything, generous to the model.
    """
    national = rng.normal(0.0, national_sd_pts, size=(n_sims, 1))
    state = rng.normal(0.0, state_sd_pts, size=(n_sims, len(races)))
    return races["predicted"].to_numpy()[None, :] + national + state


def score(
    national_sd_pts: float,
    state_sd_pts: float,
    label: str,
    races: pd.DataFrame | None = None,
    n_sims: int = 20000,
    seed: int = 20261103,
) -> BacktestResult:
    """Score one error-model configuration against history."""
    races = build_race_table() if races is None else races
    rng = np.random.default_rng(seed)
    simulated = _simulate(races, national_sd_pts, state_sd_pts, n_sims, rng)

    win_prob = (simulated > 0).mean(axis=0)
    actual_win = (races["actual"].to_numpy() > 0).astype(float)

    brier = float(np.mean((win_prob - actual_win) ** 2))
    # Baseline: call every race for whoever leads the polls, with certainty
    # bounded away from 0/1 so the log score stays finite.
    naive = np.clip((races["predicted"].to_numpy() > 0).astype(float), 0.05, 0.95)
    brier_naive = float(np.mean((naive - actual_win) ** 2))
    skill = 1.0 - brier / brier_naive if brier_naive else float("nan")

    safe = np.clip(win_prob, 1e-6, 1 - 1e-6)
    log_score = float(
        np.mean(actual_win * np.log(safe) + (1 - actual_win) * np.log(1 - safe))
    )

    coverage = {}
    for level in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(simulated, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
        coverage[level] = float(
            np.mean((races["actual"].to_numpy() >= lo) & (races["actual"].to_numpy() <= hi))
        )

    # Reliability: does a 70% call win about 70% of the time?
    edges = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    bins = []
    for lo_edge, hi_edge in zip(edges[:-1], edges[1:], strict=True):
        mask = (win_prob >= lo_edge) & (win_prob < hi_edge if hi_edge < 1 else win_prob <= 1)
        if mask.sum() == 0:
            continue
        bins.append(
            {
                "bin": f"{lo_edge:.0%}-{hi_edge:.0%}",
                "n": int(mask.sum()),
                "mean_pred": float(win_prob[mask].mean()),
                "actual": float(actual_win[mask].mean()),
            }
        )

    # Chamber level: within each cycle, how many seats did the simulation expect
    # against how many actually broke Democratic?
    seat_z = []
    for _cycle, group in races.groupby("cycle"):
        idx = races.index.get_indexer(group.index)
        seats = (simulated[:, idx] > 0).sum(axis=1)
        actual_seats = float((group["actual"] > 0).sum())
        sd = seats.std(ddof=1)
        if sd > 0:
            seat_z.append(float((actual_seats - seats.mean()) / sd))

    return BacktestResult(
        n_races=len(races),
        n_cycles=int(races["cycle"].nunique()),
        brier=brier,
        brier_skill_vs_naive=skill,
        log_score=log_score,
        reliability=pd.DataFrame(bins),
        coverage=coverage,
        point_estimate_mae=float((races["predicted"] - races["actual"]).abs().mean()),
        seat_z_scores=seat_z,
        label=label,
    )


def run_historical_backtest() -> int:
    """CLI entry point: score the fitted scales against the old asserted ones."""
    from .config import ModelConfig

    races = build_race_table()
    cfg = ModelConfig.load()
    fitted = (
        cfg.election_day_error.national_sd * POINTS_PER_LOGIT,
        cfg.election_day_error.state_sd * POINTS_PER_LOGIT,
    )

    print()
    print("=" * 78)
    print("  HISTORICAL BACKTEST — scored against elections that happened")
    print("=" * 78)

    configurations = [
        (fitted[0], fitted[1], "current config (fitted from history)"),
        (3.75, 4.25, "previous config (asserted from literature)"),
    ]
    for national_sd, state_sd, label in configurations:
        result = score(national_sd, state_sd, f"{label}: {national_sd:.2f}/{state_sd:.2f} pts", races)
        print()
        print(result.report())

    print()
    print("=" * 78)
    return 0
