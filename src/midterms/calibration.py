"""Fit the model's error scales from historical polling.

Before this module existed, the two numbers that set the width of the entire
seat distribution — ``election_day_error.national_sd`` and ``.state_sd`` — were
asserted from general knowledge of the literature. They are the most
consequential numbers the model has: they decide how confident the headline
control probability is allowed to be, and nothing in the current cycle can
check them, because the election has not happened.

``data/history/raw_polls.csv.gz`` can check them. It pairs every poll
FiveThirtyEight ever collected with the actual result of the race, so poll
error is directly observable across 1998-2022.

THE DECOMPOSITION
-----------------
A poll's error decomposes into three levels, and the model needs them
separately because they behave completely differently in a seat forecast:

    national   a miss shared by every race in a cycle. Perfectly correlated,
               so it does NOT average out across 35 races — it moves the whole
               chamber together and is what fattens the tails.
    state      a miss specific to one race, correlated only with similar
               states. Partly cancels across the map.
    poll       noise in an individual poll. Averages away as polls accumulate,
               and is what the measurement model's sampling variance covers.

Getting the *split* wrong matters as much as getting the total wrong: shifting
error from the state term to the national term makes the seat distribution
wider without changing any single race's win probability.

The subtlety is that a race's mean poll error is not its true race-level error:
it still contains poll noise divided by the number of polls. Races polled once
therefore look far more erratic than races polled twenty times. This module
subtracts that leakage explicitly rather than letting it inflate the state
term, which it otherwise does by around 10%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

log = logging.getLogger(__name__)

HISTORY_PATH = paths.DATA_DIR / "history" / "raw_polls.csv.gz"

ATTRIBUTION = (
    "Historical poll errors from FiveThirtyEight's pollster-ratings dataset "
    "(github.com/fivethirtyeight/data), licensed CC BY 4.0."
)

#: Points of margin per logit unit, near a competitive race. See METHODOLOGY §1.
POINTS_PER_LOGIT = 50.0

#: A cycle needs this many races before its mean error is a meaningful estimate
#: of that cycle's national bias. Off-year specials with one or two races
#: otherwise dominate the cycle-to-cycle spread.
MIN_RACES_PER_CYCLE = 5

#: The dataset's race-type code per chamber. Matched exactly, not by substring.
#:
#: `str.contains("Sen")` also matches `Sen-P`, so 12-17 Senate *primary* polls
#: were being used to calibrate general-election error. The effect was small
#: (national 3.27 -> 3.32 points at 0-14 days, under 2%) but a primary is a
#: different contest with different polling and does not belong here.
#:
#: Exact matching matters more for the House, where `contains("House")` would
#: sweep in `House-G-US` -- the *national* House vote, one observation per cycle
#: rather than one per district. That would put a national aggregate into the
#: per-race pool and quietly corrupt the split between the two components.
RACE_TYPE = {"senate": "Sen-G", "house": "House-G"}

#: Which window each error component should be fitted in.
#:
#: 0-14 days, not the forecast horizon. See the long note on
#: `election_day_error` in config/model.yaml: fitting at 45-120 days folds
#: roughly 3.9 points of drift into the election-day term, which the latent
#: random walk already carries and which would then stay just as wide on the eve
#: of the election. This constant exists so the reason is attached to the number.
ELECTION_DAY_WINDOW = (0, 14)

#: Where the *design effect* is fitted, which is deliberately not the same.
#:
#: The design effect describes how much noisier polls are than their sample size
#: implies, at the horizon where the model actually runs and weighs them. The
#: election-day scales describe the terminal miss and must exclude drift. Fitting
#: both in one window gets one of them wrong: the Senate's design effect is 1.18
#: at the horizon and 1.36 on the eve, the House's 1.04 and 0.95.
#:
#: So the design effect is always measured here, whatever window the caller asks
#: for, and the report says so. Otherwise one `midterms calibrate` run could not
#: produce all four config numbers on their correct bases.
HORIZON_WINDOW = (45, 120)


@dataclass(frozen=True)
class ErrorComponents:
    """Fitted error scales, in points of margin."""

    national_sd: float
    state_sd: float
    poll_sd: float
    design_effect: float

    n_polls: int
    n_races: int
    n_cycles: int
    window: tuple[int, int]
    min_cycle: int
    chamber: str = "senate"
    design_effect_window: tuple[int, int] = HORIZON_WINDOW

    @property
    def total_race_sd(self) -> float:
        """Combined national + state error on a single race."""
        return float(np.hypot(self.national_sd, self.state_sd))

    def as_logit(self) -> dict[str, float]:
        """The same scales in the model's units."""
        return {
            "national_sd": round(self.national_sd / POINTS_PER_LOGIT, 4),
            "state_sd": round(self.state_sd / POINTS_PER_LOGIT, 4),
        }

    def report(self) -> str:
        lines = [
            f"Fitted on {self.n_polls:,} {self.chamber.capitalize()} polls / "
            f"{self.n_races} races / "
            f"{self.n_cycles} cycles",
            f"  window: {self.window[0]}-{self.window[1]} days before the election, "
            f"cycles from {self.min_cycle}",
            "",
            "Error components (points of margin):",
            f"  national (cycle-wide, perfectly correlated) : {self.national_sd:5.2f}",
            f"  state    (race-specific)                    : {self.state_sd:5.2f}",
            f"  -> total race-level error                   : {self.total_race_sd:5.2f}",
            "",
            f"  poll-level noise (within a race)            : {self.poll_sd:5.2f}",
            f"  implied design effect vs binomial           : {self.design_effect:5.2f}",
            f"     (fitted at {self.design_effect_window[0]}-"
            f"{self.design_effect_window[1]} days, where the model weighs polls,"
            f" not at the window above)",
            "",
            "In model units (logit):",
            f"  election_day_error.national_sd: {self.as_logit()['national_sd']}",
            f"  election_day_error.state_sd:    {self.as_logit()['state_sd']}",
            f"  polls.design_effect:            {round(self.design_effect, 2)}",
        ]
        return "\n".join(lines)


def load_history(path: Path | None = None) -> pd.DataFrame:
    """Historical polls with a known result."""
    path = path or HISTORY_PATH
    return pd.read_csv(path)


def race_polls(
    df: pd.DataFrame,
    chamber: str = "senate",
    days_window: tuple[int, int] = (45, 120),
    min_cycle: int = 2010,
) -> pd.DataFrame:
    """General-election polls for one chamber, inside a window.

    The window matters, and which window is right depends on what is being
    estimated. Poll error shrinks as an election approaches: fitting the
    election-day term at 45-120 days folds in drift the random walk already
    carries, so :data:`ELECTION_DAY_WINDOW` is what the config is fitted on.
    """
    try:
        race_type = RACE_TYPE[chamber]
    except KeyError:
        raise ValueError(
            f"unknown chamber {chamber!r}; expected one of {sorted(RACE_TYPE)}"
        ) from None

    sub = df[df["type_simple"].astype(str) == race_type].copy()
    sub["error"] = sub["margin_poll"] - sub["margin_actual"]
    sub = sub.dropna(subset=["error", "time_to_election", "cycle"])
    sub = sub[
        (sub["time_to_election"] >= days_window[0])
        & (sub["time_to_election"] <= days_window[1])
        & (sub["cycle"] >= min_cycle)
    ]
    race_counts = sub.groupby("cycle")["race_id"].nunique()
    keep = race_counts[race_counts >= MIN_RACES_PER_CYCLE].index
    return sub[sub["cycle"].isin(keep)]


def senate_polls(
    df: pd.DataFrame,
    days_window: tuple[int, int] = (45, 120),
    min_cycle: int = 2010,
) -> pd.DataFrame:
    """Senate general-election polls. Kept as the name callers already use."""
    return race_polls(df, "senate", days_window, min_cycle)


def _design_effect(sen: pd.DataFrame) -> float:
    """How much noisier polls are than a simple random sample implies.

    Compares the observed spread of polls within a race against the binomial
    spread their sample sizes predict. The excess is everything the sampling
    formula does not know about: weighting, clustering, house effects, and real
    opinion movement inside the window. Reported as a variance ratio, which is
    exactly how `polls.design_effect` is used in the measurement model.
    """
    usable = sen.dropna(subset=["samplesize"])
    usable = usable[usable["samplesize"] > 0]

    observed, expected = [], []
    for _, group in usable.groupby(["cycle", "race_id"]):
        if len(group) < 3:
            continue  # a variance from two polls is far too noisy to pool
        observed.append(float(np.var(group["margin_poll"], ddof=1)))
        # Binomial variance of a margin at p=0.5: (2 * 100)^2 * 0.25 / n = 10000/n
        expected.append(float(np.mean(10000.0 / group["samplesize"])))

    if not observed:
        return float("nan")
    return float(np.sum(observed) / np.sum(expected))


def estimate_error_components(
    days_window: tuple[int, int] = (45, 120),
    min_cycle: int = 2010,
    path: Path | None = None,
    chamber: str = "senate",
) -> ErrorComponents:
    """Fit national, unit and poll-level error scales for one chamber."""
    sen = race_polls(load_history(path), chamber, days_window, min_cycle)
    if sen.empty:
        raise ValueError(
            f"no historical {chamber} polls matched the given window"
        )

    grouped = sen.groupby(["cycle", "race_id"])["error"]
    race = pd.DataFrame(
        {
            "mean_error": grouped.mean(),
            "n_polls": grouped.size(),
            "within_var": grouped.var(ddof=1),
        }
    ).reset_index()

    # Poll-level variance, pooled over races with more than one poll.
    multi = race[race["n_polls"] > 1].dropna(subset=["within_var"])
    poll_var = float(
        np.average(multi["within_var"], weights=multi["n_polls"] - 1)
    ) if len(multi) else 0.0

    # National: how much whole cycles miss by, relative to each other.
    cycle_mean = race.groupby("cycle")["mean_error"].mean()
    national_sd = float(cycle_mean.std(ddof=1))

    # State: spread of race means about their cycle mean, with the poll noise
    # each race mean still carries (poll_var / n_polls) removed.
    residual = race["mean_error"] - race["cycle"].map(cycle_mean)
    weights = race["n_polls"]
    observed_var = float(np.average(residual**2, weights=weights))
    leakage = float(np.average(poll_var / race["n_polls"], weights=weights))
    state_var = max(observed_var - leakage, 0.0)

    components = ErrorComponents(
        national_sd=national_sd,
        state_sd=float(np.sqrt(state_var)),
        poll_sd=float(np.sqrt(poll_var)),
        design_effect=_design_effect(
            race_polls(load_history(path), chamber, HORIZON_WINDOW, min_cycle)
        ),
        design_effect_window=HORIZON_WINDOW,
        n_polls=len(sen),
        n_races=len(race),
        n_cycles=int(sen["cycle"].nunique()),
        window=days_window,
        min_cycle=min_cycle,
        chamber=chamber,
    )
    log.info(
        "calibration: national %.2f, state %.2f, poll %.2f pts (design effect %.2f)",
        components.national_sd, components.state_sd,
        components.poll_sd, components.design_effect,
    )
    return components


@dataclass(frozen=True)
class DriftRate:
    """How fast a race's standing moves, per day, in logit units."""

    per_day_logit: float
    near_window: tuple[int, int]
    far_window: tuple[int, int]
    near_sd_pts: float
    far_sd_pts: float
    implied_drift_pts: float
    days_between: float

    def report(self) -> str:
        return "\n".join([
            f"  error {self.near_window[0]}-{self.near_window[1]} days out : "
            f"{self.near_sd_pts:5.2f} pts  (mostly systematic)",
            f"  error {self.far_window[0]}-{self.far_window[1]} days out: "
            f"{self.far_sd_pts:5.2f} pts",
            f"  => drift accumulated over ~{self.days_between:.0f} days: "
            f"{self.implied_drift_pts:5.2f} pts",
            f"  => per-day drift SD: {self.per_day_logit:.5f} logit "
            f"({self.per_day_logit * POINTS_PER_LOGIT:.3f} pts)",
        ])


def estimate_drift_rate(
    near_window: tuple[int, int] = (0, 14),
    far_window: tuple[int, int] = (45, 120),
    min_cycle: int = 2010,
) -> DriftRate:
    """Separate genuine opinion movement from systematic polling error.

    Poll error against the eventual result shrinks as an election approaches.
    The part that shrinks is *drift* — the race actually moving. The part that
    remains is the systematic miss the polls were always going to make.

    Splitting them matters because the model represents them in different
    places: drift belongs to the random walk, which grows with the time left,
    while the systematic component is added at simulation time and does not
    shrink at all. Folding drift into the second term makes a forecast issued
    three days out exactly as uncertain as one issued three months out, which
    is plainly wrong.
    """
    near = estimate_error_components(near_window, min_cycle)
    far = estimate_error_components(far_window, min_cycle)

    drift_pts = float(np.sqrt(max(far.total_race_sd**2 - near.total_race_sd**2, 0.0)))
    # Midpoint-to-midpoint, which is where each window's error is centred.
    days = (sum(far_window) / 2.0) - (sum(near_window) / 2.0)
    per_day = (drift_pts / POINTS_PER_LOGIT) / np.sqrt(max(days, 1.0))

    return DriftRate(
        per_day_logit=float(per_day),
        near_window=near_window,
        far_window=far_window,
        near_sd_pts=near.total_race_sd,
        far_sd_pts=far.total_race_sd,
        implied_drift_pts=drift_pts,
        days_between=float(days),
    )


@dataclass(frozen=True)
class ExcessNoise:
    """Poll-to-poll scatter that sampling error does not explain."""

    excess_sd_logit: float
    excess_sd_pts: float
    size_exponent: float
    n_polls: int
    n_races: int

    def report(self) -> str:
        return "\n".join([
            f"  fitted on {self.n_polls:,} polls across {self.n_races} races",
            f"  scatter scales with sample size as n^{self.size_exponent:+.3f} "
            f"(pure sampling theory: -0.500)",
            f"  => non-sampling noise, near-constant in n: "
            f"{self.excess_sd_pts:.2f} pts ({self.excess_sd_logit:.4f} logit)",
        ])


def estimate_excess_noise(
    days_window: tuple[int, int] = (0, 60),
    min_cycle: int = 2010,
    min_polls_per_race: int = 5,
) -> ExcessNoise:
    """Measure the poll noise that a bigger sample does not buy away.

    Sampling theory says a poll's error shrinks as 1/sqrt(n). Real polls do not
    behave that way: house effects, mode, likely-voter modelling and question
    wording contribute error that is the same size whatever the sample. Fitting
    the exponent on within-race scatter gives roughly -0.18 rather than -0.5.

    The practical consequence is that a model weighting purely by 1/n treats a
    2,000-person poll as far more informative than it is. This term is what
    stops that.
    """
    sen = senate_polls(load_history(), days_window, min_cycle).copy()
    sen = sen.dropna(subset=["samplesize"])
    sen = sen[sen["samplesize"].between(200, 5000)]

    grouped = sen.groupby(["cycle", "race_id"])["margin_poll"]
    sen["race_mean"] = grouped.transform("mean")
    sen["race_n"] = grouped.transform("size")
    sen = sen[sen["race_n"] >= min_polls_per_race]
    # A poll contributes to its own race mean, which shrinks its deviation.
    sen["deviation"] = (sen["margin_poll"] - sen["race_mean"]) * np.sqrt(
        sen["race_n"] / (sen["race_n"] - 1)
    )

    bins = [(200, 500), (500, 700), (700, 900), (900, 1400), (1400, 5000)]
    sizes, spreads, weights, excesses = [], [], [], []
    for low, high in bins:
        band = sen[sen["samplesize"].between(low, high)]
        if len(band) < 25:
            continue
        observed = float(band["deviation"].std(ddof=1))
        median_n = float(band["samplesize"].median())
        # Binomial SD of a margin at p = 0.5 is 2 * 100 * sqrt(0.25 / n).
        binomial = 200.0 * np.sqrt(0.25 / median_n)
        sizes.append(median_n)
        spreads.append(observed)
        weights.append(np.sqrt(len(band)))
        excesses.append(np.sqrt(max(observed**2 - binomial**2, 0.0)))

    exponent = float(
        np.polyfit(np.log(sizes), np.log(spreads), 1, w=weights)[0]
    ) if len(sizes) >= 2 else float("nan")
    excess_pts = float(np.median(excesses)) if excesses else 0.0

    return ExcessNoise(
        excess_sd_logit=excess_pts / POINTS_PER_LOGIT,
        excess_sd_pts=excess_pts,
        size_exponent=exponent,
        n_polls=len(sen),
        n_races=int(sen.groupby(["cycle", "race_id"]).ngroups),
    )


def compare_to_config(components: ErrorComponents, chamber: str = "senate") -> str:
    """Fitted scales against what config/model.yaml assumes for this chamber."""
    from .config import ModelConfig

    cfg = ModelConfig.load(chamber=chamber)
    current_national = cfg.election_day_error.national_sd * POINTS_PER_LOGIT
    current_state = cfg.election_day_error.state_sd * POINTS_PER_LOGIT
    current_total = float(np.hypot(current_national, current_state))

    def delta(fitted: float, current: float) -> str:
        if current == 0:
            return "n/a"
        return f"{fitted / current - 1:+.0%}"

    rows = [
        ("national_sd", current_national, components.national_sd),
        ("state_sd", current_state, components.state_sd),
        ("total race error", current_total, components.total_race_sd),
        ("design_effect", cfg.polls.design_effect, components.design_effect),
    ]
    lines = [f"  {'':18s} {'config':>8s} {'fitted':>8s} {'change':>8s}"]
    for name, current, fitted in rows:
        lines.append(f"  {name:18s} {current:8.2f} {fitted:8.2f} {delta(fitted, current):>8s}")
    return "\n".join(lines)


def run_calibration(
    days_window: tuple[int, int] = (45, 120),
    min_cycle: int = 2010,
    chamber: str = "senate",
) -> int:
    """CLI entry point. Returns a process exit code."""
    components = estimate_error_components(
        days_window, min_cycle, chamber=chamber
    )
    print()
    print("=" * 72)
    print(f"  CALIBRATION against historical {chamber.capitalize()} polling")
    print("=" * 72)
    print(components.report())
    print()
    print("Against the current configuration:")
    print(compare_to_config(components, chamber))
    print()
    print(f"  {ATTRIBUTION}")
    print("=" * 72)
    return 0
