"""Typed loading and validation of the YAML configuration.

The configs are the contract between "what we believe about the world" and
"what the model does with it". They are parsed into frozen dataclasses rather
than passed around as raw dicts so that a typo in a key fails loudly at load
time instead of silently becoming a wrong prior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from . import paths

PARTIES = ("D", "R")
INCUMBENT_STATUSES = ("elected", "appointed", "open")


class ConfigError(ValueError):
    """Raised when a config file is structurally invalid."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key {key!r} in {context}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Race:
    """A single contest."""

    id: str
    unit: str
    name: str
    special: bool
    incumbent_party: str
    incumbent_status: str
    pres_2024_dem_two_party: float
    pres_2020_dem_two_party: float
    region: str
    votehub_subject: str | None = None

    def subject_for(self, cycle: int) -> str:
        """The VoteHub ``subject`` string identifying this race's polls.

        VoteHub names general-election subjects ``"<cycle> <State>"`` (primary
        subjects get a trailing party name, which we exclude at fetch time).
        """
        return self.votehub_subject or f"{cycle} {self.name}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Race:
        race_id = _require(raw, "id", "race entry")
        ctx = f"race {race_id!r}"

        party = _require(raw, "incumbent_party", ctx)
        if party not in PARTIES:
            raise ConfigError(f"{ctx}: incumbent_party must be one of {PARTIES}, got {party!r}")

        status = _require(raw, "incumbent_status", ctx)
        if status not in INCUMBENT_STATUSES:
            raise ConfigError(
                f"{ctx}: incumbent_status must be one of {INCUMBENT_STATUSES}, got {status!r}"
            )

        for share_key in ("pres_2024_dem_two_party", "pres_2020_dem_two_party"):
            value = float(_require(raw, share_key, ctx))
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{ctx}: {share_key} must be a share strictly in (0, 1)")

        return cls(
            id=race_id,
            unit=_require(raw, "unit", ctx),
            name=_require(raw, "name", ctx),
            special=bool(_require(raw, "special", ctx)),
            incumbent_party=party,
            incumbent_status=status,
            pres_2024_dem_two_party=float(raw["pres_2024_dem_two_party"]),
            pres_2020_dem_two_party=float(raw["pres_2020_dem_two_party"]),
            region=_require(raw, "region", ctx),
            votehub_subject=raw.get("votehub_subject"),
        )


@dataclass(frozen=True)
class Control:
    """Chamber-control arithmetic: how seats won map to chamber control."""

    total_seats: int
    seats_not_up: Mapping[str, int]
    seats_up: Mapping[str, int]
    dem_seats_for_majority: int
    rep_seats_for_majority: int
    tiebreaker_party: str

    def validate(self, races: tuple[Race, ...]) -> None:
        """Check the seat arithmetic closes and matches the race list.

        This catches the most damaging class of silent error in a seat model:
        a miscounted chamber baseline, which shifts every control probability
        without changing any individual race.
        """
        not_up = sum(self.seats_not_up.values())
        up = sum(self.seats_up.values())
        if not_up + up != self.total_seats:
            raise ConfigError(
                f"seat arithmetic does not close: {not_up} not up + {up} up "
                f"!= {self.total_seats} total"
            )
        if up != len(races):
            raise ConfigError(
                f"control.seats_up totals {up} but {len(races)} races are defined"
            )
        for party in PARTIES:
            declared = self.seats_up.get(party, 0)
            actual = sum(1 for r in races if r.incumbent_party == party)
            if declared != actual:
                raise ConfigError(
                    f"control.seats_up[{party}] = {declared} but {actual} races "
                    f"have incumbent_party={party}"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Control:
        ctx = "control"
        return cls(
            total_seats=int(_require(raw, "total_seats", ctx)),
            seats_not_up={k: int(v) for k, v in _require(raw, "seats_not_up", ctx).items()},
            seats_up={k: int(v) for k, v in _require(raw, "seats_up", ctx).items()},
            dem_seats_for_majority=int(_require(raw, "dem_seats_for_majority", ctx)),
            rep_seats_for_majority=int(_require(raw, "rep_seats_for_majority", ctx)),
            tiebreaker_party=_require(raw, "tiebreaker_party", ctx),
        )


@dataclass(frozen=True)
class RaceSet:
    """Every contest in one chamber for one cycle, plus the control rules."""

    cycle: int
    chamber: str
    election_date: date
    control: Control
    races: tuple[Race, ...]

    def __post_init__(self) -> None:
        ids = [r.id for r in self.races]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ConfigError(f"duplicate race ids: {duplicates}")
        self.control.validate(self.races)

    @property
    def race_ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.races)

    def by_id(self, race_id: str) -> Race:
        for race in self.races:
            if race.id == race_id:
                return race
        raise KeyError(race_id)

    @classmethod
    def load(cls, path: Path | None = None) -> RaceSet:
        path = path or paths.RACES_SENATE_2026
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        election_date = _require(raw, "election_date", str(path))
        if not isinstance(election_date, date):
            election_date = date.fromisoformat(str(election_date))

        return cls(
            cycle=int(_require(raw, "cycle", str(path))),
            chamber=_require(raw, "chamber", str(path)),
            election_date=election_date,
            control=Control.from_dict(_require(raw, "control", str(path))),
            races=tuple(Race.from_dict(r) for r in _require(raw, "races", str(path))),
        )


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NationalRefs:
    pres_2024_dem_two_party: float
    pres_2020_dem_two_party: float


@dataclass(frozen=True)
class FundamentalsConfig:
    weight_pres_2024: float
    weight_pres_2020: float
    lean_shrinkage: float
    incumbency_bonus: Mapping[str, float]
    prior_sd: float

    def __post_init__(self) -> None:
        total = self.weight_pres_2024 + self.weight_pres_2020
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(
                f"fundamentals presidential weights must sum to 1, got {total}"
            )
        missing = set(INCUMBENT_STATUSES) - set(self.incumbency_bonus)
        if missing:
            raise ConfigError(f"fundamentals.incumbency_bonus missing statuses: {sorted(missing)}")


@dataclass(frozen=True)
class ApprovalConfig:
    enabled: bool
    rw_sd_per_day_prior: float
    initial_sd: float
    correlation_prior_sd: float


@dataclass(frozen=True)
class NationalEnvConfig:
    rw_sd_per_day_prior: float
    initial_sd: float
    approval: ApprovalConfig
    #: How far the generic ballot sits from the actual national vote, in logit.
    #:
    #: Zero for the Senate, deliberately: the gap was measured against the
    #: national *House* vote, and extending it to a chamber where it was never
    #: measured would be asserting rather than calibrating.
    generic_ballot_bias: float = 0.0
    #: Standard error of that estimate, folded into the election-day error so
    #: the correction carries its own uncertainty rather than pretending to be
    #: exact.
    generic_ballot_bias_se: float = 0.0


@dataclass(frozen=True)
class RaceStateConfig:
    elasticity_mean: float
    elasticity_sd: float
    elasticity_lower: float
    rw_sd_per_day_prior: float
    movement_correlation: CorrelationConfig


@dataclass(frozen=True)
class PollsterRatingsConfig:
    """Whether and how to weight polls by their pollster's track record."""

    enabled: bool
    exclude_banned: bool


@dataclass(frozen=True)
class PollsConfig:
    pollster_ratings: PollsterRatingsConfig
    student_t_nu: float
    match_student_t_variance: bool
    design_effect: float
    excess_sd_prior: float
    house_effect_sd_prior: float
    min_polls_for_house_effect: int
    population_effect_sd_prior: float
    population_reference: str
    partisan_effect_prior: float
    max_age_days: int
    min_sample_size: int
    default_sample_size: int


@dataclass(frozen=True)
class CorrelationConfig:
    length_scale: float
    region_weight: float
    nugget: float


@dataclass(frozen=True)
class ElectionDayErrorConfig:
    national_sd: float
    state_sd: float
    correlation: CorrelationConfig


@dataclass(frozen=True)
class SamplerConfig:
    draws: int
    tune: int
    chains: int
    target_accept: float
    seed: int
    backend: str


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively overlay `override` onto `base`, returning a new mapping.

    Leaf-by-leaf so a chamber can change one scale without restating the block
    it lives in. Restating a whole block would work today and quietly fork the
    moment the shared part -- the correlation kernel, say -- was edited in one
    place and not the other.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ModelConfig:
    national: NationalRefs
    fundamentals: FundamentalsConfig
    national_environment: NationalEnvConfig
    race: RaceStateConfig
    polls: PollsConfig
    election_day_error: ElectionDayErrorConfig
    sampler: SamplerConfig
    sims_per_draw: int
    grid_days: int
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None = None, chamber: str = "senate") -> ModelConfig:
        """Load the model config, applying any per-chamber overrides.

        The top-level values are the Senate's, fitted from Senate polling. A
        ``chambers:`` block may override individual leaves for another chamber --
        the House's polls are measurably noisier and its districts miss by more,
        and applying Senate numbers to them was an assumption nobody had chosen.

        Overrides are merged leaf by leaf rather than wholesale, so a chamber
        that differs in one scale does not have to restate the correlation
        kernel and silently fork it.
        """
        path = path or paths.MODEL_CONFIG
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ctx = str(path)

        overrides = (raw.get("chambers") or {}).get(chamber)
        if overrides:
            raw = _deep_merge(raw, overrides)

        ede = _require(raw, "election_day_error", ctx)
        return cls(
            national=NationalRefs(**_require(raw, "national", ctx)),
            fundamentals=FundamentalsConfig(**_require(raw, "fundamentals", ctx)),
            national_environment=_national_env_config(
                _require(raw, "national_environment", ctx)
            ),
            race=_race_state_config(_require(raw, "race", ctx)),
            polls=_polls_config(_require(raw, "polls", ctx)),
            election_day_error=ElectionDayErrorConfig(
                national_sd=float(_require(ede, "national_sd", "election_day_error")),
                state_sd=float(_require(ede, "state_sd", "election_day_error")),
                correlation=CorrelationConfig(
                    **_require(ede, "correlation", "election_day_error")
                ),
            ),
            sampler=SamplerConfig(**_require(raw, "sampler", ctx)),
            sims_per_draw=int(_require(raw, "simulation", ctx)["sims_per_draw"]),
            grid_days=int(_require(raw, "time", ctx)["grid_days"]),
            raw=raw,
        )


def _polls_config(raw: Mapping[str, Any]) -> PollsConfig:
    """Build PollsConfig, splitting out the nested ratings block."""
    values = dict(raw)
    ratings = values.pop("pollster_ratings", None)
    if ratings is None:
        raise ConfigError("polls.pollster_ratings is missing from config/model.yaml")
    return PollsConfig(
        pollster_ratings=PollsterRatingsConfig(**ratings), **values
    )


def _national_env_config(raw: Mapping[str, Any]) -> NationalEnvConfig:
    values = dict(raw)
    approval = values.pop("approval", None)
    if approval is None:
        raise ConfigError("national_environment.approval is required")
    return NationalEnvConfig(approval=ApprovalConfig(**approval), **values)


def _race_state_config(raw: Mapping[str, Any]) -> RaceStateConfig:
    values = dict(raw)
    correlation = values.pop("movement_correlation", None)
    if correlation is None:
        raise ConfigError("race.movement_correlation is required")
    return RaceStateConfig(movement_correlation=CorrelationConfig(**correlation), **values)


def load_all(
    races_path: Path | None = None,
    model_path: Path | None = None,
    chamber: str = "senate",
) -> tuple[RaceSet, ModelConfig]:
    """Load and cross-validate both configs.

    ``chamber`` selects which race file to read. The model itself is
    chamber-agnostic -- it takes whatever races it is given -- so this is the
    only place that has to know the House exists.
    """
    if races_path is None:
        try:
            races_path = paths.RACES_BY_CHAMBER[chamber]
        except KeyError:
            raise ConfigError(
                f"unknown chamber {chamber!r}; expected one of "
                f"{sorted(paths.RACES_BY_CHAMBER)}"
            ) from None
    return RaceSet.load(races_path), ModelConfig.load(model_path, chamber=chamber)
