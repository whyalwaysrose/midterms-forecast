"""State demographics, for the optional demographic correlation covariates.

These are **off by default**, and the reason is a measurement rather than a
preference. `scripts/fit_demographic_covariates.py` scored them on the same 13
cycles and 135 state pairs that validated the kernel's political covariates:
education lowers agreement with the measured error ranking (+0.173 to +0.102,
because BA+ share is +0.79 collinear with the 2024 lean the kernel already
carries), and residualised urban share, the only specification that helps,
gains +0.005 while winning 62% of bootstrap resamples with a 90% interval of
-0.108 to +0.115. See METHODOLOGY 5 and known weakness 1.

They are implemented anyway, behind `election_day_error.correlation`'s
`demographic_weight`, because the measurement has no power in the cycles where
the education realignment is claimed (14 usable pairs from 2012, none from 2016)
and none at all at district level. Someone preferring a theoretically motivated
kernel to a noise-limited measurement can turn them on; the default encodes what
was measured, not what is plausible.

Turning them on has a consequence worth knowing in advance: each added column
adds a dimension to the distance, pushing every pair further apart and shrinking
the kernel toward independence. At weight 1.5 with ``urban``, mean off-diagonal
correlation falls from +0.314 to +0.181 on the House. Because independent error
averages out across 435 seats and correlated error does not, that *narrows* the
seat interval -- the exact artefact METHODOLOGY 11 and
``scripts/fit_correlation.py`` finding 3 warn against mistaking for accuracy. A
``length_scale`` re-fit belongs with any such change.

The numbers are public-domain US Census, derived into
``data/history/state_demographics.csv`` by that script's ``--refresh``. See
DATA_SOURCES.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import paths

#: Column name in the vendored CSV for each covariate the config may name.
COVARIATE_COLUMNS = {
    "education": "ba_plus_pct",
    "urban": "urban_pct",
}

DEMOGRAPHICS_CSV = paths.DATA_DIR / "history" / "state_demographics.csv"


class DemographicsError(RuntimeError):
    """Raised rather than silently dropping a race from a covariate.

    A missing value has to stop the run. Filling it with a mean or a zero would
    place that race at the centre of the demographic axis, which reads to the
    kernel as "average on this dimension" -- a claim about the race that nobody
    made, affecting how strongly its election-day error correlates with every
    other race's.
    """


def state_of_unit(unit: str) -> str:
    """Postal code for a race's unit.

    Senate units are already states (``GA``). House units are districts
    (``AZ-01``), and the first two characters are the state.
    """
    return unit[:2].upper()


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> dict[str, dict[str, float]]:
    """``{postal: {covariate: value}}`` from the vendored CSV."""
    path = path or DEMOGRAPHICS_CSV
    if not path.exists():
        raise DemographicsError(
            f"{path} is missing. Regenerate it with "
            "`python scripts/fit_demographic_covariates.py --refresh`, or set "
            "election_day_error.correlation.demographic_weight to 0 to run "
            "without demographic covariates."
        )

    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [
            column for column in COVARIATE_COLUMNS.values()
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise DemographicsError(f"{path} has no column(s) {missing}")
        for row in reader:
            out[row["state"].strip().upper()] = {
                name: float(row[column])
                for name, column in COVARIATE_COLUMNS.items()
            }
    return out


def values_for(units: tuple[str, ...], covariate: str) -> np.ndarray:
    """One covariate, aligned to ``units``, in the order given.

    For the House this gives every district in a state the same value, so the
    covariate adds separation *between* states and none within one. That is a
    real limitation and not a bug to be papered over: the ACS publishes nothing
    for the 181 districts whose 2026 boundaries have never held an election
    (METHODOLOGY 9.0), so district-level values would have to be apportioned
    from tracts, and the error would land on exactly those seats.
    """
    if covariate not in COVARIATE_COLUMNS:
        raise DemographicsError(
            f"unknown demographic covariate {covariate!r}; "
            f"known: {sorted(COVARIATE_COLUMNS)}"
        )
    table = load()
    values, unknown = [], []
    for unit in units:
        state = state_of_unit(unit)
        row = table.get(state)
        if row is None:
            unknown.append(unit)
            values.append(np.nan)
        else:
            values.append(row[covariate])
    if unknown:
        raise DemographicsError(
            f"no demographics for {sorted(set(unknown))} (resolved to states "
            f"{sorted({state_of_unit(u) for u in unknown})}). Add them to "
            f"{DEMOGRAPHICS_CSV.name} or disable the demographic covariates."
        )
    return np.asarray(values, dtype=float)


def residualise_against(target: np.ndarray, *controls: np.ndarray) -> np.ndarray:
    """``target`` with the controls' linear explanation removed.

    The part of a demographic covariate the kernel does not already carry.
    Without this, appending a column +0.79 correlated with the 2024 lean mostly
    stretches an axis the kernel has rather than adding a new one -- which the
    measurement shows is worse than adding nothing.
    """
    design = np.column_stack([np.ones(len(target)), *controls])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ beta
