"""Canonical filesystem layout.

Every path the pipeline touches is derived from :data:`REPO_ROOT`, so the whole
project relocates cleanly and CI needs no path configuration.
"""

from __future__ import annotations

from pathlib import Path

# src/midterms/paths.py -> src/midterms -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "config"
RACES_SENATE_2026 = CONFIG_DIR / "races_senate_2026.yaml"
RACES_HOUSE_2026 = CONFIG_DIR / "races_house_2026.yaml"

#: Race config per chamber, so callers name a chamber rather than a path.
RACES_BY_CHAMBER = {
    "senate": RACES_SENATE_2026,
    "house": RACES_HOUSE_2026,
}
MODEL_CONFIG = CONFIG_DIR / "model.yaml"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORY_DIR = DATA_DIR / "history"
RATINGS_DIR = DATA_DIR / "ratings"
RATINGS_FILE = RATINGS_DIR / "pollster-ratings.csv"

OUTPUTS_DIR = REPO_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
TRACES_DIR = OUTPUTS_DIR / "traces"

SITE_DIR = REPO_ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        RUNS_DIR,
        TRACES_DIR,
        SITE_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def run_dir(run_date: str) -> Path:
    """Directory holding the archived artefacts for a single dated run."""
    return RUNS_DIR / run_date


def chamber_filename(stem: str, chamber: str) -> str:
    """File name for a chamber's copy of a site data file.

    The Senate keeps the unsuffixed names it has always had -- ``forecast.json``,
    ``history.json`` -- and other chambers are suffixed. That asymmetry is
    deliberate: the published site, every archived run going back to the start
    of the project, and any link anyone has already saved all point at the
    unsuffixed names. Renaming them to gain symmetry would break all of that to
    fix nothing.
    """
    return f"{stem}.json" if chamber == "senate" else f"{stem}_{chamber}.json"
