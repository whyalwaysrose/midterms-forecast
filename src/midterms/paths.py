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
