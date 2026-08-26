"""Client for the VoteHub Polling API.

    https://api.votehub.com   —  docs at https://votehub.com/polls/api/

The API is free, requires no key, and its data is licensed CC BY 4.0. We are
required to attribute VoteHub; see the README and the dashboard footer.

Endpoints used here:

    GET /polls?poll_type=<type>[&subject=<subject>]
    GET /subjects
    GET /poll-types
    GET /pollsters

Every fetch is written to ``data/raw`` as a dated snapshot before any parsing
happens. That snapshot is what makes a run reproducible: re-running the model
against yesterday's file reproduces yesterday's forecast exactly, which is what
lets the commentary generator attribute a change to *the polls* rather than to
an unnoticed change in the feed.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .. import paths

log = logging.getLogger(__name__)

BASE_URL = "https://api.votehub.com"
ATTRIBUTION = "Poll data from VoteHub (https://votehub.com), licensed CC BY 4.0."
USER_AGENT = (
    "midterms-forecast/0.1 (open-source Bayesian election model; "
    "+https://github.com/) python-requests"
)

# Subjects for party primaries carry a trailing party name, e.g.
# "2026 Texas Democratic". We only ever want general-election matchups.
PRIMARY_SUFFIXES = (" Democratic", " Republican", " Democrat", " GOP")


class VoteHubError(RuntimeError):
    """Raised when the API cannot be reached or returns something unusable."""


def is_primary_subject(subject: str) -> bool:
    """True for a party-primary subject such as ``"2026 Texas Democratic"``."""
    return subject.endswith(PRIMARY_SUFFIXES)


@dataclass
class VoteHubClient:
    """Thin, retrying wrapper around the VoteHub REST API."""

    base_url: str = BASE_URL
    timeout: float = 60.0
    max_retries: int = 4
    backoff: float = 2.0
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    # -- low level ---------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                # 429 and 5xx are transient; retry them. 4xx otherwise is not.
                if response.status_code == 429 or response.status_code >= 500:
                    raise VoteHubError(
                        f"{url} returned HTTP {response.status_code}"
                    )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, VoteHubError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                sleep_for = self.backoff ** (attempt - 1)
                log.warning(
                    "VoteHub request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

        raise VoteHubError(f"failed to fetch {url} after {self.max_retries} attempts") from last_error

    # -- endpoints ---------------------------------------------------------

    def poll_types(self) -> list[str]:
        return list(self._get("/poll-types"))

    def subjects(self) -> list[dict[str, Any]]:
        return list(self._get("/subjects"))

    def pollsters(self) -> list[str]:
        return list(self._get("/pollsters"))

    def polls(self, poll_type: str, subject: str | None = None) -> list[dict[str, Any]]:
        """All polls of one type, optionally restricted to a single subject."""
        params: dict[str, Any] = {"poll_type": poll_type}
        if subject is not None:
            params["subject"] = subject
        payload = self._get("/polls", params=params)
        if not isinstance(payload, list):
            raise VoteHubError(f"expected a list of polls, got {type(payload).__name__}")
        return payload

    # -- snapshotting ------------------------------------------------------

    def fetch_and_snapshot(
        self,
        poll_types: Sequence[str],
        run_date: date | None = None,
        raw_dir: Path | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch each poll type and persist a dated, gzipped raw snapshot.

        Returns a mapping of poll type to its raw poll list.
        """
        run_date = run_date or date.today()
        raw_dir = raw_dir or paths.RAW_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)

        fetched: dict[str, list[dict[str, Any]]] = {}
        for poll_type in poll_types:
            log.info("fetching poll_type=%s", poll_type)
            records = self.polls(poll_type)
            fetched[poll_type] = records
            log.info("  %d polls", len(records))

        snapshot_path = raw_dir / f"votehub-{run_date.isoformat()}.json.gz"

        # Merge into any snapshot already written for this date rather than
        # replacing it. A snapshot is keyed by date alone, but the two chambers
        # fetch different poll types -- the Senate wants `us-senator`, the House
        # `us-representative` -- and the daily job runs both. Overwriting meant
        # the House run erased the Senate's polls for that day.
        #
        # The forecasts were unaffected, because each run fetches what it needs
        # before writing. What it broke was reproducibility, which is the entire
        # reason these files exist: re-running against an archived snapshot is
        # supposed to reproduce that day's forecast exactly, and for any day
        # both chambers ran, only the last one could be reproduced. It also made
        # `--offline` fail outright for whichever chamber went first.
        #
        # Freshly fetched types win; anything not fetched this time is kept.
        existing: dict[str, Any] = {}
        if snapshot_path.exists():
            try:
                with gzip.open(snapshot_path, "rt", encoding="utf-8") as handle:
                    existing = json.load(handle).get("poll_types", {})
            except (OSError, json.JSONDecodeError, KeyError):
                log.warning("could not read %s to merge into; replacing it",
                            snapshot_path.name)
                existing = {}

        merged = {**existing, **fetched}
        payload = {
            "fetched_at": run_date.isoformat(),
            "source": self.base_url,
            "attribution": ATTRIBUTION,
            "license": "CC BY 4.0",
            "poll_types": merged,
        }
        with gzip.open(snapshot_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        kept = sorted(set(existing) - set(fetched))
        log.info(
            "wrote raw snapshot %s (%s%s)",
            snapshot_path,
            ", ".join(f"{k}={len(v)}" for k, v in sorted(fetched.items())),
            f"; kept {', '.join(kept)}" if kept else "",
        )

        return fetched


def load_snapshot(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read back a snapshot written by :meth:`VoteHubClient.fetch_and_snapshot`."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["poll_types"]


def latest_snapshot(raw_dir: Path | None = None) -> Path | None:
    """Most recent raw snapshot on disk, if any."""
    raw_dir = raw_dir or paths.RAW_DIR
    if not raw_dir.exists():
        return None
    candidates = sorted(raw_dir.glob("votehub-*.json.gz"))
    return candidates[-1] if candidates else None


def general_election_polls(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop primary-subject polls, keeping general-election matchups."""
    return [r for r in records if not is_primary_subject(str(r.get("subject", "")))]
