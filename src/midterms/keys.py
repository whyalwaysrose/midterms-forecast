"""API keys for the optional data sources.

Two sources need a key, and neither is needed for the forecast to run:

``CENSUS_API_KEY``
    Demographic covariates for the election-day correlation kernel. The kernel
    currently uses political covariates only, and `scripts/fit_correlation.py`
    confirms those genuinely track which states miss together -- so this is an
    improvement, not a repair.

``FEC_API_KEY``
    Period-level campaign finance, which is what a fundraising covariate would
    need to be tested without lookahead. See `scripts/fit_fundraising.py`: the
    bulk files are end-of-cycle, so fitting on them uses money raised after the
    date a live forecast would have been made.

Keys are read from the environment, falling back to a ``.env`` file in the repo
root, which is gitignored. They are never logged, never written to a payload,
and never committed -- the loader here is the only place they are read.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Names of the keys this project knows about, and where to get each.
SIGNUP_URLS = {
    "CENSUS_API_KEY": "https://api.census.gov/data/key_signup.html",
    "FEC_API_KEY": "https://api.data.gov/signup/",
}


def _dotenv_path() -> Path:
    from .paths import REPO_ROOT

    return REPO_ROOT / ".env"


def load_dotenv() -> dict[str, str]:
    """Parse the repo's ``.env``, if it exists.

    Deliberately tiny: ``KEY=value`` per line, ``#`` comments, optional quotes.
    A whole dependency for four lines of parsing is not worth the supply chain.
    """
    path = _dotenv_path()
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values


def get(name: str) -> str | None:
    """One key, from the environment or ``.env``. None if unset.

    The environment wins, so CI can inject a secret without a file on disk.
    """
    value = os.environ.get(name) or load_dotenv().get(name)
    return value.strip() or None if value else None


def require(name: str) -> str:
    """One key, or an error that says exactly how to get it."""
    value = get(name)
    if value:
        return value
    url = SIGNUP_URLS.get(name, "")
    raise MissingKey(
        f"{name} is not set. Request one at {url}, then either export it or add "
        f"a line to .env in the repo root:\n\n    {name}=your-key-here\n\n"
        f"Run `midterms check-keys` to confirm it works."
    )


class MissingKey(RuntimeError):
    """Raised when an optional source is used without its key configured."""


@dataclass(frozen=True)
class KeyStatus:
    """Whether one key is present and whether it actually works."""

    name: str
    present: bool
    working: bool | None  # None when not checked, because absent
    detail: str


def _redact(value: str) -> str:
    """Enough to recognise a key, never enough to use one."""
    return f"{value[:4]}...{value[-2:]}" if len(value) > 8 else "set"


def check_census(key: str) -> tuple[bool, str]:
    """Does this Census key return data?"""
    import urllib.error
    import urllib.request

    url = (
        "https://api.census.gov/data/2023/acs/acs5/profile"
        f"?get=NAME,DP02_0068PE&for=state:06&key={key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            body = response.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except OSError as exc:
        return False, f"could not reach the Census API ({exc})"
    if body.lstrip().startswith("["):
        return True, "returned data for California"
    return False, "responded, but not with data - is the key activated?"


def check_fec(key: str) -> tuple[bool, str]:
    """Does this FEC key return data, and what is the rate limit?"""
    import json
    import urllib.error
    import urllib.request

    url = (
        "https://api.open.fec.gov/v1/candidates/totals/"
        f"?api_key={key}&election_year=2026&office=S&per_page=1"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            remaining = response.headers.get("X-RateLimit-Limit")
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except OSError as exc:
        return False, f"could not reach the FEC API ({exc})"
    count = payload.get("pagination", {}).get("count")
    limit = f", {remaining}/hour" if remaining else ""
    return True, f"{count} Senate candidates for 2026{limit}"


CHECKS = {"CENSUS_API_KEY": check_census, "FEC_API_KEY": check_fec}


def status(check_network: bool = True) -> list[KeyStatus]:
    """Presence, and optionally liveness, of every key this project uses."""
    out: list[KeyStatus] = []
    for name in SIGNUP_URLS:
        value = get(name)
        if not value:
            out.append(KeyStatus(name, False, None, f"not set - {SIGNUP_URLS[name]}"))
            continue
        if not check_network:
            out.append(KeyStatus(name, True, None, f"set ({_redact(value)})"))
            continue
        works, detail = CHECKS[name](value)
        out.append(KeyStatus(name, True, works, f"{_redact(value)} - {detail}"))
    return out
