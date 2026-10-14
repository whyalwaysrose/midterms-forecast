"""The House roster build, which the daily job now runs unattended.

The roster is what turns a poll's bare candidate names into sides of a ballot.
It was generated once by hand when 40 districts were polled, and by October the
feed had 66: the districts added in between had no roster entry, so every one of
their polls was rejected as "no single D-vs-R matchup" and the House forecast ran
on 38 districts while the data for 61 sat in the feed. Nothing failed.

Putting the build in CI fixes that, and introduces a worse failure in its place:
a bad fetch that writes a near-empty roster would drop *every* district poll, on
a step that is deliberately `continue-on-error`. So the two guards below are the
point of this file. Neither is hypothetical -- the FEC answers a wrong bulk path
with an HTML page and a 200, and the 2026 weball file is only ~190 KB, so size
is no help in telling one from the other.
"""

from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_house_roster as B  # noqa: E402,I001


# --- the shrink guard --------------------------------------------------------

ROSTER = """\
races:
  house-2026-AK-01:
    R: [Nick Begich, Nick Begich III]
    other: [Bill Hill]
    # UNCLASSIFIED 'Matt Schultz' - no FEC filing with that surname
  house-2026-VA-05:
    D: [Tom Perriello]
    R: [John McGuire]
"""


def test_existing_resolved_counts_classified_names_only(tmp_path: Path) -> None:
    path = tmp_path / "roster.yaml"
    path.write_text(ROSTER, encoding="utf-8")
    # Five names across the three sides. The UNCLASSIFIED comment is not one of
    # them: it is precisely a name the build declined to classify, and counting
    # it would let the guard be satisfied by failures.
    assert B.existing_resolved(path) == 5


def test_existing_resolved_is_zero_when_there_is_no_roster(tmp_path: Path) -> None:
    # A first run has nothing to compare against and must not be blocked.
    assert B.existing_resolved(tmp_path / "absent.yaml") == 0


def _filing(cand_id: str, name: str, party: str, state: str, district: str) -> str:
    row = [""] * 20
    row[B.CAND_ID] = cand_id
    row[B.CAND_NAME] = name
    row[B.PARTY] = party
    row[B.CAND_STATE] = state
    row[B.CAND_DISTRICT] = district
    return "|".join(row)


FILINGS = [
    _filing("H0VA05001", "PERRIELLO, TOM", "DEM", "VA", "05"),
    _filing("H0VA05002", "MCGUIRE, JOHN", "REP", "VA", "05"),
    _filing("H0UT01001", "MCADAMS, BEN", "DEM", "UT", "01"),
    _filing("H0UT01002", "OWEN, RILEY", "REP", "UT", "01"),
]


def _fec_zip(tmp_path: Path) -> Path:
    path = tmp_path / "weball26.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(B.FEC_MEMBER, "\n".join(FILINGS))
    return path


def test_a_collapsed_poll_feed_does_not_overwrite_a_good_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The failure this guard exists for.

    An empty or truncated VoteHub response resolves almost nothing. Without the
    guard that writes a near-empty roster, and because the CI step is
    continue-on-error, the House would quietly lose every district poll with a
    green tick on the run.
    """
    out = tmp_path / "roster.yaml"
    out.write_text(ROSTER, encoding="utf-8")
    before = out.read_text(encoding="utf-8")

    monkeypatch.setattr(B, "poll_names", lambda: {"VA-05": {"Tom Perriello"}})
    monkeypatch.setattr(sys, "argv", [
        "build_house_roster.py",
        "--fec", str(_fec_zip(tmp_path)),
        "--out", str(out),
    ])

    assert B.main() == 1
    assert out.read_text(encoding="utf-8") == before, "roster was overwritten"
    assert "::error::" in capsys.readouterr().err


def test_allow_shrink_lets_a_genuine_drop_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "roster.yaml"
    out.write_text(ROSTER, encoding="utf-8")

    monkeypatch.setattr(B, "poll_names", lambda: {"VA-05": {"Tom Perriello"}})
    monkeypatch.setattr(sys, "argv", [
        "build_house_roster.py", "--allow-shrink",
        "--fec", str(_fec_zip(tmp_path)),
        "--out", str(out),
    ])

    assert B.main() == 0
    assert "Tom Perriello" in out.read_text(encoding="utf-8")


def test_a_full_feed_writes_every_resolved_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "roster.yaml"
    monkeypatch.setattr(B, "poll_names", lambda: {
        "VA-05": {"Tom Perriello", "John McGuire"},
        "UT-01": {"Ben McAdams", "Riley Owen", "Jesse West"},
    })
    monkeypatch.setattr(sys, "argv", [
        "build_house_roster.py",
        "--fec", str(_fec_zip(tmp_path)),
        "--out", str(out),
    ])

    assert B.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "  house-2026-VA-05:" in text
    assert "D: [Tom Perriello]" in text
    assert "R: [John McGuire]" in text
    # Unmatched names are recorded, not guessed. This is the whole reason a
    # missing roster entry costs one poll rather than corrupting a district.
    assert "UNCLASSIFIED 'Jesse West'" in text
    # And the header states the real count rather than the 40 it was written at.
    assert "2 districts, not 435" in text


# --- the download guard ------------------------------------------------------

class _Response(io.BytesIO):
    """urlopen's context manager, as much of it as fetch_filings uses."""

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    monkeypatch.setattr(
        B.urllib.request, "urlopen", lambda *a, **k: _Response(body)
    )


def _good_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(B.FEC_MEMBER, "\n".join(FILINGS))
    return buffer.getvalue()


def test_an_html_error_page_is_refused_rather_than_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FEC serves HTML with a 200 for a wrong path.

    Cached, it fails much later and much less legibly, inside `load_filings`,
    as "File is not a zip file" -- by which point the run has already decided
    the roster could not be rebuilt for a reason nobody can read.
    """
    cache = tmp_path / "weball26.zip"
    _serve(monkeypatch, b"<!doctype html><title>Federal Election Commission</title>")

    with pytest.raises(RuntimeError, match="did not return the bulk file"):
        B.fetch_filings(cache)

    assert not cache.exists()
    assert not list(tmp_path.glob("*.part")), "left a partial download behind"


def test_a_zip_without_the_expected_member_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("weball24.txt", "last cycle")
    cache = tmp_path / "weball26.zip"
    _serve(monkeypatch, buffer.getvalue())

    with pytest.raises(RuntimeError, match="weball24.txt"):
        B.fetch_filings(cache)
    assert not cache.exists()


def test_a_good_archive_is_cached_and_reused_the_same_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "weball26.zip"
    _serve(monkeypatch, _good_archive())

    assert B.fetch_filings(cache) == cache
    assert zipfile.is_zipfile(cache)

    # Second call must not go to the network at all: a cache written today is
    # today's filings, and the daily job may run the script more than once.
    def _explode(*a: object, **k: object) -> None:
        raise AssertionError("refetched a cache written today")

    monkeypatch.setattr(B.urllib.request, "urlopen", _explode)
    assert B.fetch_filings(cache) == cache


def test_a_stale_cache_is_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Yesterday's filings are the bug this whole change removes."""
    cache = tmp_path / "weball26.zip"
    cache.write_bytes(b"stale")
    yesterday = time.time() - 60 * 60 * 36
    os.utime(cache, (yesterday, yesterday))

    _serve(monkeypatch, _good_archive())

    B.fetch_filings(cache)
    assert zipfile.is_zipfile(cache), "kept the stale cache"
