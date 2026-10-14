"""`midterms fetch` has to fetch every poll type the project reads.

It did not. It listed three by hand -- `us-senator`, `generic-ballot`,
`approval` -- and omitted `us-representative`, so the one command whose entire
purpose is to populate the local snapshot was the one command that could not
populate it for the House. It exited 0 and logged the three types it had
fetched, and `run --offline --chamber house` then failed with "no usable race
polls", which reads as a problem with the data rather than a fetch that never
asked for it.

The cost was not the failed run. It was a House coverage analysis carried out
against a seven-week-old snapshot, because `data/raw` is gitignored so CI's
snapshots never reach a developer machine, and the obvious way to refresh one
silently refreshed everything except the chamber in question.

So the list is derived from RACE_POLL_TYPE rather than written out, and these
tests pin both halves: that the derived list is complete, and that the command
actually asks for all of it.
"""

from __future__ import annotations

import argparse

import pytest

from midterms import cli
from midterms.data.polls import ALL_POLL_TYPES, NATIONAL_POLL_TYPES, RACE_POLL_TYPE


def test_every_chamber_s_race_poll_type_is_in_the_fetch_list():
    """The drift guard. A third chamber would be covered automatically."""
    for chamber, poll_type in RACE_POLL_TYPE.items():
        assert poll_type in ALL_POLL_TYPES, f"{chamber} would not be fetched"


def test_the_shared_national_feeds_are_in_the_fetch_list():
    for poll_type in NATIONAL_POLL_TYPES:
        assert poll_type in ALL_POLL_TYPES


def test_the_fetch_list_has_no_duplicates():
    # A duplicate is a wasted request per run, and `fetch_and_snapshot` would
    # merge the second over the first without complaint.
    assert len(set(ALL_POLL_TYPES)) == len(ALL_POLL_TYPES)


class _Client:
    """Stands in for VoteHubClient, recording what was asked for."""

    requested: list[str] | None = None

    def __init__(self, returns=None):
        self._returns = returns

    def fetch_and_snapshot(self, poll_types, *args, **kwargs):
        type(self).requested = list(poll_types)
        if self._returns is not None:
            return self._returns
        return {t: [{"id": f"{t}-1"}] for t in poll_types}


def _run_fetch(monkeypatch: pytest.MonkeyPatch, returns=None) -> int:
    import midterms.data.votehub as votehub

    monkeypatch.setattr(votehub, "VoteHubClient", lambda: _Client(returns))
    _Client.requested = None
    return cli.cmd_fetch(argparse.Namespace())


def test_fetch_asks_the_feed_for_every_poll_type(monkeypatch: pytest.MonkeyPatch):
    """The regression itself, at the boundary that got it wrong.

    Asserting on the request rather than on the constant: the bug was not a
    wrong list, it was a command that did not use the list.
    """
    assert _run_fetch(monkeypatch) == 0
    assert _Client.requested is not None
    assert set(_Client.requested) == set(ALL_POLL_TYPES)
    assert "us-representative" in _Client.requested


def test_fetch_fails_when_a_poll_type_comes_back_empty(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A silent partial fetch is the failure that cost seven weeks.

    Exiting 0 having quietly not fetched a chamber is what made the original
    bug invisible, so an empty poll type is an error and names itself.
    """
    partial = {
        "us-senator": [{"id": "s1"}],
        "us-representative": [],
        "generic-ballot": [{"id": "g1"}],
        "approval": [{"id": "a1"}],
    }
    with caplog.at_level("ERROR"):
        assert _run_fetch(monkeypatch, returns=partial) == 1
    assert "us-representative" in caplog.text


def test_fetch_succeeds_when_every_type_returns_polls(
    monkeypatch: pytest.MonkeyPatch,
):
    full = {t: [{"id": f"{t}-1"}] for t in ALL_POLL_TYPES}
    assert _run_fetch(monkeypatch, returns=full) == 0
