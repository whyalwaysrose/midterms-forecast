"""Polling data ingestion.

`votehub` talks to the API, `roster` resolves candidate names to parties, and
`polls` turns the two into the tidy table the model consumes.
"""

from .polls import NormalisedPoll, PollTable, build_poll_table
from .roster import Roster
from .votehub import VoteHubClient

__all__ = [
    "NormalisedPoll",
    "PollTable",
    "Roster",
    "VoteHubClient",
    "build_poll_table",
]
