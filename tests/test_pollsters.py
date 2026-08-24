"""One pollster, one identity.

The failure this guards against is quiet: a pollster filing under two spellings
becomes two pollsters, each house effect fitted from half the evidence, and
nothing in the output looks wrong. Nineteen Marist polls were being read as ten
plus nine, and twelve NYT/Siena polls as seven plus five.
"""

from __future__ import annotations

import pytest

from midterms.data import pollsters as P


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("GrayHouse", "Grayhouse"),          # case only
        ("Emerson  College", "Emerson College"),  # doubled whitespace
        ("SSRS.", "SSRS"),                   # trailing punctuation
        ("CNN / SSRS", "CNN/SSRS"),          # spacing around a separator
        ("Léger", "Leger"),                  # accents
    ],
)
def test_typographic_variants_normalise_together(a, b):
    assert P.normalise(a) == P.normalise(b)


def test_different_pollsters_do_not_normalise_together():
    """The automatic merge must not be strong enough to conflate two firms."""
    distinct = ["Emerson College", "Marist College", "SSRS", "YouGov", "Ipsos"]
    keys = [P.normalise(n) for n in distinct]
    assert len(set(keys)) == len(distinct)


def test_aliases_resolve_to_one_name():
    assert P.canonical("Marist University") == "Marist College"
    assert P.canonical("The New York Times/Siena University") == (
        "The New York Times/Siena College"
    )
    assert P.canonical("University of New Hampshire Survey Center") == (
        "University of New Hampshire"
    )


def test_an_unknown_pollster_keeps_its_filed_name():
    """No coercion: a name we have no opinion about passes through."""
    assert P.canonical("Quantus Insights") == "Quantus Insights"
    assert P.canonical("") == "Unknown"


def test_unify_accepts_a_generator():
    """It walks its input twice.

    Passing a generator made the second pass see an exhausted iterator, so the
    mapping came back empty and every name was left alone -- indistinguishable
    from "nothing needed merging". That is how this shipped broken the first
    time, so it is pinned here.
    """
    names = ["GrayHouse", "GrayHouse", "Grayhouse"]
    from_list = P.unify(names)
    from_generator = P.unify(n for n in names)
    assert from_generator == from_list
    assert len(set(from_generator.values())) == 1


def test_unify_keeps_the_majority_spelling():
    mapping = P.unify(["Grayhouse"] * 5 + ["GrayHouse"])
    assert set(mapping.values()) == {"Grayhouse"}


def test_unify_breaks_ties_deterministically():
    """Equal counts must not leave the answer to dict ordering."""
    first = P.unify(["GrayHouse", "Grayhouse"])
    second = P.unify(["Grayhouse", "GrayHouse"])
    assert first == second
    assert set(first.values()) == {"GrayHouse"}  # alphabetical


def test_unify_routes_aliases_and_variants_to_the_same_place():
    mapping = P.unify(
        ["Marist University", "Marist College", "marist college"]
    )
    assert len(set(mapping.values())) == 1


def test_probable_duplicates_flags_what_aliases_have_not_covered():
    found = P.find_probable_duplicates(["Acme Polling", "Acme Research Group"])
    assert found, "two names for one firm should be proposed for review"
    assert any(len(v) > 1 for v in found.values())


def test_probable_duplicates_is_silent_when_everything_is_resolved():
    assert P.find_probable_duplicates(["Emerson College", "YouGov", "Ipsos"]) == {}


def test_the_live_feed_has_no_unresolved_duplicates():
    """A new pollster filing two ways should fail here, not go unnoticed.

    This is the regression guard for the real bug: it reads the committed
    aliases against the pollster names the current snapshot actually contains.
    """
    pytest.importorskip("midterms.data.votehub")
    from midterms.config import load_all
    from midterms.data.polls import build_poll_table
    from midterms.data.roster import Roster
    from midterms.data.votehub import latest_snapshot, load_snapshot

    snapshot = latest_snapshot()
    if snapshot is None:
        pytest.skip("no local poll snapshot; covered in CI by the synthetic cases")

    races, cfg = load_all()
    table = build_poll_table(load_snapshot(snapshot), races, cfg, Roster.load())
    leftover = P.find_probable_duplicates(p.pollster for p in table.polls)
    assert not leftover, (
        f"pollster names look like duplicates: {leftover}. If they are the same "
        "firm, add an alias in midterms.data.pollsters.ALIASES."
    )
