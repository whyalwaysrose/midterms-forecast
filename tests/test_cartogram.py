"""The district cartogram must be readable as a map and honest as a count.

Two failures matter here and neither shows up as an exception.

A square that overlaps another is a district the reader cannot click, and a
district that never got a square is a seat that vanishes from the picture while
still counting toward 218 -- the one thing this drawing could hide.

Everything else is presentation, but the geographic ordering is checked too,
because the layout is *derived* rather than hand-drawn: if the relaxation ever
stops converging, the tiles will still all be there and still not overlap, and
the only symptom will be a map that no longer looks like America.
"""

from __future__ import annotations

import itertools
import json

import pytest

from midterms import cartogram, paths
from midterms.config import load_all


@pytest.fixture(scope="module")
def centroids() -> dict[str, list[float]]:
    geo = json.loads(
        (paths.SITE_DATA_DIR / "us-states.json").read_text(encoding="utf-8")
    )
    return geo["centroids"]


@pytest.fixture(scope="module")
def house_seats() -> dict[str, int]:
    races, _ = load_all(chamber="house")
    seats: dict[str, int] = {}
    for race in races.races:
        state = race.unit.split("-")[0]
        seats[state] = seats.get(state, 0) + 1
    return seats


@pytest.fixture(scope="module")
def built(house_seats, centroids):
    return cartogram.build(house_seats, centroids)


def test_every_district_gets_exactly_one_square(built, house_seats):
    tiles, _ = built
    assert len(tiles) == sum(house_seats.values()) == 435

    codes = [t.district for t in tiles]
    assert len(set(codes)) == len(codes), "a district was placed twice"

    expected = {
        f"{state}-{n:02d}"
        for state, seats in house_seats.items()
        for n in range(1, seats + 1)
    }
    assert set(codes) == expected


def test_no_two_squares_overlap(built):
    """Overlapping squares mean a district the reader cannot hover or click."""
    tiles, _ = built
    cell = cartogram.CELL
    collisions = [
        (a.district, b.district)
        for a, b in itertools.combinations(tiles, 2)
        if abs(a.x - b.x) < cell - 1e-6 and abs(a.y - b.y) < cell - 1e-6
    ]
    assert not collisions, f"{len(collisions)} overlapping pairs, e.g. {collisions[:3]}"


def test_a_state_block_is_contiguous(built):
    """Each state's squares form one block, not a scattering.

    Checked by area: the tiles of a state must fill their own bounding box
    except for the ragged tail of the last row. A state whose squares drifted
    apart would still pass the overlap test while reading as several states.
    """
    tiles, _ = built
    step = cartogram.CELL + cartogram.GAP
    by_state: dict[str, list] = {}
    for tile in tiles:
        by_state.setdefault(tile.state, []).append(tile)

    for state, group in by_state.items():
        width = max(t.x for t in group) - min(t.x for t in group) + step
        height = max(t.y for t in group) - min(t.y for t in group) + step
        cells = round(width / step) * round(height / step)
        # The last row may be short, so the box can hold up to one row of slack.
        slack = round(width / step)
        assert len(group) <= cells <= len(group) + slack, (
            f"{state}'s {len(group)} squares are spread over {cells} cells"
        )


def test_the_layout_still_looks_like_america(built, centroids):
    """States keep their relative positions after the blocks are pushed apart.

    This is the only test that would catch a relaxation that stopped converging:
    the tiles would still all be present and still not overlap, and the map
    would simply stop being recognisable. A strict threshold rather than a loose
    one, because the measured value is 100% east-west and 99.7% north-south --
    anything materially worse is a regression, not noise.
    """
    tiles, _ = built
    middle: dict[str, tuple[float, float]] = {}
    for state in {t.state for t in tiles}:
        group = [t for t in tiles if t.state == state]
        middle[state] = (
            sum(t.x for t in group) / len(group),
            sum(t.y for t in group) / len(group),
        )

    pairs = list(itertools.combinations(sorted(middle), 2))
    for axis, index in (("east-west", 0), ("north-south", 1)):
        kept = sum(
            (middle[a][index] > middle[b][index])
            == (centroids[a][index] > centroids[b][index])
            for a, b in pairs
        )
        assert kept / len(pairs) >= 0.99, (
            f"only {kept}/{len(pairs)} state pairs keep their {axis} order"
        )


def test_view_box_contains_every_square(built):
    tiles, box = built
    x, y, w, h = box
    for tile in tiles:
        assert x <= tile.x and tile.x + cartogram.CELL <= x + w
        assert y <= tile.y and tile.y + cartogram.CELL <= y + h


def test_missing_centroid_does_not_lose_the_district(house_seats, centroids):
    """A state with no centroid is placed at the margin, not dropped.

    The layout is derived from a second file, and the failure mode of a derived
    layout is that the two fall out of step. Losing a state silently would mean
    a chamber that quietly adds up to less than 435.
    """
    trimmed = {k: v for k, v in centroids.items() if k != "OH"}
    tiles, _ = cartogram.build(house_seats, trimmed)
    assert len(tiles) == 435
    assert sum(t.state == "OH" for t in tiles) == house_seats["OH"]
