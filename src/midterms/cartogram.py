"""A district cartogram: 435 equal squares, arranged like the country.

A choropleth cannot show the House. Congressional districts have equal
population by construction, so they differ in area by three orders of magnitude
-- Wyoming's single seat is larger than the twenty that cover New York City.
Shading real district shapes would give almost all the ink to the emptiest
places and hide the seats that decide the chamber.

So every district is one square of the same size, which is the only honest way
to draw a body where every seat counts once. The squares are grouped by state
and the states are placed roughly where they belong, so the map still reads as
America without pretending to be geography.

**Derived, not hand-drawn.** Published tile-grid layouts are hand-tuned and
carry their own licensing questions. This one is computed from the state
centroids already in `site/data/us-states.json`, which come from the Census
cartographic boundaries the state map is built from. That means it needs no new
data and cannot fall out of step with the map beside it.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Side of one district square, and the gap between squares, in view units.
CELL = 13.0
GAP = 1.6

#: Space left between one state's block and the next.
STATE_PADDING = 5.0


@dataclass(frozen=True)
class Tile:
    """One district's square."""

    district: str
    state: str
    x: float
    y: float


def _state_blocks(seats: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Rows and columns for each state's block of districts.

    Roughly square, but wider than tall: the country is wider than it is tall,
    so a wide block packs against its neighbours better than a tall one.
    """
    shape: dict[str, tuple[int, int]] = {}
    for state, n in seats.items():
        cols = max(1, math.ceil(math.sqrt(n * 1.6)))
        rows = math.ceil(n / cols)
        shape[state] = (rows, cols)
    return shape


def build(
    seats: dict[str, int],
    centroids: dict[str, list[float]],
) -> tuple[list[Tile], tuple[float, float, float, float]]:
    """Place every district, and return the tiles plus a bounding box.

    States are laid out from their centroids and then pushed apart until no two
    blocks overlap. Relaxation rather than a fixed grid because the number of
    seats per state varies by a factor of fifty: California needs a block eight
    squares wide and Wyoming needs one square, and no uniform grid holds both
    without either wasting most of the canvas or overlapping.
    """
    shape = _state_blocks(seats)
    step = CELL + GAP

    # Start each state at its true centroid, scaled up so the biggest blocks
    # have room to exist before relaxation begins.
    #
    # This one number trades map size against faithfulness, and it was measured
    # rather than guessed. A larger scale spreads the blocks out, so relaxation
    # has less work to do and the geography survives better -- but the whole
    # picture grows, and since it is rendered to a fixed page width, every
    # district square gets smaller. Measured across the real 435-seat layout,
    # with square size quoted at a 900px render:
    #
    #     scale   viewBox      overlaps   E-W      N-S     square
    #     1.2      947x576        0      99.5%    97.0%    12.4px
    #     1.5     1145x706        0      99.8%    98.5%    10.2px
    #     1.8     1352x837        0     100.0%    99.5%     8.7px   <-- chosen
    #     2.4     1765x1098       0     100.0%    99.7%     6.6px
    #
    # 1.8 is the last value that keeps ordering above the 99% the tests demand,
    # and it draws squares a third larger than 2.4 did. Below it the map starts
    # telling small lies about where states are, which is the one thing this
    # picture must not do; above it the squares get too small to hover.
    scale = 1.8
    pos = {
        state: [centroids[state][0] * scale, centroids[state][1] * scale]
        for state in seats
        if state in centroids
    }
    missing = sorted(set(seats) - set(pos))
    if missing:
        log.warning("cartogram: no centroid for %s; placed at the margin", missing)
        for i, state in enumerate(missing):
            pos[state] = [0.0, i * 60.0]

    def half(state: str) -> tuple[float, float]:
        rows, cols = shape[state]
        return (cols * step + STATE_PADDING) / 2, (rows * step + STATE_PADDING) / 2

    # Push overlapping blocks apart. Converges quickly because the starting
    # positions are already roughly right; the loop is capped so a pathological
    # arrangement cannot hang the build.
    states = sorted(pos, key=lambda s: -seats[s])
    for _ in range(400):
        moved = False
        for i, a in enumerate(states):
            for b in states[i + 1:]:
                ax, ay = pos[a]
                bx, by = pos[b]
                ahw, ahh = half(a)
                bhw, bhh = half(b)
                dx, dy = bx - ax, by - ay
                overlap_x = (ahw + bhw) - abs(dx)
                overlap_y = (ahh + bhh) - abs(dy)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                moved = True
                # Separate along whichever axis needs the smaller shove, so
                # states slide past each other rather than leaping.
                if overlap_x < overlap_y:
                    shift = overlap_x / 2 + 0.5
                    sign = 1.0 if dx >= 0 else -1.0
                    pos[a][0] -= sign * shift
                    pos[b][0] += sign * shift
                else:
                    shift = overlap_y / 2 + 0.5
                    sign = 1.0 if dy >= 0 else -1.0
                    pos[a][1] -= sign * shift
                    pos[b][1] += sign * shift
        if not moved:
            break

    tiles: list[Tile] = []
    for state, n in sorted(seats.items()):
        rows, cols = shape[state]
        cx, cy = pos[state]
        left = cx - (cols * step - GAP) / 2
        top = cy - (rows * step - GAP) / 2
        for index in range(n):
            row, col = divmod(index, cols)
            # Centre the last, possibly short, row under the ones above it.
            in_row = min(cols, n - row * cols)
            offset = (cols - in_row) * step / 2
            tiles.append(Tile(
                district=f"{state}-{index + 1:02d}",
                state=state,
                x=round(left + offset + col * step, 2),
                y=round(top + row * step, 2),
            ))

    xs = [t.x for t in tiles]
    ys = [t.y for t in tiles]
    box = (
        min(xs) - CELL, min(ys) - CELL,
        max(xs) - min(xs) + 3 * CELL, max(ys) - min(ys) + 3 * CELL,
    )
    return tiles, box


def write(out_path: Path, races, site_data: Path) -> Path:
    """Build the cartogram for a race set and write it beside the state map."""
    geo = json.loads((site_data / "us-states.json").read_text(encoding="utf-8"))
    centroids = geo["centroids"]

    seats: dict[str, int] = defaultdict(int)
    for race in races.races:
        seats[race.unit.split("-")[0]] += 1

    tiles, box = build(dict(seats), centroids)
    payload = {
        "view_box": " ".join(f"{v:.1f}" for v in box),
        "cell": CELL,
        "note": (
            "Every district is one square of equal size. Congressional districts "
            "have equal population but wildly unequal area, so a geographic map "
            "would give most of its ink to the emptiest seats."
        ),
        "tiles": [
            {"district": t.district, "state": t.state, "x": t.x, "y": t.y}
            for t in tiles
        ],
    }
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="")
    log.info("cartogram: %d districts -> %s", len(tiles), out_path)
    return out_path
