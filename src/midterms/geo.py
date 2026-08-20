"""Turn vendored TopoJSON into projected SVG paths for the dashboard map.

Run once at build time (``midterms build-map``); the front end loads only the
resulting flat ``{postal_code: "M…Z"}`` JSON. Doing the projection here rather
than in the browser means the page needs no mapping library, which matters
because the dashboard ships as static files under a strict CSP and must work
with nothing but a stylesheet and one script.

PROJECTION
----------
Albers USA — the composite everyone recognises: a Albers conic equal-area for
the lower 48, plus separately-projected insets for Alaska and Hawaii tucked
under the southwest. It is *equal-area*, so no state's apparent size is
inflated relative to another's, which is the least-bad property available when
the quantity being shown (one Senate seat) has nothing to do with land area.

The parameters below reproduce d3-geo's ``geoAlbersUsa`` at its default scale of
1070 and translate of (480, 250):

    lower 48   parallels 29.5/45.5, rotate 96°, centre (-0.6, 38.7)
    Alaska     parallels 55/65,     rotate 154°, centre (-2, 58.5),  0.35x scale
    Hawaii     parallels 8/18,      rotate 157°, centre (-3, 19.9)

Alaska is drawn at 0.35x, as it is in every published Albers USA map. That is a
deliberate, conventional distortion: at true relative scale Alaska is larger
than the entire lower 48 and would dominate a map on which it holds exactly one
of a hundred seats.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

TOPOJSON_PATH = paths.DATA_DIR / "geo" / "us-states-10m.json"

#: Viewport the paths are generated into.
VIEW_WIDTH = 960
VIEW_HEIGHT = 600

ATTRIBUTION = (
    "State boundaries from us-atlas (ISC), derived from U.S. Census Bureau "
    "cartographic boundary files (public domain)."
)

STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}

#: Territories carry no Senate seat, so they are dropped rather than drawn.
DROPPED = {
    "Puerto Rico", "United States Virgin Islands", "Guam",
    "American Samoa", "Commonwealth of the Northern Mariana Islands",
}


# ---------------------------------------------------------------------------
# Albers conic equal-area
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Albers:
    """One conic-equal-area sub-projection of the composite."""

    parallel_1: float
    parallel_2: float
    rotate_lon: float
    center_lon: float
    center_lat: float
    scale: float
    translate_x: float
    translate_y: float

    def __post_init__(self) -> None:
        p1 = math.radians(self.parallel_1)
        p2 = math.radians(self.parallel_2)
        sin_p1 = math.sin(p1)
        n = (sin_p1 + math.sin(p2)) / 2.0
        # c = cos^2(p1) + 2n sin(p1), written as d3 writes it.
        c = 1.0 + sin_p1 * (2.0 * n - sin_p1)
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_c", c)
        # `center` is given in ROTATED coordinates, exactly as d3 defines it:
        # center=(-0.6, 38.7) with rotate=96 means geographic (-96.6, 38.7),
        # i.e. Kansas. So the centre is fed to _raw already rotated. Applying
        # the rotation to it a second time here shifts the whole map by
        # scale x 1.09 = about 1170px, which is not subtle but is easy to miss
        # because the *shape* still comes out correct.
        cx, cy = self._raw(self.center_lon, self.center_lat)
        object.__setattr__(self, "_cx", cx)
        object.__setattr__(self, "_cy", cy)

    def _raw(self, rotated_lon: float, lat: float) -> tuple[float, float]:
        """Conic equal-area of an ALREADY-ROTATED longitude, in projection units."""
        n: float = self._n            # type: ignore[attr-defined]
        c: float = self._c            # type: ignore[attr-defined]
        lam = math.radians(rotated_lon) * n
        phi = math.radians(lat)
        inner = c - 2.0 * n * math.sin(phi)
        # Guard the pole-side branch where the radicand can go slightly negative.
        rho = math.sqrt(max(inner, 0.0)) / n
        rho0 = math.sqrt(c) / n
        return rho * math.sin(lam), rho0 - rho * math.cos(lam)

    def project(self, lon: float, lat: float) -> tuple[float, float]:
        """Geographic longitude/latitude to viewport pixels."""
        x, y = self._raw(lon + self.rotate_lon, lat)
        # Screen y grows downward, hence the sign flip on the second term.
        return (
            self.translate_x + self.scale * (x - self._cx),   # type: ignore[attr-defined]
            self.translate_y - self.scale * (y - self._cy),   # type: ignore[attr-defined]
        )


def _build_projections(scale: float = 1070.0) -> dict[str, Albers]:
    tx, ty = 480.0, 250.0
    return {
        "lower48": Albers(29.5, 45.5, 96.0, -0.6, 38.7, scale, tx, ty),
        "AK": Albers(
            55.0, 65.0, 154.0, -2.0, 58.5, scale * 0.35,
            tx - 0.307 * scale, ty + 0.201 * scale,
        ),
        "HI": Albers(
            8.0, 18.0, 157.0, -3.0, 19.9, scale,
            tx - 0.205 * scale, ty + 0.212 * scale,
        ),
    }


# ---------------------------------------------------------------------------
# TopoJSON decoding
# ---------------------------------------------------------------------------


def _decode_arcs(topology: dict) -> list[list[tuple[float, float]]]:
    """Undo TopoJSON's delta encoding and quantisation."""
    scale = topology["transform"]["scale"]
    translate = topology["transform"]["translate"]
    decoded: list[list[tuple[float, float]]] = []
    for arc in topology["arcs"]:
        x = y = 0
        points: list[tuple[float, float]] = []
        for dx, dy in arc:
            x += dx
            y += dy
            points.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        decoded.append(points)
    return decoded


def _ring_points(
    ring: Sequence[int], arcs: list[list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """Stitch a ring's arc references into one coordinate list.

    A negative index means "traverse this arc backwards"; TopoJSON encodes that
    as the ones' complement, so arc ``~i`` is arc ``i`` reversed.
    """
    points: list[tuple[float, float]] = []
    for index in ring:
        if index >= 0:
            segment = arcs[index]
        else:
            segment = arcs[~index][::-1]
        # Consecutive arcs share an endpoint; drop the duplicate.
        points.extend(segment[1:] if points else segment)
    return points


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace area in projected units, used only to drop specks."""
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _format_ring(points: Iterable[tuple[float, float]], precision: int) -> str:
    parts = []
    previous: tuple[float, float] | None = None
    for x, y in points:
        rx, ry = round(x, precision), round(y, precision)
        if previous is not None and (rx, ry) == previous:
            continue  # rounding collapsed this point onto the last one
        parts.append(f"{rx:g},{ry:g}")
        previous = (rx, ry)
    if len(parts) < 3:
        return ""
    return "M" + "L".join(parts) + "Z"


# ---------------------------------------------------------------------------


def build_state_paths(
    topojson_path: Path | None = None,
    precision: int = 1,
    min_area: float = 1.5,
) -> dict[str, object]:
    """Project every state and return SVG path data keyed by postal code.

    ``min_area`` drops offshore specks — small islands that survive as
    sub-pixel dots but inflate the payload considerably.
    """
    topojson_path = topojson_path or TOPOJSON_PATH
    topology = json.loads(topojson_path.read_text(encoding="utf-8"))
    arcs = _decode_arcs(topology)
    projections = _build_projections()

    result: dict[str, str] = {}
    centroids: dict[str, list[float]] = {}
    dropped_specks = 0

    for geometry in topology["objects"]["states"]["geometries"]:
        name = geometry["properties"]["name"]
        if name in DROPPED:
            continue
        code = STATE_CODES.get(name)
        if code is None:
            log.warning("unmapped geography %r — skipping", name)
            continue

        projection = projections.get(code, projections["lower48"])

        if geometry["type"] == "Polygon":
            polygons = [geometry["arcs"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["arcs"]
        else:
            log.warning("unexpected geometry type %r for %s", geometry["type"], name)
            continue

        segments: list[str] = []
        for polygon in polygons:
            for ring in polygon:
                lonlat = _ring_points(ring, arcs)
                projected = [projection.project(lon, lat) for lon, lat in lonlat]
                if _polygon_area(projected) < min_area:
                    dropped_specks += 1
                    continue
                path = _format_ring(projected, precision)
                if path:
                    segments.append(path)

        if segments:
            result[code] = "".join(segments)
            centroids[code] = _largest_ring_centroid(segments)

    log.info(
        "projected %d states (%d sub-pixel rings dropped)", len(result), dropped_specks
    )

    # Derive the viewBox from what was actually drawn rather than assuming the
    # nominal 960x600. Alaska's Aleutians run past x=0 (d3 hides them with a
    # clipExtent); fitting the box to the content shows everything, wastes no
    # space, and stays correct if the projection parameters are ever retuned.
    box = _content_bounds(result.values(), margin=8.0)
    return {
        "view_box": box,
        "attribution": ATTRIBUTION,
        "states": result,
        "centroids": centroids,
    }


def _largest_ring_centroid(segments: Sequence[str]) -> list[float]:
    """Centroid of a state's biggest landmass, for label placement.

    The biggest ring rather than all of them: averaging Michigan's two
    peninsulas, or Alaska across the Aleutians, puts the label in open water.
    """
    best_area = -1.0
    best: list[float] = [0.0, 0.0]
    for segment in segments:
        points = [
            (float(a), float(b))
            for a, b in re.findall(r"(-?[\d.]+),(-?[\d.]+)", segment)
        ]
        if len(points) < 3:
            continue
        a2 = cx = cy = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            cross = x1 * y2 - x2 * y1
            a2 += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if abs(a2) < 1e-9:
            continue
        area = abs(a2) / 2.0
        if area > best_area:
            best_area = area
            best = [round(cx / (3.0 * a2), 1), round(cy / (3.0 * a2), 1)]
    return best


def _content_bounds(path_data: Iterable[str], margin: float = 8.0) -> str:
    """A viewBox that tightly encloses every generated path."""
    xs: list[float] = []
    ys: list[float] = []
    for path in path_data:
        for pair in re.findall(r"(-?[\d.]+),(-?[\d.]+)", path):
            xs.append(float(pair[0]))
            ys.append(float(pair[1]))
    if not xs:
        return f"0 0 {VIEW_WIDTH} {VIEW_HEIGHT}"
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    return f"{min_x:g} {min_y:g} {max_x - min_x:g} {max_y - min_y:g}"


def write_state_paths(output: Path | None = None) -> Path:
    """Write ``site/data/us-states.json``."""
    output = output or paths.SITE_DATA_DIR / "us-states.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_state_paths()
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info("wrote %s (%.0f KB)", output, output.stat().st_size / 1024)
    return output
