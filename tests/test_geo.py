"""The Albers USA projection.

A wrong map projection is uniquely hard to catch by eye: the shapes still look
like states, so it reads as "a map" even when it is badly wrong. During
development the centre was double-rotated, which shifted every coordinate by
about 1170px — invisible as a shape error, obvious as a numbers error.

So these tests check the projection numerically. Rank correlations are the tool:
they assert that the projection *preserves relative geography* without hardcoding
a single expected pixel, so they stay valid if the scale or viewport changes.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from midterms import paths
from midterms.geo import (
    STATE_CODES,
    Albers,
    _build_projections,
    _decode_arcs,
    _ring_points,
    build_state_paths,
)

LOWER_48_EXCLUDED = {"AK", "HI", "DC"}


@pytest.fixture(scope="module")
def projected():
    return build_state_paths()


@pytest.fixture(scope="module")
def geographic_centroids():
    """Unprojected centroid of every state, straight from the source data."""
    topology = json.loads(
        (paths.DATA_DIR / "geo" / "us-states-10m.json").read_text(encoding="utf-8")
    )
    arcs = _decode_arcs(topology)
    centroids: dict[str, tuple[float, float]] = {}
    for geometry in topology["objects"]["states"]["geometries"]:
        code = STATE_CODES.get(geometry["properties"]["name"])
        if code is None:
            continue
        polygons = (
            [geometry["arcs"]] if geometry["type"] == "Polygon" else geometry["arcs"]
        )
        points = [
            point
            for polygon in polygons
            for ring in polygon
            for point in _ring_points(ring, arcs)
        ]
        if points:
            centroids[code] = (
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points),
            )
    return centroids


def _path_points(path_data: str) -> list[tuple[float, float]]:
    return [
        (float(a), float(b))
        for a, b in re.findall(r"(-?[\d.]+),(-?[\d.]+)", path_data)
    ]


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


# ---------------------------------------------------------------- structure


def test_every_state_is_projected(projected):
    states = projected["states"]
    expected = set(STATE_CODES.values())
    assert expected - set(states) == set()


def test_territories_are_dropped(projected):
    """They have no Senate seat, so drawing them would be misleading."""
    assert "PR" not in projected["states"]
    assert len(projected["states"]) == 51  # 50 states + DC


def test_every_path_is_well_formed(projected):
    for code, d in projected["states"].items():
        assert d.startswith("M"), code
        assert d.endswith("Z"), code
        assert len(_path_points(d)) >= 3, code
        assert all(np.isfinite(v) for point in _path_points(d) for v in point), code


def test_view_box_encloses_every_path(projected):
    vx, vy, vw, vh = (float(v) for v in projected["view_box"].split())
    for code, d in projected["states"].items():
        for x, y in _path_points(d):
            assert vx <= x <= vx + vw, f"{code} x={x} outside view box"
            assert vy <= y <= vy + vh, f"{code} y={y} outside view box"


def test_centroid_lies_inside_its_state_bounding_box(projected):
    for code, centroid in projected["centroids"].items():
        points = _path_points(projected["states"][code])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        assert min(xs) <= centroid[0] <= max(xs), code
        assert min(ys) <= centroid[1] <= max(ys), code


# --------------------------------------------------------------- geography


def test_projection_preserves_east_west_order(projected, geographic_centroids):
    """Longitude rank must survive projection almost exactly."""
    codes = [c for c in projected["centroids"] if c not in LOWER_48_EXCLUDED]
    lon = [geographic_centroids[c][0] for c in codes]
    x = [projected["centroids"][c][0] for c in codes]
    assert _spearman(lon, x) > 0.99


def test_projection_preserves_north_south_order(projected, geographic_centroids):
    """Latitude rank inverts, because screen y grows downward."""
    codes = [c for c in projected["centroids"] if c not in LOWER_48_EXCLUDED]
    lat = [geographic_centroids[c][1] for c in codes]
    y = [projected["centroids"][c][1] for c in codes]
    assert _spearman(lat, y) < -0.97


def test_projection_is_equal_area(projected):
    """Projected area must track true land area — that is the whole point of
    choosing an equal-area projection for a map where every state is one seat."""
    land_area = {
        "TX": 268596, "CA": 163695, "MT": 147040, "NM": 121590, "AZ": 113990,
        "NV": 110572, "CO": 104094, "OR": 98379, "WY": 97813, "MI": 96714,
        "MN": 86936, "UT": 84897, "ID": 83569, "KS": 82278, "NE": 77348,
        "SD": 77116, "WA": 71298, "ND": 70698, "OK": 69899, "MO": 69707,
        "FL": 65758, "WI": 65496, "GA": 59425, "IL": 57914, "IA": 56273,
        "NY": 54555, "NC": 53819, "AR": 53179, "AL": 52420, "LA": 52378,
        "MS": 48432, "PA": 46054, "OH": 44826, "VA": 42775, "TN": 42144,
        "KY": 40408, "IN": 36420, "ME": 35380, "SC": 32020, "WV": 24230,
        "MD": 12406, "MA": 10554, "VT": 9616, "NH": 9349, "NJ": 8723,
        "CT": 5543, "DE": 2489, "RI": 1545,
    }

    def area(path_data: str) -> float:
        total = 0.0
        for ring in path_data.split("M")[1:]:
            points = [
                tuple(map(float, pair.split(",")))
                for pair in ring.rstrip("Z").split("L")
            ]
            shoelace = 0.0
            for i in range(len(points)):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % len(points)]
                shoelace += x1 * y2 - x2 * y1
            total += abs(shoelace) / 2.0
        return total

    codes = [c for c in land_area if c in projected["states"]]
    projected_area = [area(projected["states"][c]) for c in codes]
    true_area = [land_area[c] for c in codes]

    assert _spearman(true_area, projected_area) > 0.97

    # Pixels per square mile should be near-constant. The residual spread comes
    # from land area excluding inland water while the polygons include it.
    ratio = np.array(projected_area) / np.array(true_area)
    assert ratio.std() / ratio.mean() < 0.20


def test_extremes_land_where_they_should(projected):
    centroids = {
        c: v for c, v in projected["centroids"].items() if c not in LOWER_48_EXCLUDED
    }
    assert min(centroids, key=lambda c: centroids[c][0]) == "CA"
    assert max(centroids, key=lambda c: centroids[c][0]) == "ME"
    assert min(centroids, key=lambda c: centroids[c][1]) == "WA"
    assert max(centroids, key=lambda c: centroids[c][1]) == "FL"


def test_alaska_and_hawaii_are_inset_below_the_southwest(projected):
    """The composite tucks them under the lower 48 rather than at true position."""
    centroids = projected["centroids"]
    lower_48_bottom = max(
        centroids[c][1] for c in centroids if c not in LOWER_48_EXCLUDED
    )
    # Both insets sit low on the canvas and to the west.
    for code in ("AK", "HI"):
        assert centroids[code][1] > lower_48_bottom * 0.6, code
        assert centroids[code][0] < centroids["TX"][0], code


# ------------------------------------------------------- projection algebra


def test_centre_is_interpreted_in_rotated_coordinates():
    """Regression: the centre was rotated twice, offsetting the map by ~1170px.

    d3 defines `center` in the already-rotated frame, so centre (-0.6, 38.7)
    with rotate 96 means geographic (-96.6, 38.7) — Kansas. Projecting that
    geographic point must therefore land on the projection's translate origin.
    """
    albers = _build_projections()["lower48"]
    x, y = albers.project(-96.6, 38.7)
    assert x == pytest.approx(480.0, abs=1.0)
    assert y == pytest.approx(250.0, abs=1.0)


def test_projection_is_continuous():
    """Nearby points must stay nearby — catches a sign or branch error."""
    albers = _build_projections()["lower48"]
    a = np.array(albers.project(-96.0, 39.0))
    b = np.array(albers.project(-95.9, 39.0))
    assert 0 < np.linalg.norm(a - b) < 20


def test_projection_handles_the_pole_without_blowing_up():
    """The radicand can go negative near the far pole; it must be clamped."""
    albers = Albers(29.5, 45.5, 96.0, -0.6, 38.7, 1070.0, 480.0, 250.0)
    x, y = albers.project(-96.0, 89.9)
    assert np.isfinite(x) and np.isfinite(y)
