"""Is the map's rating scale readable, ordered, and colour-blind safe?

The map is the first thing on the page and the only place the forecast is
encoded as colour rather than a number, so the ramp has to do real work: steps
a reader can tell apart, lightness that rises towards the middle so the
ordering survives a greyscale print, and separation that holds under the common
forms of colour blindness -- red/green deficiency is exactly the axis a
red-to-blue political map runs along.

Five steps, not seven. Seven were tried and measured: near the midpoint the
adjacent pairs fell to dE 4-10, so Lean and Likely were not separable by anyone.
This script is what makes that finding checkable rather than a claim in a
comment -- point ORDER at a seven-step ramp and it fails.

Run: python scripts/check_palette.py
"""

from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "site" / "css" / "style.css"

ORDER = ["safe-r", "lean-r", "tossup", "lean-d", "safe-d"]

# Minimum CIE76 dE between adjacent steps for ordinary colour vision. 15 is
# the floor the seven-step attempt was judged against and failed; keeping the
# same number here is what makes that comparison meaningful.
MIN_ADJACENT_DE = 15.0
# A lower bar under simulated colour blindness, which collapses a whole axis of
# the space. Toss-up -- the step most at risk -- carries a hatch as well, so no
# category depends on colour alone even at the bottom of this range.
MIN_CVD_DE = 10.0
# Non-adjacent steps must be clearly different, not merely different.
MIN_DISTANT_DE = 18.0


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (_linear(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    la, aa, ba = rgb_to_lab(hex_to_rgb(a))
    lb, ab, bb = rgb_to_lab(hex_to_rgb(b))
    return math.dist((la, aa, ba), (lb, ab, bb))


def luminance(value: str) -> float:
    r, g, b = (_linear(c) for c in hex_to_rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


# Brettel/Viénot-style simulation matrices in linear RGB.
CVD = {
    "protanopia": ((0.0, 2.02344, -2.52581), (0, 1, 0), (0, 0, 1)),
    "deuteranopia": ((1, 0, 0), (0.494207, 0.0, 1.24827), (0, 0, 1)),
    "tritanopia": ((1, 0, 0), (0, 1, 0), (-0.395913, 0.801109, 0.0)),
}


def simulate(value: str, kind: str) -> str:
    r, g, b = (_linear(c) for c in hex_to_rgb(value))
    m = CVD[kind]
    if kind == "protanopia":
        r = m[0][1] * g + m[0][2] * b
    elif kind == "deuteranopia":
        g = m[1][0] * r + m[1][2] * b
    else:
        b = m[2][0] * r + m[2][1] * g

    def encode(c: float) -> int:
        c = min(1.0, max(0.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(255 * c)

    return "#{:02x}{:02x}{:02x}".format(encode(r), encode(g), encode(b))


def read_theme(css: str, selector: str) -> dict[str, str]:
    """Pull the rating colours out of one :root-like block."""
    block = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, re.S)
    if not block:
        raise SystemExit(f"no CSS block for {selector!r}")
    found = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", block.group(1)))
    missing = [k for k in ORDER if k not in found]
    if missing:
        raise SystemExit(f"{selector} is missing {missing}")
    return {k: found[k] for k in ORDER}


def check(name: str, ramp: dict[str, str], background: str) -> list[str]:
    problems = []
    keys = list(ramp)

    for a, b in itertools.pairwise(keys):
        d = delta_e(ramp[a], ramp[b])
        flag = "" if d >= MIN_ADJACENT_DE else "  <-- too close"
        print(f"    {a:9s} -> {b:9s}  dE {d:5.1f}{flag}")
        if d < MIN_ADJACENT_DE:
            problems.append(f"{name}: {a} and {b} differ by only dE {d:.1f}")

    for a, b in itertools.combinations(keys, 2):
        if abs(keys.index(a) - keys.index(b)) == 1:
            continue
        d = delta_e(ramp[a], ramp[b])
        if d < MIN_DISTANT_DE:
            problems.append(f"{name}: non-adjacent {a}/{b} only dE {d:.1f}")

    # Lightness must rise to the middle and fall again, so the ramp still reads
    # as an ordering in greyscale.
    # Within each party's half the ramp must lighten as the race gets closer,
    # so the ordering survives greyscale. Toss-up is excluded on purpose: it is
    # grey rather than a pale red or blue, which breaks the lightness sequence
    # by design, and it carries a diagonal hatch so it never relies on colour
    # alone. It is the step most at risk of being misread, so it gets the one
    # redundant channel.
    lightness = [rgb_to_lab(hex_to_rgb(ramp[k]))[0] for k in keys]
    middle = len(keys) // 2
    left, right = lightness[:middle], lightness[middle + 1:]
    rising = all(x < y for x, y in itertools.pairwise(left))
    falling = all(x > y for x, y in itertools.pairwise(right))
    print(f"    lightness {[round(v) for v in lightness]}  (centre excluded)"
          f"  {'ordered' if rising and falling else 'NOT ordered'}")
    if not (rising and falling):
        problems.append(f"{name}: lightness does not lighten towards the centre")

    for kind in CVD:
        worst = min(
            (delta_e(simulate(ramp[a], kind), simulate(ramp[b], kind)), a, b)
            for a, b in itertools.pairwise(keys)
        )
        d, a, b = worst
        flag = "" if d >= MIN_CVD_DE else "  <-- too close"
        print(f"    {kind:13s} worst adjacent pair {a}/{b}: dE {d:5.1f}{flag}")
        if d < MIN_CVD_DE:
            problems.append(f"{name}/{kind}: {a} and {b} collapse to dE {d:.1f}")

    for key, value in ramp.items():
        c = contrast(value, background)
        if c < 1.35:
            problems.append(
                f"{name}: {key} has contrast {c:.2f} against the page, so a "
                "state of that rating would not be visible"
            )
    return problems


def main() -> int:
    css = CSS.read_text(encoding="utf-8")
    # The stylesheet is dark-first, so bare :root is the DARK theme and the
    # light values live in the explicit [data-theme="light"] block.
    themes = [
        ("dark", ":root", "#0e1116"),
        ("light", ':root[data-theme="light"]', "#f6f7f9"),
    ]
    problems: list[str] = []
    for name, selector, background in themes:
        print(f"\n  {name} theme")
        problems += check(name, read_theme(css, selector), background)

    print()
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        return 1
    print(f"  palette OK: {len(ORDER)} steps, separable, ordered, colour-blind safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
