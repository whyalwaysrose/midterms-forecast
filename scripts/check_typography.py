"""Is the type on a scale, and can every colour on it be read?

Two things this page got wrong, both invisible from the code and both caught
only by measuring the rendered result:

* **31 distinct font sizes.** 13px beside 13.5px beside 12.5px, each choice
  individually reasonable and the set of them arbitrary. A constrained scale is
  not about having fewer sizes; it is that when the only options are the step
  above and the step below, the relationships stay legible instead of drifting
  apart half a pixel at a time.

* **A colour that existed in one theme only.** `--accent` was declared once and
  never redefined for light mode, where it landed at 1.74:1 -- and it carries
  the majority line on the seat histogram. Nothing failed, nothing logged; the
  line was simply not there for half the readers.

This script is what makes both findings checkable rather than claims in a
comment. Add a bare `font-size: 13px` and the first check fails; darken a
surface until a text token stops clearing AA and the second does.

Run: python scripts/check_typography.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "site" / "css" / "style.css"

# WCAG 2.2 AA. 4.5:1 for body text, 3:1 for large text (>=24px, or >=18.66px
# bold). Everything checked here is small text, so 4.5 throughout.
AA_BODY = 4.5

# Colour pairs the page actually puts together, each with the surfaces it is
# painted on. Listing them explicitly rather than testing the cross product
# keeps the failure message about a real combination a reader would meet.
SURFACES = ("bg", "bg-elev", "bg-elev-2")
TEXT_ON_SURFACES = (
    "text",        # body copy
    "text-dim",    # secondary copy: card subtitles, table cells, list items
    "text-faint",  # tertiary: kickers, axis labels, meta lines, footnotes
    "accent",      # countdown, the majority threshold line and its label
    "dem",         # Democratic figures in running text
    "rep",         # Republican figures in running text
    "dem-line",    # links
)
# A filled pill: the party blue as a background with a label on top. The fill is
# correct in both themes and the label is what has to flip, which is why the
# on-fill colour is a token rather than a hard-coded #fff.
ON_FILL = (("on-dem", "dem"),)


# Captures the token name without its leading dashes, so a lookup reads as
# tokens["accent"] rather than tokens["--accent"].
TOKEN_RE = r"--([\w-]+):\s*(#[0-9a-fA-F]{3,6});"


def _linear(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _block(css: str, selector: str) -> str:
    """The declarations of the first rule with this exact selector."""
    start = css.index(selector)
    open_brace = css.index("{", start)
    depth, i = 1, open_brace + 1
    while depth:
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
        i += 1
    return css[open_brace + 1 : i - 1]


def themes(css: str) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Resolved colour tokens per theme.

    Light is declared twice on purpose -- once under prefers-color-scheme and
    once under [data-theme="light"] -- so that an explicit choice beats the OS.
    The explicit block is the one read here, and a separate check asserts the
    two agree.
    """
    dark = dict(re.findall(TOKEN_RE, _block(css, ":root {")))
    light = dict(dark)
    light.update(
        re.findall(
            TOKEN_RE,
            _block(css, ':root[data-theme="light"] {'),
        )
    )
    return {
        "dark": {k: hex_to_rgb(v) for k, v in dark.items()},
        "light": {k: hex_to_rgb(v) for k, v in light.items()},
    }


def check_type_scale(css: str) -> list[str]:
    """Every font-size comes from the scale, or is a clamp between two steps."""
    problems = []
    for raw in re.findall(r"font-size:\s*([^;}]+)", css):
        value = raw.strip()
        if value == "inherit":
            continue
        if re.fullmatch(r"var\(--fs-[\w-]+\)", value):
            continue
        if value.startswith("clamp(") and "px" not in value:
            continue
        problems.append(
            f"font-size: {value} is off the scale -- use a var(--fs-*) step, "
            f"or a clamp() between two of them"
        )
    return problems


def check_contrast(css: str) -> list[str]:
    problems = []
    for theme, tokens in themes(css).items():
        for fg in TEXT_ON_SURFACES:
            for bg in SURFACES:
                if fg not in tokens or bg not in tokens:
                    problems.append(f"[{theme}] --{fg} or --{bg} is not declared")
                    continue
                ratio = contrast(tokens[fg], tokens[bg])
                if ratio < AA_BODY:
                    problems.append(
                        f"[{theme}] --{fg} on --{bg} is {ratio:.2f}:1, "
                        f"under the {AA_BODY} AA needs for body text"
                    )
        for fg, fill in ON_FILL:
            ratio = contrast(tokens[fg], tokens[fill])
            if ratio < AA_BODY:
                problems.append(
                    f"[{theme}] --{fg} on the --{fill} fill is {ratio:.2f}:1, "
                    f"under the {AA_BODY} AA needs for a pill label"
                )
    return problems


def check_light_is_declared_twice(css: str) -> list[str]:
    """The media-query and explicit light blocks must not drift apart.

    They exist separately so an explicit light choice beats a dark OS, but a
    token fixed in one and forgotten in the other is exactly the failure that
    left --accent unreadable, in a subtler form.
    """
    media = dict(
        re.findall(
            TOKEN_RE,
            _block(css, ':root:not([data-theme="dark"]) {'),
        )
    )
    explicit = dict(
        re.findall(
            TOKEN_RE,
            _block(css, ':root[data-theme="light"] {'),
        )
    )
    problems = []
    for token in sorted(set(media) | set(explicit)):
        a, b = media.get(token), explicit.get(token)
        if a is None:
            problems.append(f"--{token} is set for data-theme=light but not for the media query")
        elif b is None:
            problems.append(f"--{token} is set for the media query but not for data-theme=light")
        elif a.lower() != b.lower():
            problems.append(f"--{token} differs between the two light blocks: {a} vs {b}")
    return problems


def main() -> int:
    css = CSS.read_text(encoding="utf-8")
    problems = (
        check_type_scale(css)
        + check_contrast(css)
        + check_light_is_declared_twice(css)
    )
    if problems:
        print(f"Typography check FAILED ({len(problems)} problems):\n")
        for p in problems:
            print(f"  FAIL  {p}")
        return 1

    sizes = {v.strip() for v in re.findall(r"font-size:\s*([^;}]+)", css)}
    print(
        f"Typography check passed: {len(sizes)} distinct font-size declarations, "
        f"all on the scale; every text token clears {AA_BODY}:1 in both themes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
