"""The map's colour ramp and the words printed on top of it.

Both are presentation, and both can be wrong in ways that mislead: a ramp whose
steps a reader cannot separate, or a label that claims more certainty than the
number beside it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_JS = (ROOT / "site" / "js" / "map.js").read_text(encoding="utf-8")


def test_palette_is_separable_and_colour_blind_safe():
    """Runs the real checker, so its thresholds cannot drift from the CSS."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_palette.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _ratings() -> list[tuple[str, str, float]]:
    block = re.search(r"const RATINGS = \[(.*?)\];", MAP_JS, re.S)
    assert block, "map.js must declare RATINGS"
    found = re.findall(
        r"key:\s*'([^']+)',\s*label:\s*'([^']+)',\s*max:\s*([\d.]+)", block.group(1)
    )
    assert found, "could not parse RATINGS"
    return [(key, label, float(cut)) for key, label, cut in found]


def test_no_bucket_is_called_safe():
    """"Safe" is a promise the widest bucket cannot keep.

    The top bucket runs from 80% to 100%. Calling an 80% favourite safe tells
    the reader the race is settled when the model says the underdog wins one
    time in five. The label describes the size of the lead; the printed
    probability is the actual claim.
    """
    for _, label, _ in _ratings():
        assert "safe" not in label.lower(), (
            f"rating {label!r} promises certainty the bucket does not have"
        )


def test_rating_thresholds_are_ordered_and_cover_the_unit_interval():
    ratings = _ratings()
    cuts = [m for _, _, m in ratings]
    assert cuts == sorted(cuts), f"RATINGS thresholds are out of order: {cuts}"
    assert cuts[-1] > 1.0, "top bucket must catch a probability of exactly 1"


def test_the_scale_is_symmetric_about_a_coin_flip():
    """An even race must land in the middle bucket, and the ramp must be even.

    An asymmetric scale would show a systematic thumb on one party without
    anything in the model putting it there.
    """
    ratings = _ratings()
    assert len(ratings) % 2 == 1, "an even count has no middle bucket"
    cuts = [m for _, _, m in ratings][:-1]
    for low, high in zip(cuts, reversed(cuts), strict=True):
        assert low + high == 1.0, f"threshold {low} has no mirror ({high})"


def test_toss_up_never_relies_on_colour_alone():
    """The middle step sits between the hues, so it also carries a hatch."""
    assert "hatch" in MAP_JS.lower(), "the toss-up hatch pattern is gone"
