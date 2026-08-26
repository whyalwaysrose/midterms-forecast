"""Where does the +12 in unpolled red states actually come from?

Every unpolled Republican-leaning race sits about twelve points to the left of
its fundamentals prior, while every unpolled Democratic-leaning race moves about
six. The national environment is D+6.7 and the logit derivative is near 48 at
both ends, so the environment alone cannot produce a two-to-one gap.

The suspicion is pooling. Several Republican-leaning states are polled and show
large Democratic overperformance, and unpolled states that look politically
similar inherit that movement through the correlation kernel:

    Nebraska      1 poll   -22.0 -> +3.6    +25.6
    Idaho         1 poll   -37.7 -> -14.5   +23.2
    Alaska       12 polls  -17.9 -> +2.8    +20.8
    Mississippi   1 poll   -24.8 -> -4.8    +19.9

Nebraska is the one to worry about. Dan Osborn runs as an independent and is
counted on the Democratic side for control arithmetic, so his personal appeal
enters the model as Democratic strength -- and then propagates to every
politically similar state that has no polls of its own to argue otherwise.

This refits with races dropped and reports what moves. If removing Nebraska
alone shifts unpolled Montana and Oklahoma by several points, the pooling is
carrying one independent's popularity into states he is not running in.

Usage:
    python scripts/ablate_pooling.py
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from midterms import fundamentals as F  # noqa: E402
from midterms.config import load_all  # noqa: E402
from midterms.data import Roster, build_poll_table  # noqa: E402
from midterms.data.votehub import latest_snapshot, load_snapshot  # noqa: E402
from midterms.model.design import build_model_data  # noqa: E402
from midterms.model.hierarchical import build_model, sample  # noqa: E402

#: Races to try removing, and why each is a suspect.
ABLATIONS = {
    "none": (),
    "no Nebraska": ("senate-2026-NE",),
    "no Nebraska/Alaska": ("senate-2026-NE", "senate-2026-AK"),
    "no single-poll red states": (
        "senate-2026-ID", "senate-2026-MS", "senate-2026-KS",
        "senate-2026-AL", "senate-2026-AR", "senate-2026-NE",
    ),
}

#: The unpolled races the pooling is suspected of dragging.
WATCH = ("senate-2026-MT", "senate-2026-OK", "senate-2026-SC",
         "senate-2026-KY", "senate-2026-LA", "senate-2026-CO")


def median_margins(idata, data) -> dict[str, float]:
    theta = idata.posterior["theta"].isel(grid=-1)
    draws = theta.stack(sample=("chain", "draw")).transpose("sample", "race").to_numpy()
    margins = 100.0 * (2.0 * (1 / (1 + np.exp(-draws))) - 1)
    return dict(zip(data.race_ids, np.median(margins, axis=0), strict=True))


def main() -> int:
    races, cfg = load_all()
    roster = Roster.load()
    raw = load_snapshot(latest_snapshot())
    fund = F.compute(races, cfg)
    prior = dict(zip(fund.race_ids, fund.prior_mean * 50, strict=True))

    results: dict[str, dict[str, float]] = {}
    for label, dropped in ABLATIONS.items():
        table = build_poll_table(raw, races, cfg, roster)
        if dropped:
            table.polls[:] = [p for p in table.polls if p.race_id not in dropped]
        data = build_model_data(table, races, cfg, fund)
        idata = sample(build_model(data, cfg), cfg, progressbar=False)
        results[label] = median_margins(idata, data)
        print(f"  fitted: {label}")

    print()
    print("=" * 78)
    print("  Posterior median margin in UNPOLLED races, by what was removed")
    print("=" * 78)
    header = f"  {'race':8s} {'prior':>7}" + "".join(f"{k:>22s}" for k in ABLATIONS)
    print(header)
    for race_id in WATCH:
        line = f"  {race_id[-2:]:8s} {prior[race_id]:+7.1f}"
        for label in ABLATIONS:
            value = results[label][race_id]
            delta = value - results["none"][race_id]
            line += f"{value:+12.1f} ({delta:+5.1f})" if label != "none" else f"{value:+22.1f}"
        print(line)

    print()
    print("  A large movement in a race whose own polls were untouched means the")
    print("  number was being carried by other states rather than by anything")
    print("  known about that race.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
