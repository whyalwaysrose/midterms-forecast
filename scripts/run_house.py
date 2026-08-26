"""First end-to-end House fit, to see whether the model scales to 435 races.

Nothing here is new modelling. `model/` is chamber-agnostic -- it takes whatever
races the config gives it -- so this is a check that the existing machinery
copes with an order of magnitude more of them, and a look at whether the answer
is sane before any of it is wired into the dashboard.

What to watch:

* **Geometry.** 435 races means 435 baselines, 435 elasticities and a 435-wide
  correlated random walk. Divergences or a collapsed ESS here would mean the
  parameterisation that works for 35 does not survive the scale-up.
* **Runtime.** The Senate fit takes about a minute. If the House takes an hour
  the daily job needs rethinking, not just extending.
* **The answer.** 91% of districts have no polls at all, so the chamber number
  is almost entirely fundamentals plus the national environment. It should land
  near what the seat-by-seat lean implies, and if it does not, something is
  wrong that a prettier front end would only hide.

Usage:
    python scripts/run_house.py
"""

from __future__ import annotations

import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from midterms import fundamentals as F  # noqa: E402
from midterms import paths  # noqa: E402
from midterms.config import load_all  # noqa: E402
from midterms.data.polls import RACE_POLL_TYPE, build_poll_table  # noqa: E402
from midterms.data.roster import Roster  # noqa: E402
from midterms.data.votehub import latest_snapshot, load_snapshot  # noqa: E402
from midterms.model.design import build_model_data  # noqa: E402
from midterms.model.hierarchical import build_model, convergence_report, sample  # noqa: E402
from midterms.model.simulate import simulate_chamber  # noqa: E402


def main() -> int:
    races, cfg = load_all(chamber="house")
    roster = Roster.load(paths.CONFIG_DIR / "candidates_house_2026.yaml")
    raw = load_snapshot(latest_snapshot())
    table = build_poll_table(
        raw, races, cfg, roster, race_poll_type=RACE_POLL_TYPE["house"]
    )
    fund = F.compute(races, cfg)
    data = build_model_data(table, races, cfg, fund)

    print(f"{len(races.races)} districts, {len(table.race_polls)} district polls, "
          f"{len(table.national)} generic-ballot polls")
    print(f"design: {len(data.race_ids)} races x {len(data.grid_dates)} grid points, "
          f"{len(data.pollster_names)} house effects")

    started = time.time()
    idata = sample(build_model(data, cfg), cfg, progressbar=False)
    elapsed = time.time() - started
    print(f"\nsampled in {elapsed:.0f}s")
    print("convergence:", convergence_report(idata))

    # Persist before doing anything else with it. A 435-race fit costs a quarter
    # of an hour, and the first version of this script threw away exactly one of
    # those to an argument-order mistake on the line below.
    #
    # Then this line threw away the second one. nutpie records its sampler
    # settings as a nested dict on `sample_stats.attrs`, and netCDF attributes
    # may only be strings, numbers or arrays -- so to_netcdf raises *after* the
    # sampling is done and the object is still only in memory. Flattened to a
    # string rather than dropped, because those settings are the record of how
    # the fit was configured and are worth keeping in the file.
    #
    # Guarded, and the guard is the point: a failure while saving must never be
    # the reason the thing being saved is lost.
    paths.TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = paths.TRACES_DIR / "house_probe.nc"
    try:
        for group in idata.groups():
            attrs = getattr(idata, group).attrs
            for key, value in list(attrs.items()):
                if not isinstance(value, (str, int, float, list, tuple, bytes)):
                    attrs[key] = repr(value)
        idata.to_netcdf(trace_path)
        print(f"trace -> {trace_path}")
    except Exception as exc:  # noqa: BLE001 - never lose the fit to the backup
        print(f"could not save the trace ({exc}); continuing with it in memory")

    sim = simulate_chamber(idata, races, cfg, fund)
    print()
    print("=" * 70)
    print(f"  P(Democratic House) {sim.dem_control_prob * 100:5.1f}%")
    print(f"  Democratic seats    {sim.median_seats} "
          f"(90% interval {sim.seats_p05}-{sim.seats_p95})")
    print("=" * 70)

    # Sanity: the seat count should sit near what the district leans imply
    # once the national environment is applied, because almost nothing here is
    # driven by district polling.
    theta = idata.posterior["theta"].isel(grid=-1)
    draws = theta.stack(sample=("chain", "draw")).transpose("sample", "race").to_numpy()
    margins = 100.0 * (2.0 * (1 / (1 + np.exp(-draws))) - 1)
    median = np.median(margins, axis=0)
    order = np.argsort(np.abs(median))

    print("\nclosest 12 districts:")
    for i in order[:12]:
        code = data.race_ids[i].replace("house-2026-", "")
        race = next(r for r in races.races if r.id == data.race_ids[i])
        polls = table.counts_by_race().get(data.race_ids[i], 0)
        print(f"   {code:6s} D{median[i]:+6.1f}  held {race.incumbent_party} "
              f"{race.incumbent_status:8s} {polls} polls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
