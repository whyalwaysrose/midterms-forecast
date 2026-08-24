"""How much drift does the model itself believe in between now and election day?

The election-day error scales were fitted from polls 45-120 days out, which
means they contain both the systematic polling miss AND the opinion change
between the poll and the election. The model random-walks to election day, so
it represents that second part already — the two may be double-counting.

Whether that is actually a problem depends on a number nothing has measured
yet: how much drift the fitted random walk itself produces. This measures it, by
comparing the posterior spread of each race today against its spread on election
day. The difference is the model's own view of how much can still change.

Compare that against the empirical drift implied by `midterms calibrate` run at
different horizons, and the double-count is either real or it is not.
"""
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from datetime import date

import numpy as np

from midterms import fundamentals as F
from midterms.calibration import estimate_error_components
from midterms.config import load_all
from midterms.data import Roster, build_poll_table
from midterms.data.votehub import latest_snapshot, load_snapshot
from midterms.model.design import build_model_data
from midterms.model.hierarchical import build_model, sample

races, cfg = load_all()
roster = Roster.load()
table = build_poll_table(load_snapshot(latest_snapshot()), races, cfg, roster)
fund = F.compute(races, cfg)
data = build_model_data(table, races, cfg, fund)
idata = sample(build_model(data, cfg), cfg, progressbar=False)

grid = list(data.grid_dates)
today = date.today()
# Nearest grid point to today, and the final point (election day).
now_idx = min(range(len(grid)), key=lambda i: abs((grid[i] - today).days))
end_idx = len(grid) - 1
horizon_days = (grid[end_idx] - grid[now_idx]).days

theta = idata.posterior["theta"]


def margin_sd(index):
    draws = theta.isel(grid=index).stack(sample=("chain", "draw"))
    draws = draws.transpose("sample", "race").to_numpy()
    margins = 100.0 * (2.0 * (1 / (1 + np.exp(-draws))) - 1)
    return margins.std(axis=0)


sd_now = margin_sd(now_idx)
sd_end = margin_sd(end_idx)
# theta_T = theta_t + future innovations, so the extra variance is the drift.
implied_drift = np.sqrt(np.maximum(sd_end**2 - sd_now**2, 0.0))

counts = table.counts_by_race()
polled = np.array([counts.get(r, 0) > 0 for r in races.race_ids])

print("=" * 74)
print(f"  MODEL-IMPLIED DRIFT over {horizon_days} days "
      f"({grid[now_idx]} to {grid[end_idx]})")
print("=" * 74)
print(f"  posterior SD today       : median {np.median(sd_now):5.2f} pts")
print(f"  posterior SD election day: median {np.median(sd_end):5.2f} pts")
print(f"  => model-implied drift   : median {np.median(implied_drift):5.2f} pts")
print(f"       polled races  (n={polled.sum():2d}): {np.median(implied_drift[polled]):5.2f} pts")
print(f"       unpolled      (n={(~polled).sum():2d}): {np.median(implied_drift[~polled]):5.2f} pts")

near = estimate_error_components(days_window=(0, 14))
far = estimate_error_components(days_window=(45, 120))
empirical_drift = np.sqrt(max(far.total_race_sd**2 - near.total_race_sd**2, 0))

print()
print("  Empirically, from historical polls:")
print(f"    error 0-14 days out   : {near.total_race_sd:5.2f} pts  (mostly systematic)")
print(f"    error 45-120 days out : {far.total_race_sd:5.2f} pts")
print(f"    => drift between them : {empirical_drift:5.2f} pts")
print()
model_drift = float(np.median(implied_drift[polled]))
print(f"  model drift {model_drift:.2f} vs empirical {empirical_drift:.2f} pts")
if model_drift < 0.5 * empirical_drift:
    print("  -> the walk is far too rigid; it is NOT covering the drift, so the")
    print("     wide election-day error is doing that work. Do not shrink it.")
elif model_drift > 1.5 * empirical_drift:
    print("  -> the walk drifts more than history does; shrinking the error scales")
    print("     is warranted and may not be enough on its own.")
else:
    print("  -> the walk covers drift about right, so the election-day error should")
    print("     be the NEAR-election figure. The current scales double-count.")

print()
print("  Implied totals per polled race (posterior + election-day error):")
for label, nat, st in (
    ("current config (45-120d)", cfg.election_day_error.national_sd * 50,
     cfg.election_day_error.state_sd * 50),
    ("near-election (0-14d)", near.national_sd, near.state_sd),
):
    ede = float(np.hypot(nat, st))
    total = float(np.median(np.sqrt(sd_end[polled] ** 2 + ede**2)))
    print(f"    {label:26s} error {ede:5.2f} -> total {total:5.2f} pts")
