"""A/B the approval series: does it actually tighten the national environment?

Fits the same model twice on the same polls, with the approval link on and off,
and reports what it bought. If the answer is "nothing measurable", that is a
result worth having rather than a feature worth keeping.
"""
import dataclasses
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np

from midterms import fundamentals as F
from midterms.config import load_all
from midterms.data import Roster, build_poll_table
from midterms.data.votehub import latest_snapshot, load_snapshot
from midterms.model.design import build_model_data
from midterms.model.hierarchical import build_model, convergence_report, sample
from midterms.model.simulate import national_environment_summary, simulate_chamber

races, cfg = load_all()
roster = Roster.load()
raw = load_snapshot(latest_snapshot())
table = build_poll_table(raw, races, cfg, roster)
fund = F.compute(races, cfg)
print(table.summary())

results = {}
for label, enabled in (("with approval", True), ("without approval", False)):
    approval = dataclasses.replace(cfg.national_environment.approval, enabled=enabled)
    national = dataclasses.replace(cfg.national_environment, approval=approval)
    variant = dataclasses.replace(cfg, national_environment=national)

    data = build_model_data(table, races, variant, fund)
    idata = sample(build_model(data, variant), variant, progressbar=False)

    eta = idata.posterior["eta"].isel(grid=-1).to_numpy().ravel()
    margin = 100.0 * (2.0 * (1 / (1 + np.exp(-eta))) - 1)
    sim = simulate_chamber(idata, races, variant, fund)
    diag = convergence_report(idata)

    results[label] = {
        "eta_sd_pts": float(np.std(margin)),
        "eta_width90_pts": float(np.quantile(margin, 0.95) - np.quantile(margin, 0.05)),
        "eta_median": national_environment_summary(idata)["dem_margin_median"],
        "dem_control": sim.dem_control_prob,
        "seat_sd": float(np.std(sim.dem_seats)),
        "max_r_hat": diag["max_r_hat"],
        "min_ess": diag["min_ess_bulk"],
        "divergences": diag["divergences"],
    }
    if enabled and "approval_corr" in idata.posterior:
        rho = idata.posterior["approval_corr"].to_numpy().ravel()
        results[label]["approval_corr"] = (
            f"{np.mean(rho):+.2f} [{np.quantile(rho,0.05):+.2f}, {np.quantile(rho,0.95):+.2f}]"
        )
    print(f"  done: {label}")

print("\n" + "=" * 78)
print("  DOES PRESIDENTIAL APPROVAL BUY ANYTHING?")
print("=" * 78)
def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


keys = ["eta_median", "eta_sd_pts", "eta_width90_pts", "dem_control", "seat_sd",
        "max_r_hat", "min_ess", "divergences", "approval_corr"]
print(f"  {'':20s} {'with approval':>16s} {'without':>16s}")
for k in keys:
    a = results["with approval"].get(k)
    b = results["without approval"].get(k)
    print(f"  {k:20s} {fmt(a):>16s} {fmt(b):>16s}")

a = results["with approval"]["eta_width90_pts"]
b = results["without approval"]["eta_width90_pts"]
print(f"\n  national-environment 90% width: {a:.2f} vs {b:.2f} pts "
      f"({a/b - 1:+.1%} with approval)")
