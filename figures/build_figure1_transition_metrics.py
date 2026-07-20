import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Computed directly from aggregate_ensemble.py / aggregate_ensemble_d2f.py
# outputs (WY1982-2010 baseline). Median and IQR of each catchment's
# delta_gap / delta_smdr (f2d) or delta_gap / delta_recovery_rate (d2f)
# across all valid catchments, per RCP.
#
# Drought severity is deliberately not shown here: it shows significant
# country-level structure (England/Wales vs Scotland, all four RCPs;
# Section 3.3) rather than a uniform national trend, so it is reported in
# Table 2 / Section 3.3 instead of this RCP-trend figure.
# ---------------------------------------------------------------------------

AGGREGATED_DIR = Path("../aggregated_output")
D2F_DIR = Path("../d2f_aggregated_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
rcps = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"]
x = np.arange(len(rcps))


def median_iqr(rcp: str, aggregated_dir: Path, fname_fmt: str, col: str):
    path = aggregated_dir / fname_fmt.format(rcp=rcp)
    df = pd.read_parquet(path)
    vals = df[col].dropna().values
    return np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)


def build_series(aggregated_dir: Path, fname_fmt: str, col: str):
    return [median_iqr(rcp, aggregated_dir, fname_fmt, col) for rcp in RCPS]


ftd_gap = build_series(AGGREGATED_DIR, "catchment_metrics_{rcp}.parquet", "delta_gap")
dtf_gap = build_series(D2F_DIR, "d2f_catchment_metrics_{rcp}.parquet", "delta_gap")
ftd_rate = build_series(AGGREGATED_DIR, "catchment_metrics_{rcp}.parquet", "delta_smdr")
dtf_rate = build_series(D2F_DIR, "d2f_catchment_metrics_{rcp}.parquet", "delta_recovery_rate")


def to_arrays(data):
    med = np.array([d[0] for d in data])
    lo = np.array([d[1] for d in data])
    hi = np.array([d[2] for d in data])
    err = np.vstack([med - lo, hi - med])
    return med, err


FTD_COLOR = "#c1666b"   # muted red -- FTD (drying)
DTF_COLOR = "#4a7c8c"   # muted teal -- DTF (wetting)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))

# --- Panel A: transition gap ---
ax = axes[0]
med, err = to_arrays(ftd_gap)
ax.errorbar(x - 0.08, med, yerr=err, fmt="o", color=FTD_COLOR, capsize=3, label="FTD", markersize=6)
med, err = to_arrays(dtf_gap)
ax.errorbar(x + 0.08, med, yerr=err, fmt="s", color=DTF_COLOR, capsize=3, label="DTF", markersize=6)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(rcps)
ax.set_ylabel(r"$\Delta$ Transition gap (days)")
ax.set_title("(a) Transition gap", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# --- Panel B: depletion/recovery rate ---
ax = axes[1]
med, err = to_arrays(ftd_rate)
ax.errorbar(x - 0.08, med, yerr=err, fmt="o", color=FTD_COLOR, capsize=3, markersize=6)
med, err = to_arrays(dtf_rate)
ax.errorbar(x + 0.08, med, yerr=err, fmt="s", color=DTF_COLOR, capsize=3, markersize=6)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(rcps)
ax.set_ylabel(r"$\Delta$ Depletion / recovery rate (mm/day)")
ax.set_title("(b) Depletion (FTD) / recovery (DTF) rate", fontsize=10, loc="left")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig("./Figure1_transition_metrics.png", dpi=300, bbox_inches="tight")
print("Saved Figure1_transition_metrics.png (2-panel, drought severity removed)")
print()
print("FTD gap (med, IQR-lo, IQR-hi):", [(round(a,2), round(b,2), round(c,2)) for a,b,c in ftd_gap])
print("DTF gap (med, IQR-lo, IQR-hi):", [(round(a,2), round(b,2), round(c,2)) for a,b,c in dtf_gap])
print("FTD rate (med, IQR-lo, IQR-hi):", [(round(a,3), round(b,3), round(c,3)) for a,b,c in ftd_rate])
print("DTF rate (med, IQR-lo, IQR-hi):", [(round(a,3), round(b,3), round(c,3)) for a,b,c in dtf_rate])
