import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Computed directly from test_snowmelt_hypothesis(_d2f).py,
# analyze_dynamic_snow.py, and test_frac_snow_incremental(_d2f).py outputs
# (WY1982-2010 baseline). Target metric throughout is delta_gap, the
# primary shared metric between FTD and DTF.
# ---------------------------------------------------------------------------

SPATIAL_DIR = Path("../spatial_output")
D2F_SPATIAL_DIR = Path("../d2f_spatial_output")
D2F_REGRESSION_DIR = Path("../d2f_regression_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
rcps = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"]
x = np.arange(len(rcps))

# --- Static frac_snow correlation with delta_gap ---
static_f2d = pd.read_csv(SPATIAL_DIR / "snowmelt_correlation_test.csv")
static_d2f = pd.read_csv(D2F_SPATIAL_DIR / "d2f_snowmelt_correlation.csv")

def static_series(df, rcp_col="rcp"):
    vals, sigs = [], []
    for rcp in RCPS:
        row = df[(df[rcp_col] == rcp) & (df["delta_metric"] == "delta_gap")].iloc[0]
        vals.append(row["rho"])
        sigs.append(bool(row["significant"]))
    return vals, sigs

ftd_static, ftd_static_sig = static_series(static_f2d)
dtf_static, dtf_static_sig = static_series(static_d2f)

# --- Dynamic snow (delta_mean_sp) correlation with delta_gap, both directions ---
dynamic = pd.read_csv(SPATIAL_DIR / "dynamic_snow_correlation.csv")

def dynamic_series(target):
    vals, sigs = [], []
    for rcp in RCPS:
        row = dynamic[(dynamic["target"] == target) & (dynamic["rcp"] == rcp) &
                       (dynamic["snow_metric"] == "delta_mean_sp")].iloc[0]
        vals.append(row["rho"])
        sigs.append(bool(row["significant"]))
    return vals, sigs

ftd_dynamic, ftd_dynamic_sig = dynamic_series("f2d")
dtf_dynamic, dtf_dynamic_sig = dynamic_series("d2f")

# --- Incremental R^2 gain from frac_snow, target = delta_gap ---
gain_f2d = pd.read_csv(SPATIAL_DIR / "frac_snow_nested_f_test.csv")
gain_d2f = pd.read_csv(D2F_REGRESSION_DIR / "frac_snow_nested_f_test_d2f.csv")

def gain_series(df):
    vals, sigs = [], []
    for rcp in RCPS:
        row = df[(df["rcp"] == rcp) & (df["target"] == "delta_gap")].iloc[0]
        vals.append(row["r2_gain"])
        sigs.append(bool(row["significant"]))
    return vals, sigs

ftd_gain, ftd_gain_sig = gain_series(gain_f2d)
dtf_gain, dtf_gain_sig = gain_series(gain_d2f)

FTD_COLOR = "#c1666b"
DTF_COLOR = "#4a7c8c"
DYNAMIC_COLOR = "#8a6bbf"

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

# --- Panel A: correlation strength ---
ax = axes[0]
width = 0.2
for i, (val, sig, color, label, offset) in enumerate([
    (ftd_static, ftd_static_sig, FTD_COLOR, "FTD: static frac_snow", -1.5 * width),
    (np.abs(ftd_dynamic), ftd_dynamic_sig, DYNAMIC_COLOR, "FTD: dynamic $\\Delta$snow", -0.5 * width),
    (dtf_static, dtf_static_sig, DTF_COLOR, "DTF: static frac_snow", 0.5 * width),
    (np.abs(dtf_dynamic), dtf_dynamic_sig, "#2e5560", "DTF: dynamic $\\Delta$snow", 1.5 * width),
]):
    bars = ax.bar(x + offset, val, width=width, color=color, label=label,
                   edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, sig):
        bar.set_alpha(1.0 if s else 0.35)
ax.set_xticks(x)
ax.set_xticklabels(rcps)
ax.set_ylabel(r"Spearman $|\rho|$ with transition gap")
ax.set_title("(a) Snow-gap correlation strength", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
ax.axhline(0, color="grey", linewidth=0.6)

# --- Panel B: incremental R^2 gain ---
ax = axes[1]
for i, (val, sig, color, label, offset) in enumerate([
    (ftd_gain, ftd_gain_sig, FTD_COLOR, "FTD", -width/2),
    (dtf_gain, dtf_gain_sig, DTF_COLOR, "DTF", width/2),
]):
    bars = ax.bar(x + offset, val, width=width, color=color, label=label,
           edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, sig):
        bar.set_alpha(1.0 if s else 0.35)
ax.set_xticks(x)
ax.set_xticklabels(rcps)
ax.set_ylabel(r"Incremental $R^2$ gain from frac_snow")
ax.set_title("(b) Incremental value beyond terrain/precipitation", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

fig.text(0.5, -0.04, "Faded bars indicate non-significant result (p>0.05)", ha="center", fontsize=8, style="italic")
fig.tight_layout()
fig.savefig("./Figure4_snow_relationships.png", dpi=300, bbox_inches="tight")
print("Saved Figure4_snow_relationships.png")
print()
print("FTD static:", [round(v,3) for v in ftd_static], ftd_static_sig)
print("DTF static:", [round(v,3) for v in dtf_static], dtf_static_sig)
print("FTD dynamic:", [round(v,3) for v in ftd_dynamic], ftd_dynamic_sig)
print("DTF dynamic:", [round(v,3) for v in dtf_dynamic], dtf_dynamic_sig)
print("FTD gain:", [round(v,4) for v in ftd_gain], ftd_gain_sig)
print("DTF gain:", [round(v,4) for v in dtf_gain], dtf_gain_sig)
