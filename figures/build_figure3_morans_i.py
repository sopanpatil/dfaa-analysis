import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Computed directly from spatial_pattern_f2d.py / spatial_pattern_d2f.py
# morans_i_summary(_d2f).csv outputs (WY1982-2010 baseline). All cells
# significant at p=0.001 (999 permutations) in both directions.
# ---------------------------------------------------------------------------

SPATIAL_DIR = Path("../spatial_output")
D2F_SPATIAL_DIR = Path("../d2f_spatial_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
rcps = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"]

# (display label, column name) in the desired row order
ftd_rows = [
    ("Transition\nfrequency", "delta_freq"),
    ("Flood\npeak", "delta_flood_peak"),
    ("Transition\ngap", "delta_gap"),
    ("Whiplash", "delta_whiplash"),
    ("Drought\nseverity", "delta_drought_sev"),
    ("Depletion\nrate", "delta_smdr"),
    ("SM at\ndrought onset", "delta_sm_drought"),
]

dtf_rows = [
    ("Transition\ngap", "delta_gap"),
    ("Transition\nfrequency", "delta_freq"),
    ("Recovery\nrate", "delta_recovery_rate"),
    ("Drought\nduration", "delta_drought_dur"),
    ("Flood\npeak", "delta_flood_peak"),
    ("SM at\ndrought end", "delta_sm_drought_end"),
    ("SM at\nflood onset", "delta_sm_flood_onset"),
]


def build_matrix(csv_path: Path, rows):
    df = pd.read_csv(csv_path)
    matrix = np.zeros((len(rows), len(RCPS)))
    for i, (label, col) in enumerate(rows):
        for j, rcp in enumerate(RCPS):
            sub = df[(df["rcp"] == rcp) & (df["metric"] == col)]
            if sub.empty:
                raise ValueError(f"No Moran's I value found for {rcp}/{col} in {csv_path}")
            matrix[i, j] = sub["morans_i"].iloc[0]
    return matrix


ftd_matrix = build_matrix(SPATIAL_DIR / "morans_i_summary.csv", ftd_rows)
dtf_matrix = build_matrix(D2F_SPATIAL_DIR / "morans_i_summary_d2f.csv", dtf_rows)
ftd_metrics = [label for label, _ in ftd_rows]
dtf_metrics = [label for label, _ in dtf_rows]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.35})


def plot_panel(ax, matrix, metrics, label):
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(np.arange(len(rcps)))
    ax.set_xticklabels(rcps, fontsize=9)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if matrix[i, j] > 0.35 else "black")
    ax.set_title(label, fontsize=10, loc="left")
    return im


plot_panel(axes[0], ftd_matrix, ftd_metrics, "(a) FTD: Moran's I across RCPs")
im = plot_panel(axes[1], dtf_matrix, dtf_metrics, "(b) DTF: Moran's I across RCPs")

cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Moran's I", pad=0.02)
cbar.ax.tick_params(labelsize=8)

fig.savefig("./Figure3_morans_i.png", dpi=300, bbox_inches="tight")
print("Saved Figure3_morans_i.png (both panels, full 4-RCP heatmaps)")
