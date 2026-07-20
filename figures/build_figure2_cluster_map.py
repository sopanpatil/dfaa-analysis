import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Typology (WY1982-2010 baseline): optimal k selected independently per
# direction via silhouette score. Both FTD and DTF: k=2.
# ---------------------------------------------------------------------------
catchments = gpd.read_file("./catchment_boundaries/camels_gb_v2_catchment_boundaries.shp")
catchments["ID_STRING"] = catchments["ID_STRING"].astype(str)
catchments = catchments.set_index("ID_STRING")

boundary = gpd.read_file("../great_britain_boundary.geojson")
boundary = boundary.to_crs(epsg=27700)  # match catchments (British National Grid)

ftd = pd.read_parquet("../typology_output/cluster_assignments.parquet")
ftd.index = ftd.index.astype(str)
dtf = pd.read_parquet("../d2f_typology_output/cluster_assignments.parquet")
dtf.index = dtf.index.astype(str)

ftd_gdf = catchments.join(ftd[["cluster_kmeans"]], how="inner")
dtf_gdf = catchments.join(dtf[["cluster_kmeans"]], how="inner")

n_ftd_clusters = ftd_gdf["cluster_kmeans"].nunique()
n_dtf_clusters = dtf_gdf["cluster_kmeans"].nunique()
print(f"FTD catchments plotted: {len(ftd_gdf)} ({n_ftd_clusters} clusters)")
print(f"DTF catchments plotted: {len(dtf_gdf)} ({n_dtf_clusters} clusters)")

# Shared plot extent (from the boundary, identical for both panels) --
# without this, the two panels' slightly different catchment subsets give
# each axes a different data bounding box, which (combined with geopandas'
# aspect='equal' auto-scaling) shifts each axes box to a different height
# and misaligns the (a)/(b) titles anchored to them.
xmin, ymin, xmax, ymax = boundary.total_bounds

# ---------------------------------------------------------------------------
# Colors: both directions use k=2 (minority large-gap-change vs majority
# modest-change), same 2-color scheme.
# ---------------------------------------------------------------------------
FTD_COLORS = ["#d1495b", "#8da0ab"]   # cluster 0 = minority large-gap-change; 1 = majority modest
DTF_COLORS = ["#d1495b", "#8da0ab"]   # same scheme, DTF's own cluster 0/1

fig, axes = plt.subplots(1, 2, figsize=(9, 7.5))

for ax, gdf, title, colors in [
    (axes[0], ftd_gdf, "(a) FTD", FTD_COLORS),
    (axes[1], dtf_gdf, "(b) DTF", DTF_COLORS),
]:
    clusters_sorted = sorted(gdf["cluster_kmeans"].unique())
    cmap = matplotlib.colors.ListedColormap(colors[:len(clusters_sorted)])
    gdf.plot(
        ax=ax,
        column="cluster_kmeans",
        categorical=True,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.1,
        legend=False,
    )
    boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    counts = gdf["cluster_kmeans"].value_counts().sort_index()
    count_str = ", ".join(f"C{c} n={counts[c]}" for c in clusters_sorted)
    ax.set_title(f"{title}: {count_str}", fontsize=9, loc="left")
    ax.set_axis_off()

# Single shared horizontal legend below both panels (labels apply to both
# directions -- cluster 0 is always the minority, large-gap-change group).
from matplotlib.patches import Patch
shared_legend_elements = [
    Patch(facecolor=FTD_COLORS[0], edgecolor="white", label="Cluster 0 (minority, large gap change)"),
    Patch(facecolor=FTD_COLORS[1], edgecolor="white", label="Cluster 1 (majority, modest change)"),
]
fig.legend(handles=shared_legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           ncol=2, frameon=False, fontsize=8.5, columnspacing=1.2, handletextpad=0.5)

fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("./Figure2_cluster_map.png", dpi=300, bbox_inches="tight")
print("Saved Figure2_cluster_map.png (FTD: 2 clusters, DTF: 2 clusters)")
