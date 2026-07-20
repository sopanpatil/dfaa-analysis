# spatial_pattern_f2d.py
# ------------------------
# Spatial analysis of FTD delta metrics across Great Britain (Methods 2.4),
# backing the Moran's I results in Section 3.3 / Figure 3 and the cluster
# maps in Figure 2.
#
# Checks two things:
#   1. Visual: scatter map of catchments at their BNG easting/northing,
#      coloured by a chosen delta metric or by typology cluster assignment.
#   2. Statistical: Moran's I spatial autocorrelation (k-nearest-neighbour
#      weights, permutation-based p-value) for each delta metric, to check
#      whether any apparent spatial clustering is stronger than random
#      chance -- rather than relying on eyeballing a coloured map alone.
#
# Inputs
# ------
#   aggregated_output/catchment_metrics_<rcp>.parquet   (delta metrics)
#   camels_gb_v2_topographic_attributes.csv              (gauge_easting/northing)
#   typology_output/cluster_assignments.parquet          (optional, for cluster overlay)
#
# Outputs (in spatial_output/)
# ------
#   spatial_<metric>_<rcp>.png    -- coloured scatter map
#   spatial_clusters_<rcp>.png    -- catchments coloured by typology cluster
#   morans_i_summary.csv          -- Moran's I and permutation p-value per metric/RCP
#
# Usage
# -----
#   python spatial_pattern_f2d.py --rcp rcp85 --metric delta_gap
#   python spatial_pattern_f2d.py --rcp rcp85 --clusters
#   python spatial_pattern_f2d.py --all-metrics --rcp rcp85   # full sweep + Moran's I table

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

try:
    import geopandas as gpd
    HAVE_GEOPANDAS = True
except ImportError:
    HAVE_GEOPANDAS = False

AGGREGATED_DIR = Path("./aggregated_output")
ATTRS_DIR = Path(".")
TYPOLOGY_DIR = Path("./typology_output")
OUTPUT_DIR = Path("./spatial_output")

# Optional GB boundary for map context, e.g. from ONS Open Geography Portal
# "Countries (December 2022) Boundaries GB BUC" -- already in EPSG:27700 (BNG),
# so no reprojection needed. GB-specific (not UK), so Northern Ireland is
# already excluded, matching prior figure convention. Set to None to skip.
BOUNDARY_PATH = Path("./great_britain_boundary.geojson")

TOPO_ATTR_FILE = "camels_gb_v2_topographic_attributes.csv"

DELTA_METRICS = ["delta_gap", "delta_freq", "delta_smdr", "delta_sm_drought",
                  "delta_whiplash", "delta_flood_peak", "delta_drought_sev"]

K_NEIGHBORS = 8       # for Moran's I spatial weights
N_PERMUTATIONS = 999  # for Moran's I significance test
RANDOM_SEED = 42


def load_boundary():
    """Load the GB boundary polygon, if available. Returns a GeoDataFrame or None."""
    if not HAVE_GEOPANDAS:
        print("  NOTE: geopandas not installed, skipping boundary overlay")
        return None
    if not BOUNDARY_PATH.exists():
        return None
    gdf = gpd.read_file(BOUNDARY_PATH)
    # Reproject to BNG only if it isn't already -- ONS GB layers ship in
    # EPSG:27700 already, so this is normally a no-op.
    if gdf.crs is not None and gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf


def draw_boundary(ax, boundary):
    if boundary is None:
        return
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=0)


def assign_countries(df: pd.DataFrame, boundary) -> pd.DataFrame:
    """
    Spatially join each catchment (at gauge_easting/gauge_northing) to the
    country polygon (England/Scotland/Wales) it falls within. Requires the
    boundary GeoDataFrame to have a name column (ONS files use CTRY22NM).
    """
    if boundary is None:
        raise ValueError("No boundary loaded -- cannot assign countries.")

    name_col = next((c for c in ["CTRY22NM", "CTRY21NM", "name", "NAME"] if c in boundary.columns), None)
    if name_col is None:
        raise ValueError(f"No recognised country-name column in boundary file. Columns: {list(boundary.columns)}")

    points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["gauge_easting"], df["gauge_northing"]), crs=boundary.crs
    )
    joined = gpd.sjoin(points, boundary[[name_col, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={name_col: "country"}).drop(columns=["index_right"], errors="ignore")

    n_unmatched = joined["country"].isna().sum()
    if n_unmatched:
        print(f"  NOTE: {n_unmatched} catchments did not fall within any country polygon "
              f"(likely coastal/estuarine gauge points just outside the coastline boundary)")
    return joined


def summarise_country_by_cluster(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Cross-tab of country vs cluster, counts and row-normalised percentages."""
    counts = pd.crosstab(df["country"], df[cluster_col])
    pct = counts.div(counts.sum(axis=0), axis=1) * 100
    pct.columns = [f"cluster_{c}_pct" for c in pct.columns]
    return pd.concat([counts.add_prefix("cluster_").add_suffix("_n"), pct], axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coords() -> pd.DataFrame:
    """gauge_id (int) -> gauge_easting, gauge_northing, from CAMELS-GB topographic attrs."""
    path = ATTRS_DIR / TOPO_ATTR_FILE
    df = pd.read_csv(path, index_col="gauge_id")
    cols = [c for c in ["gauge_easting", "gauge_northing"] if c in df.columns]
    if len(cols) < 2:
        raise ValueError(
            f"gauge_easting/gauge_northing not found in {TOPO_ATTR_FILE}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[cols]


def load_metrics_with_coords(rcp: str) -> pd.DataFrame:
    metrics_path = AGGREGATED_DIR / f"catchment_metrics_{rcp}.parquet"
    metrics = pd.read_parquet(metrics_path)
    metrics.index = metrics.index.astype(int)
    metrics.index.name = "gauge_id"

    coords = load_coords()

    merged = metrics.join(coords, how="inner")
    n_dropped = len(metrics) - len(merged)
    if n_dropped:
        print(f"  NOTE: {n_dropped} catchments dropped (missing coordinates)")
    return merged


def load_clusters() -> pd.DataFrame:
    path = TYPOLOGY_DIR / "cluster_assignments.parquet"
    df = pd.read_parquet(path)
    df.index.name = "gauge_id"
    return df


# ---------------------------------------------------------------------------
# Moran's I spatial autocorrelation
# ---------------------------------------------------------------------------

def build_knn_weights(easting: np.ndarray, northing: np.ndarray, k: int) -> np.ndarray:
    """Row-standardised k-nearest-neighbour spatial weight matrix."""
    n = len(easting)
    coords = np.column_stack([easting, northing])
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)  # +1: includes self
    _, idx = nbrs.kneighbors(coords)

    W = np.zeros((n, n))
    for i in range(n):
        neighbors = idx[i, 1:]  # drop self (first column)
        W[i, neighbors] = 1.0 / k  # row-standardised
    return W


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    n = len(values)
    z = values - values.mean()
    numerator = z @ W @ z
    denominator = (z ** 2).sum()
    S0 = W.sum()
    return (n / S0) * (numerator / denominator)


def morans_i_permutation_test(values: np.ndarray, W: np.ndarray, n_perm: int, rng: np.random.Generator):
    observed = morans_i(values, W)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(values)
        perm_stats[i] = morans_i(shuffled, W)
    # Two-sided permutation p-value
    p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)
    return observed, p_value, perm_stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_map(df: pd.DataFrame, metric: str, rcp: str, out_path: Path, boundary=None):
    vals = df[metric].values
    vmax = np.nanpercentile(np.abs(vals), 98)  # robust symmetric colour range

    fig, ax = plt.subplots(figsize=(7, 9))
    draw_boundary(ax, boundary)
    sc = ax.scatter(
        df["gauge_easting"], df["gauge_northing"],
        c=vals, cmap="PRGn", vmin=-vmax, vmax=vmax,
        s=25, edgecolors="none", zorder=2,
    )
    ax.set_aspect("equal")
    ax.set_title(f"{metric} -- {rcp}")
    ax.set_xlabel("Easting (BNG)")
    ax.set_ylabel("Northing (BNG)")
    fig.colorbar(sc, ax=ax, label=metric, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cluster_map(df: pd.DataFrame, cluster_col: str, rcp: str, out_path: Path, boundary=None):
    fig, ax = plt.subplots(figsize=(7, 9))
    draw_boundary(ax, boundary)
    clusters = sorted(df[cluster_col].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
    for k, color in zip(clusters, colors):
        sub = df[df[cluster_col] == k]
        ax.scatter(sub["gauge_easting"], sub["gauge_northing"],
                    c=[color], s=25, edgecolors="none", label=f"Cluster {k} (n={len(sub)})", zorder=2)
    ax.set_aspect("equal")
    ax.set_title(f"Typology clusters -- {rcp} ({cluster_col})")
    ax.set_xlabel("Easting (BNG)")
    ax.set_ylabel("Northing (BNG)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Spatial pattern exploration for f2d delta metrics.")
    p.add_argument("--rcp", default="rcp85")
    p.add_argument("--metric", default=None, help="Single delta metric to map.")
    p.add_argument("--all-metrics", action="store_true", help="Map + test all delta metrics.")
    p.add_argument("--clusters", action="store_true", help="Map typology cluster assignment.")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Loading {args.rcp} metrics with coordinates...", flush=True)
    df = load_metrics_with_coords(args.rcp)
    print(f"  {len(df)} catchments with valid coordinates", flush=True)

    boundary = load_boundary()
    if boundary is not None:
        print(f"  Loaded GB boundary from {BOUNDARY_PATH}", flush=True)
    else:
        print(f"  No boundary file found at {BOUNDARY_PATH} -- maps will show points only", flush=True)

    if args.clusters:
        clusters = load_clusters()
        merged = df.join(clusters[["cluster_kmeans", "cluster_ward"]], how="inner")
        print(f"  {len(merged)} catchments with cluster assignment", flush=True)
        out = OUTPUT_DIR / f"spatial_clusters_{args.rcp}.png"
        plot_cluster_map(merged, "cluster_kmeans", args.rcp, out, boundary=boundary)
        print(f"  Written: {out}", flush=True)

        if boundary is not None and HAVE_GEOPANDAS:
            print("\n  Assigning catchments to countries (England/Scotland/Wales)...", flush=True)
            with_country = assign_countries(merged, boundary)
            summary = summarise_country_by_cluster(with_country, "cluster_kmeans")
            print("\n  Country composition by cluster:")
            print(summary.to_string())
            summary_path = OUTPUT_DIR / f"country_by_cluster_{args.rcp}.csv"
            summary.to_csv(summary_path)
            print(f"\n  Written: {summary_path}", flush=True)
        return

    metrics_to_run = DELTA_METRICS if args.all_metrics else [args.metric or "delta_gap"]

    results = []
    W = None
    for metric in metrics_to_run:
        if metric not in df.columns:
            print(f"  WARNING: {metric} not found, skipping")
            continue
        sub = df[[metric, "gauge_easting", "gauge_northing"]].dropna()

        out = OUTPUT_DIR / f"spatial_{metric}_{args.rcp}.png"
        plot_metric_map(sub, metric, args.rcp, out, boundary=boundary)
        print(f"  Written: {out}", flush=True)

        if W is None or W.shape[0] != len(sub):
            W = build_knn_weights(sub["gauge_easting"].values, sub["gauge_northing"].values, K_NEIGHBORS)

        observed, p_value, _ = morans_i_permutation_test(sub[metric].values, W, N_PERMUTATIONS, rng)
        results.append({"rcp": args.rcp, "metric": metric, "n": len(sub),
                         "morans_i": observed, "p_value": p_value})
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"    Moran's I = {observed:+.3f}  p = {p_value:.4f} {sig}", flush=True)

    if results:
        result_df = pd.DataFrame(results)
        out_path = OUTPUT_DIR / "morans_i_summary.csv"
        if out_path.exists():
            existing = pd.read_csv(out_path)
            result_df = pd.concat([existing, result_df], ignore_index=True).drop_duplicates(
                subset=["rcp", "metric"], keep="last"
            )
        result_df.to_csv(out_path, index=False)
        print(f"\nWritten: {out_path}")
        print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
