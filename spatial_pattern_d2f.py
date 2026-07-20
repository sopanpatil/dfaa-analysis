# spatial_pattern_d2f.py
# ------------------------
# DTF equivalent of spatial_pattern_f2d.py. Same logic (coordinate join,
# GB boundary overlay, Moran's I permutation test) pointed at
# d2f_aggregated_output instead of aggregated_output, plus a --clusters
# mode that maps typology cluster assignment and cross-tabulates cluster
# vs. country, mirroring spatial_pattern_f2d.py's --clusters mode.
#
# The main question this answers: does DTF's delta_gap (shortening,
# opposite direction from FTD) show the SAME Scotland/Pennines geography
# as FTD's delta_gap (widening), or a different spatial pattern entirely?
# The same region showing up in both, with opposite sign, is evidence the
# two directions are physically linked (same catchments, same mechanism,
# opposite-signed response) -- this underlies the Discussion's
# interpretation of catchment storage as a shared mechanism across both
# transition directions (Section 4).
#
# Inputs
# ------
#   d2f_aggregated_output/d2f_catchment_metrics_<rcp>.parquet
#   camels_gb_v2_topographic_attributes.csv (gauge_easting/northing)
#   great_britain_boundary.geojson (optional, for map context + country test)
#   d2f_typology_output/cluster_assignments.parquet (optional, for --clusters)
#
# Outputs (in d2f_spatial_output/)
# ------
#   spatial_<metric>_<rcp>.png
#   spatial_clusters_d2f_<rcp>.png    -- catchments coloured by typology cluster
#   country_by_cluster_d2f_<rcp>.csv  -- country vs cluster cross-tab
#   morans_i_summary_d2f.csv
#
# Usage
# -----
#   python spatial_pattern_d2f.py --rcp rcp85 --metric delta_gap
#   python spatial_pattern_d2f.py --rcp rcp85 --all-metrics
#   python spatial_pattern_d2f.py --rcp rcp85 --clusters

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

D2F_DIR = Path("./d2f_aggregated_output")
ATTRS_DIR = Path(".")
OUTPUT_DIR = Path("./d2f_spatial_output")
BOUNDARY_PATH = Path("./great_britain_boundary.geojson")
TOPO_ATTR_FILE = "camels_gb_v2_topographic_attributes.csv"
D2F_TYPOLOGY_DIR = Path("./d2f_typology_output")

DELTA_METRICS = ["delta_gap", "delta_freq", "delta_recovery_rate", "delta_drought_dur",
                  "delta_flood_peak", "delta_sm_drought_end", "delta_sm_flood_onset"]

K_NEIGHBORS = 8
N_PERMUTATIONS = 999
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

def load_boundary():
    if not HAVE_GEOPANDAS or not BOUNDARY_PATH.exists():
        return None
    gdf = gpd.read_file(BOUNDARY_PATH)
    if gdf.crs is not None and gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf


def draw_boundary(ax, boundary):
    if boundary is None:
        return
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coords() -> pd.DataFrame:
    df = pd.read_csv(ATTRS_DIR / TOPO_ATTR_FILE, index_col="gauge_id")
    cols = [c for c in ["gauge_easting", "gauge_northing"] if c in df.columns]
    if len(cols) < 2:
        raise ValueError(f"gauge_easting/gauge_northing not found in {TOPO_ATTR_FILE}")
    return df[cols]


def load_metrics_with_coords(rcp: str) -> pd.DataFrame:
    path = D2F_DIR / f"d2f_catchment_metrics_{rcp}.parquet"
    metrics = pd.read_parquet(path)
    metrics.index = metrics.index.astype(int)
    metrics.index.name = "gauge_id"

    coords = load_coords()
    merged = metrics.join(coords, how="inner")
    n_dropped = len(metrics) - len(merged)
    if n_dropped:
        print(f"  NOTE: {n_dropped} catchments dropped (missing coordinates)")
    return merged


def load_clusters() -> pd.DataFrame:
    path = D2F_TYPOLOGY_DIR / "cluster_assignments.parquet"
    df = pd.read_parquet(path)
    df.index.name = "gauge_id"
    return df


def assign_countries(df: pd.DataFrame, boundary) -> pd.DataFrame:
    """
    Spatially join each catchment (at gauge_easting/gauge_northing) to the
    country polygon (England/Scotland/Wales) it falls within. Identical
    logic to spatial_pattern_f2d.py's assign_countries.
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
# Moran's I spatial autocorrelation (identical to spatial_pattern_f2d.py)
# ---------------------------------------------------------------------------

def build_knn_weights(easting: np.ndarray, northing: np.ndarray, k: int) -> np.ndarray:
    n = len(easting)
    coords = np.column_stack([easting, northing])
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i, 1:]] = 1.0 / k
    return W


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    n = len(values)
    z = values - values.mean()
    return (n / W.sum()) * (z @ W @ z) / (z ** 2).sum()


def morans_i_permutation_test(values, W, n_perm, rng):
    observed = morans_i(values, W)
    perm_stats = np.array([morans_i(rng.permutation(values), W) for _ in range(n_perm)])
    p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_perm + 1)
    return observed, p_value


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_map(df: pd.DataFrame, metric: str, rcp: str, out_path: Path, boundary=None):
    vals = df[metric].values
    vmax = np.nanpercentile(np.abs(vals), 98)

    fig, ax = plt.subplots(figsize=(7, 9))
    draw_boundary(ax, boundary)
    sc = ax.scatter(
        df["gauge_easting"], df["gauge_northing"],
        c=vals, cmap="PRGn", vmin=-vmax, vmax=vmax,
        s=25, edgecolors="none", zorder=2,
    )
    ax.set_aspect("equal")
    ax.set_title(f"d2f {metric} -- {rcp}")
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
    ax.set_title(f"d2f typology clusters -- {rcp} ({cluster_col})")
    ax.set_xlabel("Easting (BNG)")
    ax.set_ylabel("Northing (BNG)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Spatial pattern exploration for d2f delta metrics.")
    p.add_argument("--rcp", default="rcp85")
    p.add_argument("--metric", default=None)
    p.add_argument("--all-metrics", action="store_true")
    p.add_argument("--all-rcps", action="store_true",
                    help="Run all four RCPs (rcp26, rcp45, rcp60, rcp85) in one invocation.")
    p.add_argument("--clusters", action="store_true", help="Map typology cluster assignment.")
    return p.parse_args()


def run_one_rcp(rcp, metrics_to_run, rng):
    print(f"\n{'#'*70}\n# RCP: {rcp}\n{'#'*70}", flush=True)
    print(f"Loading {rcp} d2f metrics with coordinates...", flush=True)
    df = load_metrics_with_coords(rcp)
    print(f"  {len(df)} catchments with valid coordinates", flush=True)

    boundary = load_boundary()
    if boundary is not None:
        print(f"  Loaded GB boundary from {BOUNDARY_PATH}", flush=True)
    else:
        print(f"  No boundary file found -- maps will show points only", flush=True)

    results = []
    W = None
    for metric in metrics_to_run:
        if metric not in df.columns:
            print(f"  WARNING: {metric} not found, skipping")
            continue
        sub = df[[metric, "gauge_easting", "gauge_northing"]].dropna()

        out = OUTPUT_DIR / f"spatial_{metric}_{rcp}.png"
        plot_metric_map(sub, metric, rcp, out, boundary=boundary)
        print(f"  Written: {out}", flush=True)

        if W is None or W.shape[0] != len(sub):
            W = build_knn_weights(sub["gauge_easting"].values, sub["gauge_northing"].values, K_NEIGHBORS)

        observed, p_value = morans_i_permutation_test(sub[metric].values, W, N_PERMUTATIONS, rng)
        results.append({"rcp": rcp, "metric": metric, "n": len(sub),
                         "morans_i": observed, "p_value": p_value})
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"    Moran's I = {observed:+.3f}  p = {p_value:.4f} {sig}", flush=True)

    return results


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    if args.clusters:
        print(f"Loading {args.rcp} d2f metrics with coordinates...", flush=True)
        df = load_metrics_with_coords(args.rcp)
        print(f"  {len(df)} catchments with valid coordinates", flush=True)

        boundary = load_boundary()
        if boundary is not None:
            print(f"  Loaded GB boundary from {BOUNDARY_PATH}", flush=True)
        else:
            print(f"  No boundary file found at {BOUNDARY_PATH} -- maps will show points only", flush=True)

        clusters = load_clusters()
        merged = df.join(clusters[["cluster_kmeans", "cluster_ward"]], how="inner")
        print(f"  {len(merged)} catchments with cluster assignment", flush=True)
        out = OUTPUT_DIR / f"spatial_clusters_d2f_{args.rcp}.png"
        plot_cluster_map(merged, "cluster_kmeans", args.rcp, out, boundary=boundary)
        print(f"  Written: {out}", flush=True)

        if boundary is not None and HAVE_GEOPANDAS:
            print("\n  Assigning catchments to countries (England/Scotland/Wales)...", flush=True)
            with_country = assign_countries(merged, boundary)
            summary = summarise_country_by_cluster(with_country, "cluster_kmeans")
            print("\n  Country composition by cluster:")
            print(summary.to_string())
            summary_path = OUTPUT_DIR / f"country_by_cluster_d2f_{args.rcp}.csv"
            summary.to_csv(summary_path)
            print(f"\n  Written: {summary_path}", flush=True)
        return

    metrics_to_run = DELTA_METRICS if args.all_metrics else [args.metric or "delta_gap"]
    rcps_to_run = ["rcp26", "rcp45", "rcp60", "rcp85"] if args.all_rcps else [args.rcp]

    all_results = []
    for rcp in rcps_to_run:
        all_results.extend(run_one_rcp(rcp, metrics_to_run, rng))

    if all_results:
        result_df = pd.DataFrame(all_results)
        out_path = OUTPUT_DIR / "morans_i_summary_d2f.csv"
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
