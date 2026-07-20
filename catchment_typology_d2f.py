# catchment_typology_d2f.py
# ---------------------------
# d2f equivalent of catchment_typology.py. Same method, same structure --
# k-means (primary) + Ward hierarchical (cross-check), silhouette-based k
# selection, ARI agreement check, labels assigned by descending delta_gap
# -- applied to d2f_catchment_metrics instead of catchment_metrics.
#
# Clustering variables (RCP8.5 delta metrics, z-score standardised):
#   delta_gap             -- recovery gap change (days). NOTE: for d2f this
#                             is typically NEGATIVE (gap shortening) where
#                             f2d's delta_gap is positive (gap widening) --
#                             relabelling by descending delta_gap therefore
#                             puts the LEAST-negative (closest to zero or
#                             positive) cluster first, mirroring f2d's
#                             "largest slowdown first" convention in sign-
#                             appropriate terms for this direction.
#   delta_sm_drought_end  -- SM at drought end change (mm) -- d2f analogue
#                             of f2d's delta_sm_drought
#   delta_freq            -- transition frequency change (events/year)
#   delta_recovery_rate   -- SM recovery rate change (mm/day) -- d2f-specific,
#                             no f2d analogue, included in place of
#                             delta_whiplash (which is direction-agnostic
#                             and already reported for f2d; not duplicated
#                             here to keep the two typologies' variable
#                             sets comparable in kind, not identical)
#
# Outputs (in d2f_typology_output/)
# ------------------------------
#   cluster_assignments.parquet
#   cluster_profiles.parquet
#   optimal_k.parquet
#
# Usage
# -----
#   python catchment_typology_d2f.py
#   python catchment_typology_d2f.py --n-clusters 3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D2F_DIR    = Path("./d2f_aggregated_output")
ATTRS_DIR  = Path(".")
OUTPUT_DIR = Path("./d2f_typology_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
CLUSTER_RCP = "rcp85"

CLUSTER_VARS = [
    "delta_gap",
    "delta_sm_drought_end",
    "delta_freq",
    "delta_recovery_rate",
]

CHARACTERISE_ATTRS = [
    "baseflow_index", "aridity", "p_mean", "runoff_ratio",
    "dpsbar", "frac_high_perc", "p_seasonality", "slope_fdc",
    "frac_snow",  # snow fraction -- included given its role in transition-gap change (Section 3.4)
]

ATTR_FILES = {
    "hydrologic":   "camels_gb_v2_hydrologic_attributes.csv",
    "climatic":     "camels_gb_v2_climatic_attributes.csv",
    "hydrogeology": "camels_gb_v2_hydrogeology_attributes.csv",
    "topographic":  "camels_gb_v2_topographic_attributes.csv",
}

K_RANGE = range(2, 7)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_metrics(rcp: str) -> pd.DataFrame:
    path = D2F_DIR / f"d2f_catchment_metrics_{rcp}.parquet"
    df = pd.read_parquet(path)
    df.index = df.index.astype(int)
    return df


def load_attributes() -> pd.DataFrame:
    frames = []
    for group, fname in ATTR_FILES.items():
        fpath = ATTRS_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Attribute file not found: {fpath}")
        df = pd.read_csv(fpath, index_col="gauge_id")
        frames.append(df)
    attrs = pd.concat(frames, axis=1)

    keep = CHARACTERISE_ATTRS + [
        "gauge_lat", "gauge_lon", "gauge_easting", "gauge_northing",
        "gauge_name", "area",
    ]
    available = [c for c in keep if c in attrs.columns]
    return attrs[available]


# ---------------------------------------------------------------------------
# Optimal k selection (identical method to f2d)
# ---------------------------------------------------------------------------

def select_optimal_k(X_scaled: np.ndarray, k_range: range) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"    k={k}: inertia={km.inertia_:.1f}  silhouette={sil:.3f}", flush=True)
    return pd.DataFrame(rows)


def kmeans_cluster(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=200, max_iter=1000)
    return km.fit_predict(X_scaled)


def ward_cluster(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    Z = linkage(X_scaled, method="ward")
    return fcluster(Z, t=n_clusters, criterion="maxclust") - 1


def relabel_by_delta_gap(labels: np.ndarray, delta_gap: np.ndarray, n_clusters: int) -> np.ndarray:
    means = [delta_gap[labels == k].mean() for k in range(n_clusters)]
    rank = np.argsort(means)[::-1]
    mapping = {old: new for new, old in enumerate(rank)}
    return np.array([mapping[l] for l in labels])


# ---------------------------------------------------------------------------
# Cluster profiles: median + IQR across all RCPs
# ---------------------------------------------------------------------------

def compute_cluster_profiles(assignments: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    rows = []

    for rcp in RCPS:
        metrics = load_metrics(rcp)
        merged = assignments[["cluster_kmeans"]].join(metrics, how="inner")
        delta_cols = [c for c in metrics.columns if c.startswith("delta_")]

        for k in range(n_clusters):
            sub = merged[merged["cluster_kmeans"] == k]
            for col in delta_cols:
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna()
                rows.append({"cluster": k, "rcp": rcp, "variable": col,
                            "median": float(vals.median()), "q25": float(vals.quantile(0.25)),
                            "q75": float(vals.quantile(0.75)), "n": len(vals)})

    attrs = load_attributes()
    merged_attrs = assignments[["cluster_kmeans"]].join(attrs[CHARACTERISE_ATTRS], how="inner")
    for k in range(n_clusters):
        sub = merged_attrs[merged_attrs["cluster_kmeans"] == k]
        for col in CHARACTERISE_ATTRS:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            rows.append({"cluster": k, "rcp": "attribute", "variable": col,
                        "median": float(vals.median()), "q25": float(vals.quantile(0.25)),
                        "q75": float(vals.quantile(0.75)), "n": len(vals)})

    return pd.DataFrame(rows)


def clustering_agreement(labels_km: np.ndarray, labels_ward: np.ndarray) -> float:
    return adjusted_rand_score(labels_km, labels_ward)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Catchment typology clustering on d2f response metrics.")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help=f"Force number of clusters. Default: auto-select from k={list(K_RANGE)}.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {CLUSTER_RCP} d2f metrics ...", flush=True)
    metrics = load_metrics(CLUSTER_RCP)

    print("Loading CAMELS-GB v2 attributes ...", flush=True)
    attrs = load_attributes()

    common = metrics.index.intersection(attrs.index)
    print(f"Common catchments for clustering: {len(common)}", flush=True)

    X_df = metrics.loc[common, CLUSTER_VARS].copy()
    valid = X_df.dropna()
    n_drop = len(X_df) - len(valid)
    if n_drop > 0:
        print(f"  Dropped {n_drop} catchments with NaN in clustering variables", flush=True)

    gauge_ids = valid.index.values
    X_raw = valid.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"\nClustering on {len(gauge_ids)} catchments, "
          f"{len(CLUSTER_VARS)} variables: {CLUSTER_VARS}", flush=True)
    print(f"Scaling: mean={scaler.mean_.round(3)}, std={scaler.scale_.round(3)}", flush=True)

    print("\nSelecting optimal k ...", flush=True)
    k_results = select_optimal_k(X_scaled, K_RANGE)
    k_results.to_parquet(OUTPUT_DIR / "optimal_k.parquet", index=False)
    print("Written: optimal_k.parquet", flush=True)

    best_k_sil = int(k_results.loc[k_results["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette score: {best_k_sil}", flush=True)

    n_clusters = args.n_clusters if args.n_clusters else best_k_sil
    print(f"Using n_clusters = {n_clusters}", flush=True)

    print(f"\nFitting k-means (k={n_clusters}, n_init=200) ...", flush=True)
    labels_km = kmeans_cluster(X_scaled, n_clusters)
    labels_km = relabel_by_delta_gap(labels_km, X_raw[:, 0], n_clusters)

    for k in range(n_clusters):
        n = (labels_km == k).sum()
        mean_gap = X_raw[labels_km == k, 0].mean()
        print(f"  Cluster {k}: n={n}, mean delta_gap={mean_gap:.2f}", flush=True)

    sil_final = silhouette_score(X_scaled, labels_km)
    print(f"  Final silhouette score: {sil_final:.3f}", flush=True)

    print(f"\nFitting Ward hierarchical (k={n_clusters}) ...", flush=True)
    labels_ward = ward_cluster(X_scaled, n_clusters)
    labels_ward = relabel_by_delta_gap(labels_ward, X_raw[:, 0], n_clusters)

    ari = clustering_agreement(labels_km, labels_ward)
    print(f"  Adjusted Rand Index (k-means vs Ward): {ari:.3f}", flush=True)
    print("  (ARI > 0.7 indicates good agreement between methods)", flush=True)

    assignments = pd.DataFrame({
        "gauge_id": gauge_ids, "cluster_kmeans": labels_km, "cluster_ward": labels_ward,
    }).set_index("gauge_id")

    for i, var in enumerate(CLUSTER_VARS):
        assignments[var] = X_raw[:, i]

    attrs_sub = attrs.loc[
        attrs.index.isin(gauge_ids),
        CHARACTERISE_ATTRS + ["gauge_lat", "gauge_lon", "gauge_easting", "gauge_northing", "gauge_name"]
    ]
    assignments = assignments.join(attrs_sub, how="left")

    out = OUTPUT_DIR / "cluster_assignments.parquet"
    assignments.to_parquet(out)
    print(f"\nWritten: {out.name}  shape={assignments.shape}", flush=True)

    print("\nComputing cluster profiles across all RCPs ...", flush=True)
    profiles = compute_cluster_profiles(assignments, n_clusters)
    out = OUTPUT_DIR / "cluster_profiles.parquet"
    profiles.to_parquet(out, index=False)
    print(f"Written: {out.name}  shape={profiles.shape}", flush=True)

    print("\n--- Cluster profiles (RCP8.5 delta metrics) ---")
    rcp85_profiles = profiles[profiles["rcp"] == "rcp85"]
    for var in CLUSTER_VARS:
        sub = rcp85_profiles[rcp85_profiles["variable"] == var].sort_values("cluster")
        print(f"\n  {var}:")
        for _, row in sub.iterrows():
            print(f"    Cluster {int(row['cluster'])}: median={row['median']:+.3f}  "
                  f"IQR=[{row['q25']:+.3f}, {row['q75']:+.3f}]  n={int(row['n'])}")

    print("\n--- Cluster attribute profiles ---")
    attr_profiles = profiles[profiles["rcp"] == "attribute"]
    for attr in ["baseflow_index", "aridity", "runoff_ratio", "dpsbar", "frac_snow"]:
        sub = attr_profiles[attr_profiles["variable"] == attr].sort_values("cluster")
        print(f"\n  {attr}:")
        for _, row in sub.iterrows():
            print(f"    Cluster {int(row['cluster'])}: median={row['median']:.3f}  "
                  f"IQR=[{row['q25']:.3f}, {row['q75']:.3f}]")

    print(f"\nDone. Optimal k by silhouette: {best_k_sil}  Used: {n_clusters}  ARI: {ari:.3f}")


if __name__ == "__main__":
    main()
