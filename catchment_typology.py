# catchment_typology.py
# ----------------------
# Cluster catchments by their flood-to-drought transition response profile
# under RCP8.5 (strongest signal) using k-means and hierarchical clustering.
#
# Clustering variables (RCP8.5 delta metrics, z-score standardised):
#   delta_gap        -- transition slowdown (days)
#   delta_sm_drought -- SM storage depletion at drought onset (mm)
#   delta_freq       -- transition frequency change (events/year)
#   delta_whiplash   -- SSI whiplash frequency change (events/30yr)
#
# Outputs (in typology_output/)
# ------------------------------
#   cluster_assignments.parquet
#       gauge_id | cluster_kmeans | cluster_ward | delta_* | attr_* | coords
#
#   cluster_profiles.parquet
#       Cluster-mean and IQR for all delta metrics and key attributes,
#       across all four RCPs (to show behaviour under different emissions).
#
#   optimal_k.parquet
#       Silhouette scores and inertia for k=2..6 (for elbow plot).
#
# Usage
# -----
#   python catchment_typology.py
#   python catchment_typology.py --n-clusters 3   # force k=3
#
# Method note
# -----------
# K-means is the primary method (reported in main paper).
# Ward hierarchical clustering is a cross-check (supplementary).
# Cluster labels are assigned by descending delta_gap so that
# Cluster 1 always has the largest transition slowdown -- this
# makes labels consistent across runs despite k-means randomness.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGGREGATED_DIR = Path("./aggregated_output")
ATTRS_DIR      = Path(".")
OUTPUT_DIR     = Path("./typology_output")

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
CLUSTER_RCP = "rcp85"   # RCP used for clustering (strongest signal)

# Variables used for clustering
CLUSTER_VARS = [
    "delta_gap",
    "delta_sm_drought",
    "delta_freq",
    "delta_whiplash",
]

# Key attributes to characterise clusters post-hoc
CHARACTERISE_ATTRS = [
    "baseflow_index",
    "aridity",
    "p_mean",
    "runoff_ratio",
    "dpsbar",
    "frac_high_perc",
    "p_seasonality",
    "slope_fdc",
    "frac_snow",  # snow fraction -- included given its role in transition-gap change (Section 3.4)
]

# Attribute CSV files
ATTR_FILES = {
    "hydrologic":   "camels_gb_v2_hydrologic_attributes.csv",
    "climatic":     "camels_gb_v2_climatic_attributes.csv",
    "hydrogeology": "camels_gb_v2_hydrogeology_attributes.csv",
    "topographic":  "camels_gb_v2_topographic_attributes.csv",
}

K_RANGE    = range(2, 7)   # test k=2..6
K_DEFAULT  = 3             # hypothesis-driven starting point
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_metrics(rcp: str) -> pd.DataFrame:
    path = AGGREGATED_DIR / f"catchment_metrics_{rcp}.parquet"
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

    # Keep characterisation attributes + coordinates
    keep = CHARACTERISE_ATTRS + [
        "gauge_lat", "gauge_lon",
        "gauge_easting", "gauge_northing",
        "gauge_name", "area",
    ]
    available = [c for c in keep if c in attrs.columns]
    return attrs[available]


# ---------------------------------------------------------------------------
# Optimal k selection
# ---------------------------------------------------------------------------

def select_optimal_k(
    X_scaled: np.ndarray,
    k_range: range,
) -> pd.DataFrame:
    """
    Compute silhouette score and inertia for each k.
    Returns DataFrame for elbow/silhouette plot.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=50)
        labels = km.fit_predict(X_scaled)
        sil    = silhouette_score(X_scaled, labels)
        rows.append({
            "k":         k,
            "inertia":   km.inertia_,
            "silhouette": sil,
        })
        print(f"    k={k}: inertia={km.inertia_:.1f}  silhouette={sil:.3f}",
              flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------

def kmeans_cluster(
    X_scaled: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Fit k-means with many initialisations for stability.
    Returns cluster labels (0-indexed).
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=200,       # many initialisations for stability
        max_iter=1000,
    )
    return km.fit_predict(X_scaled)


# ---------------------------------------------------------------------------
# Ward hierarchical clustering
# ---------------------------------------------------------------------------

def ward_cluster(
    X_scaled: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Ward linkage hierarchical clustering.
    Returns cluster labels (0-indexed).
    """
    Z      = linkage(X_scaled, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels


# ---------------------------------------------------------------------------
# Relabel clusters by descending delta_gap
# (Cluster 0 = largest transition slowdown, for reproducibility)
# ---------------------------------------------------------------------------

def relabel_by_delta_gap(
    labels:     np.ndarray,
    delta_gap:  np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Reassign cluster integer labels so that cluster 0 has the highest
    mean delta_gap, cluster 1 the second highest, etc.
    This makes labels consistent across runs and methods.
    """
    means = [delta_gap[labels == k].mean() for k in range(n_clusters)]
    rank  = np.argsort(means)[::-1]   # descending
    mapping = {old: new for new, old in enumerate(rank)}
    return np.array([mapping[l] for l in labels])


# ---------------------------------------------------------------------------
# Cluster profiles: mean ± IQR across all RCPs
# ---------------------------------------------------------------------------

def compute_cluster_profiles(
    assignments: pd.DataFrame,
    n_clusters:  int,
) -> pd.DataFrame:
    """
    For each cluster, compute median and IQR of:
    - All delta metrics across all four RCPs
    - Key catchment attributes (time-invariant)

    Returns long-format DataFrame:
        cluster | rcp | variable | median | q25 | q75
    """
    rows = []

    # Delta metrics: vary by RCP
    for rcp in RCPS:
        metrics = load_metrics(rcp)
        merged  = assignments[["cluster_kmeans"]].join(
            metrics, how="inner"
        )
        delta_cols = [c for c in metrics.columns if c.startswith("delta_")]

        for k in range(n_clusters):
            sub = merged[merged["cluster_kmeans"] == k]
            for col in delta_cols:
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna()
                rows.append({
                    "cluster":  k,
                    "rcp":      rcp,
                    "variable": col,
                    "median":   float(vals.median()),
                    "q25":      float(vals.quantile(0.25)),
                    "q75":      float(vals.quantile(0.75)),
                    "n":        len(vals),
                })

    # Catchment attributes: time-invariant, use once
    attrs = load_attributes()
    merged_attrs = assignments[["cluster_kmeans"]].join(
        attrs[CHARACTERISE_ATTRS], how="inner"
    )
    for k in range(n_clusters):
        sub = merged_attrs[merged_attrs["cluster_kmeans"] == k]
        for col in CHARACTERISE_ATTRS:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            rows.append({
                "cluster":  k,
                "rcp":      "attribute",
                "variable": col,
                "median":   float(vals.median()),
                "q25":      float(vals.quantile(0.25)),
                "q75":      float(vals.quantile(0.75)),
                "n":        len(vals),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement between k-means and Ward
# ---------------------------------------------------------------------------

def clustering_agreement(
    labels_km:   np.ndarray,
    labels_ward: np.ndarray,
) -> float:
    """
    Adjusted Rand Index measuring agreement between two clusterings.
    ARI = 1.0: perfect agreement. ARI = 0.0: random. ARI < 0: worse than random.
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_km, labels_ward)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Catchment typology clustering on flood-to-drought response metrics."
    )
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help=f"Force number of clusters. Default: auto-select from k={list(K_RANGE)}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load RCP8.5 metrics and attributes
    # ------------------------------------------------------------------
    print(f"Loading {CLUSTER_RCP} metrics ...", flush=True)
    metrics = load_metrics(CLUSTER_RCP)

    print("Loading CAMELS-GB v2 attributes ...", flush=True)
    attrs = load_attributes()

    # Restrict to common catchments
    common = metrics.index.intersection(attrs.index)
    print(f"Common catchments for clustering: {len(common)}", flush=True)

    # Extract clustering variables, drop any NaN rows
    X_df   = metrics.loc[common, CLUSTER_VARS].copy()
    valid  = X_df.dropna()
    n_drop = len(X_df) - len(valid)
    if n_drop > 0:
        print(f"  Dropped {n_drop} catchments with NaN in clustering variables",
              flush=True)

    gauge_ids = valid.index.values
    X_raw     = valid.values

    # Standardise
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"\nClustering on {len(gauge_ids)} catchments, "
          f"{len(CLUSTER_VARS)} variables: {CLUSTER_VARS}", flush=True)
    print(f"Scaling: mean={scaler.mean_.round(3)}, "
          f"std={scaler.scale_.round(3)}", flush=True)

    # ------------------------------------------------------------------
    # Optimal k selection
    # ------------------------------------------------------------------
    print("\nSelecting optimal k ...", flush=True)
    k_results = select_optimal_k(X_scaled, K_RANGE)
    k_results.to_parquet(OUTPUT_DIR / "optimal_k.parquet", index=False)
    print(f"Written: optimal_k.parquet", flush=True)

    # Best k by silhouette
    best_k_sil = int(k_results.loc[k_results["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette score: {best_k_sil}", flush=True)

    # Use forced k if provided, otherwise silhouette-optimal
    n_clusters = args.n_clusters if args.n_clusters else best_k_sil
    print(f"Using n_clusters = {n_clusters}", flush=True)

    # ------------------------------------------------------------------
    # K-means clustering
    # ------------------------------------------------------------------
    print(f"\nFitting k-means (k={n_clusters}, n_init=200) ...", flush=True)
    labels_km = kmeans_cluster(X_scaled, n_clusters)
    labels_km = relabel_by_delta_gap(labels_km, X_raw[:, 0], n_clusters)

    # Cluster sizes
    for k in range(n_clusters):
        n = (labels_km == k).sum()
        mean_gap = X_raw[labels_km == k, 0].mean()
        print(f"  Cluster {k}: n={n}, mean delta_gap={mean_gap:.2f}", flush=True)

    # Silhouette score for chosen k
    sil_final = silhouette_score(X_scaled, labels_km)
    print(f"  Final silhouette score: {sil_final:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Ward hierarchical clustering (cross-check)
    # ------------------------------------------------------------------
    print(f"\nFitting Ward hierarchical (k={n_clusters}) ...", flush=True)
    labels_ward = ward_cluster(X_scaled, n_clusters)
    labels_ward = relabel_by_delta_gap(labels_ward, X_raw[:, 0], n_clusters)

    ari = clustering_agreement(labels_km, labels_ward)
    print(f"  Adjusted Rand Index (k-means vs Ward): {ari:.3f}", flush=True)
    print(f"  (ARI > 0.7 indicates good agreement between methods)", flush=True)

    # ------------------------------------------------------------------
    # Build assignment DataFrame
    # ------------------------------------------------------------------
    assignments = pd.DataFrame({
        "gauge_id":      gauge_ids,
        "cluster_kmeans": labels_km,
        "cluster_ward":   labels_ward,
    }).set_index("gauge_id")

    # Add clustering variables (raw, unstandardised)
    for i, var in enumerate(CLUSTER_VARS):
        assignments[var] = X_raw[:, i]

    # Add key attributes
    attrs_sub = attrs.loc[
        attrs.index.isin(gauge_ids),
        CHARACTERISE_ATTRS + ["gauge_lat", "gauge_lon",
                               "gauge_easting", "gauge_northing",
                               "gauge_name"]
    ]
    assignments = assignments.join(attrs_sub, how="left")

    out = OUTPUT_DIR / "cluster_assignments.parquet"
    assignments.to_parquet(out)
    print(f"\nWritten: {out.name}  shape={assignments.shape}", flush=True)

    # ------------------------------------------------------------------
    # Cluster profiles across all RCPs
    # ------------------------------------------------------------------
    print("\nComputing cluster profiles across all RCPs ...", flush=True)
    profiles = compute_cluster_profiles(assignments, n_clusters)
    out = OUTPUT_DIR / "cluster_profiles.parquet"
    profiles.to_parquet(out, index=False)
    print(f"Written: {out.name}  shape={profiles.shape}", flush=True)

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    print("\n--- Cluster profiles (RCP8.5 delta metrics) ---")
    rcp85_profiles = profiles[profiles["rcp"] == "rcp85"]
    for var in CLUSTER_VARS:
        sub = rcp85_profiles[rcp85_profiles["variable"] == var]
        sub = sub.sort_values("cluster")
        print(f"\n  {var}:")
        for _, row in sub.iterrows():
            print(
                f"    Cluster {int(row['cluster'])}: "
                f"median={row['median']:+.3f}  "
                f"IQR=[{row['q25']:+.3f}, {row['q75']:+.3f}]  "
                f"n={int(row['n'])}"
            )

    print("\n--- Cluster attribute profiles ---")
    attr_profiles = profiles[profiles["rcp"] == "attribute"]
    for attr in ["baseflow_index", "aridity", "runoff_ratio", "dpsbar", "frac_snow"]:
        sub = attr_profiles[attr_profiles["variable"] == attr]
        sub = sub.sort_values("cluster")
        print(f"\n  {attr}:")
        for _, row in sub.iterrows():
            print(
                f"    Cluster {int(row['cluster'])}: "
                f"median={row['median']:.3f}  "
                f"IQR=[{row['q25']:.3f}, {row['q75']:.3f}]"
            )

    print(f"\nDone. Optimal k by silhouette: {best_k_sil}  "
          f"Used: {n_clusters}  ARI: {ari:.3f}")


if __name__ == "__main__":
    main()
