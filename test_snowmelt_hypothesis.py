# test_snowmelt_hypothesis.py
# ------------------------------
# Tests whether frac_snow (fraction of precipitation falling as snow,
# CAMELS-GB v2 climatic attribute) explains the Scotland/Pennines spatial
# concentration of Cluster 0 (large transition-gap slowdown, declining
# frequency) found in spatial_pattern_f2d.py.
#
# This is complementary to, not a duplicate of, frac_snow's role in
# catchment_attribute_regression.py: that script tests frac_snow jointly
# with the other 13 attributes in a single multivariate model, whereas
# this script tests frac_snow's bivariate relationship with each delta
# metric directly, and its association with cluster membership
# specifically -- results feeding Section 3.2 and Methods 2.7.
#
# Two tests:
#   1. Spearman correlation: frac_snow vs each delta metric, across all
#      four RCPs (same nonparametric approach as the existing regression
#      script, for consistency).
#   2. Cluster comparison: is frac_snow itself significantly higher in
#      Cluster 0 than Cluster 1? Mann-Whitney U test (nonparametric,
#      doesn't assume normality) rather than a t-test.
#
# Requires:
#   aggregated_output/catchment_metrics_<rcp>.parquet
#   camels_gb_v2_climatic_attributes.csv
#   typology_output/cluster_assignments.parquet
#
# Usage
# -----
#   python test_snowmelt_hypothesis.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AGGREGATED_DIR = Path("./aggregated_output")
ATTRS_DIR = Path(".")
TYPOLOGY_DIR = Path("./typology_output")
OUTPUT_DIR = Path("./spatial_output")

CLIMATIC_ATTR_FILE = "camels_gb_v2_climatic_attributes.csv"
RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
DELTA_METRICS = ["delta_gap", "delta_freq", "delta_smdr", "delta_sm_drought",
                  "delta_whiplash", "delta_flood_peak", "delta_drought_sev"]

ALPHA = 0.05


def load_frac_snow() -> pd.Series:
    path = ATTRS_DIR / CLIMATIC_ATTR_FILE
    df = pd.read_csv(path, index_col="gauge_id")
    if "frac_snow" not in df.columns:
        raise ValueError(f"frac_snow not found in {CLIMATIC_ATTR_FILE}. Columns: {list(df.columns)}")
    return df["frac_snow"]


def load_metrics(rcp: str) -> pd.DataFrame:
    path = AGGREGATED_DIR / f"catchment_metrics_{rcp}.parquet"
    df = pd.read_parquet(path)
    df.index = df.index.astype(int)
    df.index.name = "gauge_id"
    return df


def load_clusters() -> pd.DataFrame:
    path = TYPOLOGY_DIR / "cluster_assignments.parquet"
    df = pd.read_parquet(path)
    df.index.name = "gauge_id"
    return df


# ---------------------------------------------------------------------------
# Test 1: Spearman correlation, frac_snow vs delta metrics, all RCPs
# ---------------------------------------------------------------------------

def correlation_test(frac_snow: pd.Series) -> pd.DataFrame:
    rows = []
    for rcp in RCPS:
        metrics = load_metrics(rcp)
        common = metrics.index.intersection(frac_snow.index)
        x = frac_snow.loc[common].values

        for metric in DELTA_METRICS:
            if metric not in metrics.columns:
                continue
            y = metrics.loc[common, metric].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 30:
                continue
            rho, pval = stats.spearmanr(x[valid], y[valid])
            rows.append({
                "rcp": rcp, "delta_metric": metric,
                "rho": rho, "pvalue": pval, "n": int(valid.sum()),
                "significant": pval < ALPHA,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2: frac_snow, Cluster 0 vs Cluster 1 (Mann-Whitney U)
# ---------------------------------------------------------------------------

def cluster_comparison_test(frac_snow: pd.Series, clusters: pd.DataFrame) -> dict:
    common = clusters.index.intersection(frac_snow.index)
    merged = pd.DataFrame({
        "frac_snow": frac_snow.loc[common],
        "cluster": clusters.loc[common, "cluster_kmeans"],
    })

    c0 = merged.loc[merged["cluster"] == 0, "frac_snow"].dropna()
    c1 = merged.loc[merged["cluster"] == 1, "frac_snow"].dropna()

    u_stat, pval = stats.mannwhitneyu(c0, c1, alternative="two-sided")

    return {
        "cluster0_n": len(c0), "cluster0_median": c0.median(),
        "cluster0_q25": c0.quantile(0.25), "cluster0_q75": c0.quantile(0.75),
        "cluster1_n": len(c1), "cluster1_median": c1.median(),
        "cluster1_q25": c1.quantile(0.25), "cluster1_q75": c1.quantile(0.75),
        "mannwhitney_u": u_stat, "pvalue": pval, "significant": pval < ALPHA,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading frac_snow...", flush=True)
    frac_snow = load_frac_snow()
    print(f"  {len(frac_snow)} catchments, range [{frac_snow.min():.3f}, {frac_snow.max():.3f}]", flush=True)

    print("\n" + "=" * 80)
    print("TEST 1: Spearman correlation -- frac_snow vs delta metrics, all RCPs")
    print("=" * 80)
    corr_df = correlation_test(frac_snow)
    out_path = OUTPUT_DIR / "snowmelt_correlation_test.csv"
    corr_df.to_csv(out_path, index=False)
    print(corr_df.to_string(index=False))
    print(f"\nWritten: {out_path}")

    print("\n" + "=" * 80)
    print("TEST 2: frac_snow, Cluster 0 (large slowdown) vs Cluster 1 (moderate)")
    print("=" * 80)
    clusters = load_clusters()
    result = cluster_comparison_test(frac_snow, clusters)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    result_df = pd.DataFrame([result])
    out_path2 = OUTPUT_DIR / "snowmelt_cluster_comparison.csv"
    result_df.to_csv(out_path2, index=False)
    print(f"\nWritten: {out_path2}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if result["significant"] and result["cluster0_median"] > result["cluster1_median"]:
        print("  Cluster 0 (Scotland/Pennines-concentrated, large slowdown) has significantly")
        print("  HIGHER frac_snow than Cluster 1 -- consistent with the snowmelt hypothesis.")
    elif result["significant"] and result["cluster0_median"] < result["cluster1_median"]:
        print("  Cluster 0 has significantly LOWER frac_snow than Cluster 1 -- opposite of")
        print("  the snowmelt hypothesis. Worth reconsidering the physical mechanism.")
    else:
        print("  No significant difference in frac_snow between clusters -- snowmelt does")
        print("  not appear to explain the cluster split on its own.")


if __name__ == "__main__":
    main()
