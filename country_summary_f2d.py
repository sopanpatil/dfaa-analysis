# country_summary_f2d.py
# -------------------------
# Mirrors country_summary_d2f_allrcp.py, adapted for FTD (Methods 2.4,
# Section 3.3, Table 2 / Supplementary Tables S5 and S6).
#
# Tests whether FTD's delta_drought_dur shows the same England-specific
# geographic concentration as DTF's (Section 3.3), alongside the other
# country-level comparisons (transition gap, drought severity, flood
# peak) reported in Table 2.
#
#   1. Per-country median/IQR/n for each delta metric.
#   2. Kruskal-Wallis test (nonparametric one-way ANOVA): does the metric
#      differ significantly across England/Scotland/Wales overall?
#   3. Pairwise Mann-Whitney U (Scotland vs England, Scotland vs Wales,
#      England vs Wales) for delta_gap AND delta_drought_dur specifically.
#
# Requires:
#   aggregated_output/catchment_metrics_<rcp>.parquet
#   camels_gb_v2_topographic_attributes.csv
#   great_britain_boundary.geojson
#
# Usage
# -----
#   python country_summary_f2d.py --rcp rcp85

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import geopandas as gpd
    HAVE_GEOPANDAS = True
except ImportError:
    HAVE_GEOPANDAS = False

F2D_DIR = Path("./aggregated_output")
ATTRS_DIR = Path(".")
OUTPUT_DIR = Path("./spatial_output")
BOUNDARY_PATH = Path("./great_britain_boundary.geojson")
TOPO_ATTR_FILE = "camels_gb_v2_topographic_attributes.csv"

# NOTE: mirrors DELTA_METRICS from aggregate_ensemble.py / catchment_attribute_regression.py.
# Must match the column names in catchment_metrics_<rcp>.parquet exactly.
DELTA_METRICS = ["delta_gap", "delta_freq", "delta_smdr", "delta_drought_sev", "delta_drought_dur",
                  "delta_flood_peak", "delta_sm_drought", "delta_whiplash"]
ALPHA = 0.05


def load_boundary():
    if not HAVE_GEOPANDAS or not BOUNDARY_PATH.exists():
        raise RuntimeError("geopandas + boundary file required for country assignment.")
    gdf = gpd.read_file(BOUNDARY_PATH)
    if gdf.crs is not None and gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf


def load_metrics_with_country(rcp: str, boundary) -> pd.DataFrame:
    metrics_path = F2D_DIR / f"catchment_metrics_{rcp}.parquet"
    metrics = pd.read_parquet(metrics_path)
    metrics.index = metrics.index.astype(str)
    metrics.index.name = "gauge_id"

    coords = pd.read_csv(ATTRS_DIR / TOPO_ATTR_FILE, index_col="gauge_id")
    coords.index = coords.index.astype(str)
    coords = coords[["gauge_easting", "gauge_northing"]]
    df = metrics.join(coords, how="inner")

    name_col = next((c for c in ["CTRY22NM", "CTRY21NM", "name", "NAME"] if c in boundary.columns), None)
    points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["gauge_easting"], df["gauge_northing"]), crs=boundary.crs
    )
    joined = gpd.sjoin(points, boundary[[name_col, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={name_col: "country"}).drop(columns=["index_right"], errors="ignore")

    n_unmatched = joined["country"].isna().sum()
    if n_unmatched:
        print(f"  NOTE: {n_unmatched} catchments unmatched to a country")
    return joined


def per_country_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for country in sorted(df["country"].dropna().unique()):
        vals = df.loc[df["country"] == country, metric].dropna()
        rows.append({
            "country": country, "n": len(vals),
            "median": vals.median(), "q25": vals.quantile(0.25), "q75": vals.quantile(0.75),
        })
    return pd.DataFrame(rows)


def kruskal_wallis_test(df: pd.DataFrame, metric: str) -> dict:
    groups = [df.loc[df["country"] == c, metric].dropna().values
              for c in df["country"].dropna().unique()]
    groups = [g for g in groups if len(g) > 0]
    h_stat, p_value = stats.kruskal(*groups)
    return {"metric": metric, "h_stat": h_stat, "pvalue": p_value, "significant": p_value < ALPHA}


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values, preserving input order.

    The three pairwise country comparisons within a given metric and RCP are
    treated as one post-hoc family (Methods 2.4), applied where the
    Kruskal-Wallis test for that metric is significant.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (n - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def pairwise_mannwhitney(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    countries = sorted(df["country"].dropna().unique())
    rows = []
    for i in range(len(countries)):
        for j in range(i + 1, len(countries)):
            c1, c2 = countries[i], countries[j]
            v1 = df.loc[df["country"] == c1, metric].dropna()
            v2 = df.loc[df["country"] == c2, metric].dropna()
            if len(v1) < 5 or len(v2) < 5:
                continue
            u_stat, pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            rows.append({
                "pair": f"{c1} vs {c2}", "median1": v1.median(), "median2": v2.median(),
                "n1": len(v1), "n2": len(v2), "pvalue": pval, "significant": pval < ALPHA,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pvalue_holm"] = holm_adjust(out["pvalue"].values)
    out["significant"] = out["pvalue_holm"] < ALPHA
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Per-country summary of f2d delta metrics.")
    p.add_argument("--rcp", default=None, help="Single RCP. Default: all four.")
    return p.parse_args()


def run_one_rcp(rcp, boundary):
    print(f"\n{'#'*90}\n# RCP: {rcp}\n{'#'*90}", flush=True)
    df = load_metrics_with_country(rcp, boundary)
    print(f"  {len(df)} catchments assigned", flush=True)

    all_summaries = []
    all_kw = []
    for metric in DELTA_METRICS:
        if metric not in df.columns:
            print(f"  NOTE: {metric} not found in catchment_metrics -- skipping")
            continue
        summary = per_country_summary(df, metric)
        summary["metric"] = metric
        all_summaries.append(summary)
        all_kw.append(kruskal_wallis_test(df, metric))

    print("\n" + "=" * 90)
    print(f"PER-COUNTRY SUMMARY -- f2d delta metrics, {rcp}")
    print("=" * 90)
    summary_df = pd.concat(all_summaries, ignore_index=True)[
        ["metric", "country", "n", "median", "q25", "q75"]
    ]
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / f"f2d_country_summary_{rcp}.csv", index=False)

    print("\n" + "=" * 90)
    print("KRUSKAL-WALLIS TEST -- does the metric differ significantly by country?")
    print("=" * 90)
    kw_df = pd.DataFrame(all_kw)
    print(kw_df.to_string(index=False))
    kw_df.to_csv(OUTPUT_DIR / f"f2d_kruskal_wallis_{rcp}.csv", index=False)

    # Pairwise tests: delta_gap and delta_drought_dur (original focus), plus
    # delta_freq and delta_flood_peak (checking whether their significant KW
    # result reflects genuinely new geographic structure, or is redundant with
    # the already-documented cluster/country correspondence).
    for metric in ["delta_gap", "delta_drought_dur", "delta_freq", "delta_flood_peak", "delta_drought_sev", "delta_smdr"]:
        if metric not in df.columns:
            continue
        print("\n" + "=" * 90)
        print(f"PAIRWISE COMPARISON -- {metric}")
        print("=" * 90)
        pairwise = pairwise_mannwhitney(df, metric)
        print(pairwise.to_string(index=False))
        pairwise.to_csv(OUTPUT_DIR / f"f2d_{metric}_pairwise_{rcp}.csv", index=False)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading boundary...", flush=True)
    boundary = load_boundary()

    rcps = [args.rcp] if args.rcp else ["rcp26", "rcp45", "rcp60", "rcp85"]
    for rcp in rcps:
        run_one_rcp(rcp, boundary)


if __name__ == "__main__":
    main()
