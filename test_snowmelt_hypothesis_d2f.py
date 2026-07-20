# test_snowmelt_hypothesis_d2f.py
# ----------------------------------
# DTF equivalent of test_snowmelt_hypothesis.py. Uses the
# England/Scotland/Wales country split (rather than the DTF typology
# cluster labels in catchment_typology_d2f.py) as the grouping variable,
# since Table 2 / Supplementary Table S5 establishes a strong,
# significant country-level divide in delta_gap independent of the
# cluster-based typology (Section 3.3).
#
# Three tests:
#   1. Spearman correlation: frac_snow vs each DTF delta metric, across
#      all four RCPs (same approach as the FTD snowmelt test).
#   2. Scotland vs England: is frac_snow itself significantly higher in
#      Scotland than England? (Mann-Whitney U -- mirrors the FTD Cluster 0
#      vs Cluster 1 comparison, but for the country split.)
#   3. Within-country correlation: does frac_snow correlate with delta_gap
#      WITHIN Scotland specifically, and WITHIN England specifically? If
#      snow only explains the average LEVEL difference between countries
#      but not variation within Scotland, that's a different (weaker)
#      claim than snow explaining catchment-level variation everywhere.
#
# Requires:
#   d2f_aggregated_output/d2f_catchment_metrics_<rcp>.parquet
#   camels_gb_v2_climatic_attributes.csv (frac_snow)
#   camels_gb_v2_topographic_attributes.csv (coordinates)
#   great_britain_boundary.geojson
#
# Usage
# -----
#   python test_snowmelt_hypothesis_d2f.py --rcp rcp85

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

D2F_DIR = Path("./d2f_aggregated_output")
ATTRS_DIR = Path(".")
OUTPUT_DIR = Path("./d2f_spatial_output")
BOUNDARY_PATH = Path("./great_britain_boundary.geojson")
TOPO_ATTR_FILE = "camels_gb_v2_topographic_attributes.csv"
CLIMATIC_ATTR_FILE = "camels_gb_v2_climatic_attributes.csv"

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
DELTA_METRICS = ["delta_gap", "delta_freq", "delta_recovery_rate", "delta_drought_dur",
                  "delta_flood_peak", "delta_sm_drought_end", "delta_sm_flood_onset"]
ALPHA = 0.05


def load_frac_snow() -> pd.Series:
    df = pd.read_csv(ATTRS_DIR / CLIMATIC_ATTR_FILE, index_col="gauge_id")
    return df["frac_snow"]


def load_metrics(rcp: str) -> pd.DataFrame:
    df = pd.read_parquet(D2F_DIR / f"d2f_catchment_metrics_{rcp}.parquet")
    df.index = df.index.astype(int)
    df.index.name = "gauge_id"
    return df


def load_boundary():
    if not HAVE_GEOPANDAS or not BOUNDARY_PATH.exists():
        return None
    gdf = gpd.read_file(BOUNDARY_PATH)
    if gdf.crs is not None and gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf


def assign_countries(index_with_coords: pd.DataFrame, boundary) -> pd.Series:
    name_col = next((c for c in ["CTRY22NM", "CTRY21NM", "name", "NAME"] if c in boundary.columns), None)
    points = gpd.GeoDataFrame(
        index_with_coords,
        geometry=gpd.points_from_xy(index_with_coords["gauge_easting"], index_with_coords["gauge_northing"]),
        crs=boundary.crs,
    )
    joined = gpd.sjoin(points, boundary[[name_col, "geometry"]], how="left", predicate="within")
    return joined[name_col].rename("country")


# ---------------------------------------------------------------------------
# Test 1: Spearman correlation, all RCPs
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
            rows.append({"rcp": rcp, "delta_metric": metric, "rho": rho, "pvalue": pval,
                         "n": int(valid.sum()), "significant": pval < ALPHA})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2: frac_snow, Scotland vs England
# ---------------------------------------------------------------------------

def country_frac_snow_test(frac_snow: pd.Series, country: pd.Series) -> dict:
    common = country.index.intersection(frac_snow.index)
    merged = pd.DataFrame({"frac_snow": frac_snow.loc[common], "country": country.loc[common]})

    scot = merged.loc[merged["country"] == "Scotland", "frac_snow"].dropna()
    eng = merged.loc[merged["country"] == "England", "frac_snow"].dropna()

    u_stat, pval = stats.mannwhitneyu(scot, eng, alternative="two-sided")
    return {
        "scotland_n": len(scot), "scotland_median": scot.median(),
        "england_n": len(eng), "england_median": eng.median(),
        "pvalue": pval, "significant": pval < ALPHA,
    }


# ---------------------------------------------------------------------------
# Test 3: within-country correlation, frac_snow vs delta_gap
# ---------------------------------------------------------------------------

def within_country_correlation(frac_snow: pd.Series, country: pd.Series, metrics: pd.DataFrame,
                                 target: str = "delta_gap") -> pd.DataFrame:
    common = country.index.intersection(frac_snow.index).intersection(metrics.index)
    merged = pd.DataFrame({
        "frac_snow": frac_snow.loc[common], "country": country.loc[common],
        target: metrics.loc[common, target],
    }).dropna()

    rows = []
    for c in ["England", "Scotland", "Wales"]:
        sub = merged[merged["country"] == c]
        if len(sub) < 10:
            rows.append({"country": c, "n": len(sub), "rho": np.nan, "pvalue": np.nan,
                         "note": "n<10, too few for a reliable test"})
            continue
        rho, pval = stats.spearmanr(sub["frac_snow"], sub[target])
        rows.append({"country": c, "n": len(sub), "rho": rho, "pvalue": pval,
                     "significant": pval < ALPHA, "note": ""})
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Test snowmelt hypothesis against d2f delta_gap.")
    p.add_argument("--rcp", default="rcp85")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frac_snow...", flush=True)
    frac_snow = load_frac_snow()

    print("\n" + "=" * 80)
    print("TEST 1: Spearman correlation -- frac_snow vs d2f delta metrics, all RCPs")
    print("=" * 80)
    corr_df = correlation_test(frac_snow)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(OUTPUT_DIR / "d2f_snowmelt_correlation.csv", index=False)

    boundary = load_boundary()
    if boundary is None:
        print("\nNo boundary/geopandas available -- skipping country-based tests.")
        return

    rcp85_metrics = load_metrics(args.rcp)
    coords = pd.read_csv(ATTRS_DIR / TOPO_ATTR_FILE, index_col="gauge_id")[
        ["gauge_easting", "gauge_northing"]
    ]
    with_coords = rcp85_metrics.join(coords, how="inner")
    country = assign_countries(with_coords, boundary)

    print("\n" + "=" * 80)
    print(f"TEST 2: frac_snow -- Scotland vs England ({args.rcp} catchment set)")
    print("=" * 80)
    result2 = country_frac_snow_test(frac_snow, country)
    for k, v in result2.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print(f"TEST 3: WITHIN-country correlation -- frac_snow vs delta_gap ({args.rcp})")
    print("=" * 80)
    result3 = within_country_correlation(frac_snow, country, rcp85_metrics, target="delta_gap")
    print(result3.to_string(index=False))
    result3.to_csv(OUTPUT_DIR / f"d2f_snowmelt_within_country_{args.rcp}.csv", index=False)

    print("\n  Interpretation guide:")
    print("  - If Scotland has significantly higher frac_snow (Test 2) AND a significant")
    print("    within-Scotland correlation (Test 3), snow explains both WHY Scotland differs")
    print("    on average AND which Scottish catchments are most affected.")
    print("  - If Test 2 is significant but Test 3 is not, snow explains the country-level")
    print("    average difference but not catchment-level variation within Scotland --")
    print("    a weaker, more limited claim.")


if __name__ == "__main__":
    main()
