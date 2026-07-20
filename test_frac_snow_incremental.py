# test_frac_snow_incremental.py
# --------------------------------
# Nested F-test backing Methods 2.7 / Results 3.4: does frac_snow add
# independent explanatory power for delta_gap / delta_freq beyond the
# other 14 attributes in catchment_attribute_regression.py's 15-attribute
# SELECTED_ATTRIBUTES set (Methods 2.6), given that frac_snow is
# plausibly correlated with several of them -- especially dpsbar
# (terrain slope, a rough elevation proxy) and the climatic attributes
# (aridity, p_mean), since colder/wetter high ground gets more snow?
# If so, frac_snow's apparent effect could be riding on signal already
# captured by those, rather than adding new information.
#
# This tests three things, using the SAME 14 non-frac_snow attributes and
# log-transform conventions as catchment_attribute_regression.py, plus
# frac_snow itself, for consistency:
#
#   1. Pairwise Spearman correlation: frac_snow vs each of the other 14
#      SELECTED_ATTRIBUTES, to see what it's most redundant with.
#   2. Variance Inflation Factor (VIF) for frac_snow against the other 14
#      predictors -- VIF > 5-10 signals problematic multicollinearity.
#   3. Nested OLS F-test: does adding frac_snow to the 14-attribute
#      restricted model significantly improve R² for delta_gap /
#      delta_freq, beyond what the other 14 attributes already explain?
#
# Requires:
#   aggregated_output/catchment_metrics_<rcp>.parquet
#   camels_gb_v2_{hydrologic,climatic,hydrogeology,topographic}_attributes.csv
#
# Usage
# -----
#   python test_frac_snow_incremental.py --rcp rcp85
#   python test_frac_snow_incremental.py --all

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

AGGREGATED_DIR = Path("./aggregated_output")
ATTRS_DIR = Path(".")
OUTPUT_DIR = Path("./spatial_output")

ATTR_FILES = {
    "hydrologic":   "camels_gb_v2_hydrologic_attributes.csv",
    "climatic":     "camels_gb_v2_climatic_attributes.csv",
    "hydrogeology": "camels_gb_v2_hydrogeology_attributes.csv",
    "topographic":  "camels_gb_v2_topographic_attributes.csv",
}

# Same as catchment_attribute_regression.py's SELECTED_ATTRIBUTES
EXISTING_ATTRIBUTES = [
    "baseflow_index", "runoff_ratio", "slope_fdc", "high_q_freq", "low_q_freq",
    "aridity", "p_mean", "p_seasonality", "high_prec_freq",
    "frac_high_perc", "frac_low_perc", "no_gw_perc",
    "area", "dpsbar",
]
LOG_TRANSFORM_ATTRS = {"area", "high_q_freq", "low_q_freq", "high_prec_freq"}

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
TARGET_METRICS = ["delta_gap", "delta_freq"]
ALPHA = 0.05


def load_attributes() -> pd.DataFrame:
    frames = []
    for group, fname in ATTR_FILES.items():
        fpath = ATTRS_DIR / fname
        df = pd.read_csv(fpath, index_col="gauge_id")
        frames.append(df)
    attrs = pd.concat(frames, axis=1)

    keep = EXISTING_ATTRIBUTES + ["frac_snow"]
    available = [a for a in keep if a in attrs.columns]
    missing = [a for a in keep if a not in attrs.columns]
    if missing:
        print(f"  WARNING: missing attributes: {missing}")
    attrs = attrs[available].copy()

    for col in LOG_TRANSFORM_ATTRS:
        if col in attrs.columns:
            attrs[f"log_{col}"] = np.log1p(attrs[col])
            attrs = attrs.drop(columns=[col])

    return attrs


def load_metrics(rcp: str) -> pd.DataFrame:
    path = AGGREGATED_DIR / f"catchment_metrics_{rcp}.parquet"
    df = pd.read_parquet(path)
    df.index = df.index.astype(int)
    df.index.name = "gauge_id"
    return df


# ---------------------------------------------------------------------------
# Test 1: pairwise correlation with existing attributes
# ---------------------------------------------------------------------------

def frac_snow_redundancy(attrs: pd.DataFrame) -> pd.DataFrame:
    predictor_cols = [c for c in attrs.columns if c != "frac_snow"]
    rows = []
    for col in predictor_cols:
        sub = attrs[["frac_snow", col]].dropna()
        rho, pval = stats.spearmanr(sub["frac_snow"], sub[col])
        rows.append({"attribute": col, "rho_with_frac_snow": rho, "pvalue": pval,
                      "significant": pval < ALPHA})
    return pd.DataFrame(rows).sort_values("rho_with_frac_snow", key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Test 2: VIF for frac_snow against the existing predictors
# ---------------------------------------------------------------------------

def compute_vif(X: np.ndarray, target_idx: int) -> float:
    """VIF for column target_idx: 1 / (1 - R^2) of regressing it on all others."""
    y = X[:, target_idx]
    others = np.delete(X, target_idx, axis=1)
    others_aug = np.column_stack([np.ones(len(others)), others])
    beta, _, _, _ = np.linalg.lstsq(others_aug, y, rcond=None)
    y_pred = others_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return 1 / (1 - r2) if r2 < 1 else np.inf


# ---------------------------------------------------------------------------
# Test 3: nested OLS F-test (with vs without frac_snow)
# ---------------------------------------------------------------------------

def ols_fit(X: np.ndarray, y: np.ndarray):
    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    return beta, ss_res, n, p


def nested_f_test(attrs: pd.DataFrame, metrics: pd.DataFrame, target: str) -> dict:
    common = attrs.index.intersection(metrics.index)
    a = attrs.loc[common]
    y_full = metrics.loc[common, target].values

    existing_cols = [c for c in a.columns if c != "frac_snow"]
    valid = ~(np.isnan(y_full) | a[existing_cols + ["frac_snow"]].isna().any(axis=1).values)
    a_valid = a.loc[valid]
    y = y_full[valid]

    scaler = StandardScaler()

    # Restricted model: existing attributes only
    X_restricted = scaler.fit_transform(a_valid[existing_cols].values)
    _, ss_res_restricted, n, p_restricted = ols_fit(X_restricted, y)

    # Full model: existing + frac_snow
    X_full = scaler.fit_transform(a_valid[existing_cols + ["frac_snow"]].values)
    beta_full, ss_res_full, _, p_full = ols_fit(X_full, y)

    # Partial F-test for the added variable (frac_snow), df = (1, n - p_full - 1)
    df1 = p_full - p_restricted
    df2 = n - p_full - 1
    f_stat = ((ss_res_restricted - ss_res_full) / df1) / (ss_res_full / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_restricted = 1 - ss_res_restricted / ss_tot
    r2_full = 1 - ss_res_full / ss_tot

    frac_snow_coef = beta_full[-1]  # last column is frac_snow (standardised coef)

    return {
        "target": target, "n": n,
        "r2_restricted": r2_restricted, "r2_full": r2_full,
        "r2_gain": r2_full - r2_restricted,
        "frac_snow_std_coef": frac_snow_coef,
        "f_stat": f_stat, "pvalue": p_value, "significant": p_value < ALPHA,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Test incremental value of frac_snow beyond existing attributes.")
    p.add_argument("--rcp", default=None, choices=RCPS)
    p.add_argument("--all", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rcps = RCPS if args.all else [args.rcp or "rcp85"]

    print("Loading attributes...", flush=True)
    attrs = load_attributes()
    print(f"  {len(attrs)} catchments, {len(attrs.columns)} attributes (incl. frac_snow)", flush=True)

    # Restrict the redundancy/VIF diagnostics (Tests 1-2) to the same catchment
    # subsample used by the nested F-test (Test 3), rather than the full,
    # unfiltered 671-catchment attribute table. This keeps every reported
    # frac_snow diagnostic anchored to the sample the transition-gap analysis
    # actually uses. RCP8.5 is used as the reference sample here since it is
    # also the RCP the catchment typology (Section 2.5) is defined on;
    # attrs_full (unrestricted) is retained separately only for reference/QA,
    # not for any reported statistic.
    ref_metrics = load_metrics("rcp85")
    analysis_idx = attrs.index.intersection(ref_metrics.index)
    attrs_full = attrs  # kept for reference; not used in reported diagnostics
    attrs = attrs.loc[analysis_idx]
    print(f"  Restricting Tests 1-2 to the RCP8.5 analysis subsample: "
          f"n={len(attrs)} (full attribute table was n={len(attrs_full)})", flush=True)

    print("\n" + "=" * 80)
    print("TEST 1: frac_snow correlation with EXISTING attributes (redundancy check)")
    print(f"[n={len(attrs)}, RCP8.5 analysis subsample]")
    print("=" * 80)
    redundancy = frac_snow_redundancy(attrs)
    print(redundancy.to_string(index=False))
    redundancy.to_csv(OUTPUT_DIR / "frac_snow_redundancy.csv", index=False)

    print("\n" + "=" * 80)
    print("TEST 2: Variance Inflation Factor for frac_snow")
    print(f"[n={len(attrs)}, RCP8.5 analysis subsample]")
    print("=" * 80)
    valid_attrs = attrs.dropna()
    X = StandardScaler().fit_transform(valid_attrs.values)
    frac_snow_idx = list(valid_attrs.columns).index("frac_snow")
    vif = compute_vif(X, frac_snow_idx)
    print(f"  VIF(frac_snow) = {vif:.2f}  (n={len(valid_attrs)})  "
          f"({'HIGH multicollinearity' if vif > 5 else 'acceptable' if vif < 5 else 'borderline'})")
    pd.DataFrame([{"n": len(valid_attrs), "vif_frac_snow": vif}]).to_csv(
        OUTPUT_DIR / "frac_snow_vif.csv", index=False)

    # Test 3 (nested F-test) restricts to each RCP's own metrics table
    # internally, exactly as before -- it uses attrs_full so that every RCP
    # gets its own correct (RCP-specific) sample rather than being pre-cut
    # to RCP8.5's catchments.
    attrs = attrs_full

    print("\n" + "=" * 80)
    print("TEST 3: Nested F-test -- does frac_snow add explanatory power?")
    print("=" * 80)
    all_results = []
    for rcp in rcps:
        metrics = load_metrics(rcp)
        for target in TARGET_METRICS:
            if target not in metrics.columns:
                continue
            result = nested_f_test(attrs, metrics, target)
            result["rcp"] = rcp
            all_results.append(result)
            sig = "***" if result["pvalue"] < 0.001 else "**" if result["pvalue"] < 0.01 else "*" if result["significant"] else ""
            print(f"  {rcp} / {target}: R2 {result['r2_restricted']:.3f} -> {result['r2_full']:.3f} "
                  f"(gain={result['r2_gain']:+.3f})  std_coef={result['frac_snow_std_coef']:+.3f}  "
                  f"F={result['f_stat']:.2f}  p={result['pvalue']:.4f} {sig}", flush=True)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_DIR / "frac_snow_nested_f_test.csv", index=False)
    print(f"\nWritten: {OUTPUT_DIR / 'frac_snow_nested_f_test.csv'}")


if __name__ == "__main__":
    main()
