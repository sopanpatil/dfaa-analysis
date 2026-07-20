# catchment_attribute_regression_d2f.py
# ----------------------------------------
# d2f equivalent of catchment_attribute_regression.py. Identical method
# (Spearman, OLS, random forest) applied to d2f's delta metrics instead
# of f2d's.
#
# frac_snow is included in SELECTED_ATTRIBUTES for both the FTD and DTF
# regressions (Methods 2.6), given the role of snow storage established
# in Section 3.4 / Methods 2.7 for both transition directions.
#
# Inputs
# ------
#   d2f_aggregated_output/d2f_catchment_metrics_<rcp>.parquet
#   camels_gb_v2_{hydrologic,climatic,hydrogeology,topographic}_attributes.csv
#
# Outputs (in d2f_regression_output/)
# --------------------------------
#   spearman_<rcp>.parquet
#   ols_<rcp>.parquet
#   rf_importance_<rcp>.parquet
#   combined_attributes.parquet
#
# Usage
# -----
#   python catchment_attribute_regression_d2f.py
#   python catchment_attribute_regression_d2f.py --rcp rcp85

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D2F_DIR    = Path("./d2f_aggregated_output")
ATTRS_DIR  = Path(".")
OUTPUT_DIR = Path("./d2f_regression_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]

ATTR_FILES = {
    "hydrologic":   "camels_gb_v2_hydrologic_attributes.csv",
    "climatic":     "camels_gb_v2_climatic_attributes.csv",
    "hydrogeology": "camels_gb_v2_hydrogeology_attributes.csv",
    "topographic":  "camels_gb_v2_topographic_attributes.csv",
}

SELECTED_ATTRIBUTES = [
    # Hydrological
    "baseflow_index", "runoff_ratio", "slope_fdc", "high_q_freq", "low_q_freq",
    # Climatic
    "aridity", "p_mean", "p_seasonality", "high_prec_freq",
    "frac_snow",  # snow fraction -- included for both FTD and DTF regressions
    # Hydrogeology
    "frac_high_perc", "frac_low_perc", "no_gw_perc",
    # Topographic
    "area", "dpsbar",
]

# d2f's delta metrics (see aggregate_ensemble_d2f.py for definitions)
DELTA_METRICS = [
    "delta_gap",
    "delta_freq",
    "delta_recovery_rate",
    "delta_sm_drought_end",
    "delta_sm_flood_onset",
    "delta_flood_peak",
    "delta_drought_dur",
]

LOG_TRANSFORM_ATTRS = {"area", "high_q_freq", "low_q_freq", "high_prec_freq"}

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Load and merge CAMELS-GB v2 attributes (identical to f2d script)
# ---------------------------------------------------------------------------

def load_attributes(attrs_dir: Path) -> pd.DataFrame:
    frames = []
    for group, fname in ATTR_FILES.items():
        fpath = attrs_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Attribute file not found: {fpath}")
        df = pd.read_csv(fpath, index_col="gauge_id")
        frames.append(df)
        print(f"  Loaded {group}: {df.shape[1]} attributes, {len(df)} catchments", flush=True)

    attrs = pd.concat(frames, axis=1)

    available = [a for a in SELECTED_ATTRIBUTES if a in attrs.columns]
    missing = [a for a in SELECTED_ATTRIBUTES if a not in attrs.columns]
    if missing:
        print(f"  WARNING: attributes not found and skipped: {missing}", flush=True)

    attrs = attrs[available].copy()

    for col in LOG_TRANSFORM_ATTRS:
        if col in attrs.columns:
            attrs[f"log_{col}"] = np.log1p(attrs[col])
            attrs = attrs.drop(columns=[col])
            print(f"  Log-transformed: {col} -> log_{col}", flush=True)

    print(f"  Combined attribute table: {attrs.shape}", flush=True)
    return attrs


# ---------------------------------------------------------------------------
# Spearman correlation (identical method to f2d script)
# ---------------------------------------------------------------------------

def spearman_analysis(metrics_df: pd.DataFrame, attrs_df: pd.DataFrame, rcp: str) -> pd.DataFrame:
    common = metrics_df.index.intersection(attrs_df.index)
    m = metrics_df.loc[common]
    a = attrs_df.loc[common]

    rows = []
    for delta in DELTA_METRICS:
        if delta not in m.columns:
            continue
        y = m[delta].values

        for attr in a.columns:
            x = a[attr].values
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 30:
                continue
            rho, pval = stats.spearmanr(x[valid], y[valid])
            rows.append({"rcp": rcp, "delta_metric": delta, "attribute": attr,
                        "rho": rho, "pvalue": pval, "n": int(valid.sum()),
                        "significant": pval < ALPHA})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OLS multiple linear regression (identical method to f2d script)
# ---------------------------------------------------------------------------

def ols_analysis(metrics_df: pd.DataFrame, attrs_df: pd.DataFrame, rcp: str) -> pd.DataFrame:
    common = metrics_df.index.intersection(attrs_df.index)
    m = metrics_df.loc[common]
    a = attrs_df.loc[common]

    scaler = StandardScaler()
    rows = []

    for delta in DELTA_METRICS:
        if delta not in m.columns:
            continue

        y = m[delta].values
        attr_cols = a.columns.tolist()
        X_raw = a[attr_cols].values
        valid = ~(np.isnan(y) | np.any(np.isnan(X_raw), axis=1))

        if valid.sum() < 50:
            continue

        X = scaler.fit_transform(X_raw[valid])
        y_v = y[valid]

        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        y_pred = X_aug @ beta
        ss_res = np.sum((y_v - y_pred) ** 2)
        ss_tot = np.sum((y_v - y_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        sigma2 = ss_res / (n - p - 1) if n > p + 1 else np.nan
        try:
            cov = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
            se = np.sqrt(np.diag(cov))
            t_stat = beta / se
            pvals = 2 * t_dist.sf(np.abs(t_stat), df=n - p - 1)
        except np.linalg.LinAlgError:
            pvals = np.full(len(beta), np.nan)

        for i, attr in enumerate(attr_cols):
            rows.append({"rcp": rcp, "delta_metric": delta, "attribute": attr,
                        "coef": float(beta[i + 1]), "pvalue": float(pvals[i + 1]),
                        "r2_adj": float(r2_adj), "n": int(valid.sum()),
                        "significant": float(pvals[i + 1]) < ALPHA})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Random forest variable importance (identical method to f2d script)
# ---------------------------------------------------------------------------

def rf_importance_analysis(metrics_df: pd.DataFrame, attrs_df: pd.DataFrame, rcp: str,
                             n_trees: int = 500) -> pd.DataFrame:
    common = metrics_df.index.intersection(attrs_df.index)
    m = metrics_df.loc[common]
    a = attrs_df.loc[common]

    attr_cols = a.columns.tolist()
    rows = []

    for delta in DELTA_METRICS:
        if delta not in m.columns:
            continue

        y = m[delta].values
        X_raw = a[attr_cols].values
        valid = ~(np.isnan(y) | np.any(np.isnan(X_raw), axis=1))

        if valid.sum() < 50:
            continue

        rf = RandomForestRegressor(n_estimators=n_trees, max_features="sqrt",
                                     random_state=42, n_jobs=-1, oob_score=True)
        rf.fit(X_raw[valid], y[valid])

        train_r2 = rf.score(X_raw[valid], y[valid])
        oob_r2 = rf.oob_score_

        for attr, imp in zip(attr_cols, rf.feature_importances_):
            rows.append({"rcp": rcp, "delta_metric": delta, "attribute": attr,
                        "importance": float(imp), "oob_r2": float(oob_r2)})

        print(f"    RF {delta}: n={valid.sum()}, train R²={train_r2:.3f}  OOB R²={oob_r2:.3f}", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rcp(rcp: str, attrs_df: pd.DataFrame) -> None:
    print(f"\n{'='*55}", flush=True)
    print(f"  RCP: {rcp}", flush=True)
    print(f"{'='*55}", flush=True)

    metrics_path = D2F_DIR / f"d2f_catchment_metrics_{rcp}.parquet"
    if not metrics_path.exists():
        print(f"  ERROR: {metrics_path} not found", flush=True)
        return

    metrics_df = pd.read_parquet(metrics_path)
    metrics_df.index = metrics_df.index.astype(int)

    delta_cols = [c for c in metrics_df.columns if c.startswith("delta_")]
    metrics_df = metrics_df[delta_cols]

    common = metrics_df.index.intersection(attrs_df.index)
    print(f"  Common catchments: {len(common)}", flush=True)

    print("  Running Spearman correlations ...", flush=True)
    spearman_df = spearman_analysis(metrics_df, attrs_df, rcp)
    out = OUTPUT_DIR / f"spearman_{rcp}.parquet"
    spearman_df.to_parquet(out, index=False)
    print(f"  Written: {out.name}", flush=True)

    for delta in ["delta_gap", "delta_recovery_rate"]:
        sub = spearman_df[spearman_df["delta_metric"] == delta].copy()
        sub = sub.sort_values("rho", key=abs, ascending=False).head(5)
        print(f"\n  Top Spearman correlates for {delta}:")
        for _, row in sub.iterrows():
            sig = "*" if row["significant"] else ""
            print(f"    {row['attribute']:25s} rho={row['rho']:+.3f}  p={row['pvalue']:.3f}{sig}")

    print("\n  Running OLS regression ...", flush=True)
    ols_df = ols_analysis(metrics_df, attrs_df, rcp)
    out = OUTPUT_DIR / f"ols_{rcp}.parquet"
    ols_df.to_parquet(out, index=False)
    print(f"  Written: {out.name}", flush=True)

    print("\n  OLS adjusted R² per delta metric:")
    for delta in DELTA_METRICS:
        sub = ols_df[ols_df["delta_metric"] == delta]
        if not sub.empty:
            r2 = sub["r2_adj"].iloc[0]
            print(f"    {delta:25s} adj-R²={r2:.3f}")

    print("\n  Running random forest importance ...", flush=True)
    rf_df = rf_importance_analysis(metrics_df, attrs_df, rcp)
    out = OUTPUT_DIR / f"rf_importance_{rcp}.parquet"
    rf_df.to_parquet(out, index=False)
    print(f"  Written: {out.name}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regress d2f transition change metrics against CAMELS-GB v2 attributes."
    )
    parser.add_argument("--rcp", choices=RCPS, default=None, help="Single RCP. Default: all four.")
    parser.add_argument("--attrs-dir", type=Path, default=ATTRS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CAMELS-GB v2 attributes ...", flush=True)
    attrs_df = load_attributes(args.attrs_dir)

    combined_path = OUTPUT_DIR / "combined_attributes.parquet"
    if not combined_path.exists():
        attrs_df.to_parquet(combined_path)
        print(f"Written: {combined_path.name}", flush=True)

    rcps_to_run = [args.rcp] if args.rcp else RCPS
    for rcp in rcps_to_run:
        run_rcp(rcp, attrs_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
