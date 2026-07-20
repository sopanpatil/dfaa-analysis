# catchment_attribute_regression.py
# ----------------------------------
# Regress flood-to-drought transition change metrics (delta_gap,
# delta_sm_drought, delta_freq, delta_whiplash) against CAMELS-GB v2
# static catchment attributes to identify landscape controls.
#
# Method: Spearman rank correlation (non-parametric, no distributional
# assumptions) between each delta metric and each catchment attribute.
# Multiple linear regression (OLS) for combined predictors.
# Random forest variable importance as a non-linear cross-check.
#
# Why Spearman rather than Pearson?
#   Delta metrics and catchment attributes are often skewed and
#   non-normally distributed. Spearman is more robust and is standard
#   in large-sample catchment hydrology (e.g. Addor et al. 2018).
#
# Inputs
# ------
#   aggregated_output/catchment_metrics_<rcp>.parquet  (one per RCP)
#   camels_gb_v2_hydrologic_attributes.csv
#   camels_gb_v2_climatic_attributes.csv
#   camels_gb_v2_hydrogeology_attributes.csv
#   camels_gb_v2_topographic_attributes.csv
#
# Outputs (in regression_output/)
# --------------------------------
#   spearman_<rcp>.parquet
#       Spearman rho and p-value for each delta metric x attribute pair.
#       Columns: attribute, delta_metric, rho, pvalue, significant (p<0.05)
#
#   ols_<rcp>.parquet
#       OLS regression summary for each delta metric.
#       Columns: delta_metric, attribute, coef, pvalue, r2_adj
#
#   rf_importance_<rcp>.parquet
#       Random forest feature importance for each delta metric.
#       Columns: delta_metric, attribute, importance
#
#   combined_attributes.parquet
#       Merged CAMELS-GB v2 attribute table (617 good catchments).
#
# Usage
# -----
#   python catchment_attribute_regression.py
#   python catchment_attribute_regression.py --rcp rcp85   # single RCP
#
# Attribute selection rationale
# -----------------------------
#   baseflow_index    : primary storage/recession control (BFI)
#   aridity           : PET/P climate dryness index
#   p_mean            : mean annual precipitation (mm/day)
#   p_seasonality     : precipitation seasonality (-1 winter, +1 summer)
#   high_prec_freq    : frequency of high precipitation days
#   area              : catchment scale
#   dpsbar            : mean drainage path slope (flashiness proxy)
#   frac_high_perc    : fraction highly permeable geology
#   frac_low_perc     : fraction low permeability geology
#   no_gw_perc        : fraction with negligible groundwater
#   runoff_ratio      : long-term Q/P ratio
#   slope_fdc         : slope of flow duration curve (flow variability)
#   low_q_freq        : frequency of low flow days (drought proneness)
#   high_q_freq       : frequency of high flow days (flood proneness)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGGREGATED_DIR = Path("./aggregated_output")
ATTRS_DIR      = Path(".")        # directory containing the four attribute CSVs
OUTPUT_DIR     = Path("./regression_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]

# Attribute CSV filenames
ATTR_FILES = {
    "hydrologic":   "camels_gb_v2_hydrologic_attributes.csv",
    "climatic":     "camels_gb_v2_climatic_attributes.csv",
    "hydrogeology": "camels_gb_v2_hydrogeology_attributes.csv",
    "topographic":  "camels_gb_v2_topographic_attributes.csv",
}

# Attributes to include in regression analysis
# Selected for physical relevance to flood-to-drought transitions
SELECTED_ATTRIBUTES = [
    # Hydrological
    "baseflow_index",    # storage / slow flow dominance
    "runoff_ratio",      # long-term water balance
    "slope_fdc",         # flow variability
    "high_q_freq",       # flood proneness
    "low_q_freq",        # drought proneness
    # Climatic
    "aridity",           # PET/P dryness
    "p_mean",            # mean precipitation
    "p_seasonality",     # seasonal precipitation regime
    "high_prec_freq",    # storm frequency
    "frac_snow",         # snow fraction -- included for both FTD and DTF regressions
    # Hydrogeology
    "frac_high_perc",    # highly permeable geology fraction
    "frac_low_perc",     # low permeability geology fraction
    "no_gw_perc",        # negligible groundwater fraction
    # Topographic
    "area",              # catchment scale (log-transformed below)
    "dpsbar",            # mean drainage path slope
]

# Delta metrics to use as dependent variables
DELTA_METRICS = [
    "delta_gap",
    "delta_freq",
    "delta_sm_drought",
    "delta_smdr",
    "delta_whiplash",
    "delta_flood_peak",
    "delta_drought_sev",
]

# Log-transform these attributes before regression (right-skewed)
LOG_TRANSFORM_ATTRS = {"area", "high_q_freq", "low_q_freq", "high_prec_freq"}

# Significance threshold
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Load and merge CAMELS-GB v2 attributes
# ---------------------------------------------------------------------------

def load_attributes(attrs_dir: Path) -> pd.DataFrame:
    """
    Load and merge all four CAMELS-GB v2 attribute files.
    Returns DataFrame indexed by gauge_id (integer).
    Retains only SELECTED_ATTRIBUTES plus gauge_id.
    """
    frames = []
    for group, fname in ATTR_FILES.items():
        fpath = attrs_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Attribute file not found: {fpath}")
        df = pd.read_csv(fpath, index_col="gauge_id")
        frames.append(df)
        print(f"  Loaded {group}: {df.shape[1]} attributes, {len(df)} catchments",
              flush=True)

    attrs = pd.concat(frames, axis=1)

    # Keep only selected attributes
    available = [a for a in SELECTED_ATTRIBUTES if a in attrs.columns]
    missing   = [a for a in SELECTED_ATTRIBUTES if a not in attrs.columns]
    if missing:
        print(f"  WARNING: attributes not found and skipped: {missing}", flush=True)

    attrs = attrs[available].copy()

    # Log-transform skewed attributes
    for col in LOG_TRANSFORM_ATTRS:
        if col in attrs.columns:
            attrs[f"log_{col}"] = np.log1p(attrs[col])
            attrs = attrs.drop(columns=[col])
            print(f"  Log-transformed: {col} -> log_{col}", flush=True)

    print(f"  Combined attribute table: {attrs.shape}", flush=True)
    return attrs


# ---------------------------------------------------------------------------
# Spearman correlation analysis
# ---------------------------------------------------------------------------

def spearman_analysis(
    metrics_df: pd.DataFrame,
    attrs_df:   pd.DataFrame,
    rcp:        str,
) -> pd.DataFrame:
    """
    Compute Spearman rho and p-value for each delta metric x attribute pair.
    Only uses catchments present in both DataFrames.
    """
    # Align on common catchments
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

            # Drop NaN pairs
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 30:
                continue

            rho, pval = stats.spearmanr(x[valid], y[valid])
            rows.append({
                "rcp":          rcp,
                "delta_metric": delta,
                "attribute":    attr,
                "rho":          rho,
                "pvalue":       pval,
                "n":            int(valid.sum()),
                "significant":  pval < ALPHA,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OLS multiple linear regression
# ---------------------------------------------------------------------------

def ols_analysis(
    metrics_df: pd.DataFrame,
    attrs_df:   pd.DataFrame,
    rcp:        str,
) -> pd.DataFrame:
    """
    For each delta metric, fit OLS with all selected attributes as predictors.
    Standardise predictors (zero mean, unit variance) for comparable
    coefficients. Returns coefficient, p-value, and adjusted R² per metric.
    """
    common = metrics_df.index.intersection(attrs_df.index)
    m = metrics_df.loc[common]
    a = attrs_df.loc[common]

    scaler = StandardScaler()
    rows = []

    for delta in DELTA_METRICS:
        if delta not in m.columns:
            continue

        y = m[delta].values

        # Drop catchments with any NaN in attributes or metric
        attr_cols = a.columns.tolist()
        X_raw = a[attr_cols].values
        valid  = ~(np.isnan(y) | np.any(np.isnan(X_raw), axis=1))

        if valid.sum() < 50:
            continue

        X = scaler.fit_transform(X_raw[valid])
        y_v = y[valid]

        # Fit OLS via scipy for p-values
        from scipy.stats import t as t_dist
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])   # add intercept
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        y_pred    = X_aug @ beta
        ss_res    = np.sum((y_v - y_pred) ** 2)
        ss_tot    = np.sum((y_v - y_v.mean()) ** 2)
        r2        = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2_adj    = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Standard errors and t-statistics
        sigma2    = ss_res / (n - p - 1) if n > p + 1 else np.nan
        try:
            cov   = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
            se    = np.sqrt(np.diag(cov))
            t_stat = beta / se
            pvals  = 2 * t_dist.sf(np.abs(t_stat), df=n - p - 1)
        except np.linalg.LinAlgError:
            pvals = np.full(len(beta), np.nan)

        # Skip intercept (index 0)
        for i, attr in enumerate(attr_cols):
            rows.append({
                "rcp":          rcp,
                "delta_metric": delta,
                "attribute":    attr,
                "coef":         float(beta[i + 1]),
                "pvalue":       float(pvals[i + 1]),
                "r2_adj":       float(r2_adj),
                "n":            int(valid.sum()),
                "significant":  float(pvals[i + 1]) < ALPHA,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Random forest variable importance
# ---------------------------------------------------------------------------

def rf_importance_analysis(
    metrics_df: pd.DataFrame,
    attrs_df:   pd.DataFrame,
    rcp:        str,
    n_trees:    int = 500,
) -> pd.DataFrame:
    """
    Fit a random forest for each delta metric and return
    mean decrease in impurity (MDI) feature importance.

    RF captures non-linear relationships and interactions
    that OLS may miss. Use as a cross-check on Spearman/OLS.
    """
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

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            oob_score=True,    # honest out-of-bag R² -- report this, not train R²
        )
        rf.fit(X_raw[valid], y[valid])

        train_r2 = rf.score(X_raw[valid], y[valid])
        oob_r2   = rf.oob_score_   # unbiased estimate of generalisation R²

        for attr, imp in zip(attr_cols, rf.feature_importances_):
            rows.append({
                "rcp":          rcp,
                "delta_metric": delta,
                "attribute":    attr,
                "importance":   float(imp),
                "oob_r2":       float(oob_r2),
            })

        print(
            f"    RF {delta}: n={valid.sum()}, "
            f"train R²={train_r2:.3f}  OOB R²={oob_r2:.3f}",
            flush=True,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_rcp(rcp: str, attrs_df: pd.DataFrame) -> None:
    print(f"\n{'='*55}", flush=True)
    print(f"  RCP: {rcp}", flush=True)
    print(f"{'='*55}", flush=True)

    metrics_path = AGGREGATED_DIR / f"catchment_metrics_{rcp}.parquet"
    if not metrics_path.exists():
        print(f"  ERROR: {metrics_path} not found", flush=True)
        return

    metrics_df = pd.read_parquet(metrics_path)

    # Align index: metrics uses gauge_id strings, attrs uses integers
    metrics_df.index = metrics_df.index.astype(int)

    # Keep only delta columns + align
    delta_cols = [c for c in metrics_df.columns if c.startswith("delta_")]
    metrics_df = metrics_df[delta_cols]

    common = metrics_df.index.intersection(attrs_df.index)
    print(f"  Common catchments: {len(common)}", flush=True)

    # Spearman
    print("  Running Spearman correlations ...", flush=True)
    spearman_df = spearman_analysis(metrics_df, attrs_df, rcp)
    out = OUTPUT_DIR / f"spearman_{rcp}.parquet"
    spearman_df.to_parquet(out, index=False)
    print(f"  Written: {out.name}", flush=True)

    # Print top correlations for delta_gap and delta_sm_drought
    for delta in ["delta_gap", "delta_sm_drought"]:
        sub = spearman_df[spearman_df["delta_metric"] == delta].copy()
        sub = sub.sort_values("rho", key=abs, ascending=False).head(5)
        print(f"\n  Top Spearman correlates for {delta}:")
        for _, row in sub.iterrows():
            sig = "*" if row["significant"] else ""
            print(
                f"    {row['attribute']:25s} rho={row['rho']:+.3f}  "
                f"p={row['pvalue']:.3f}{sig}"
            )

    # OLS
    print("\n  Running OLS regression ...", flush=True)
    ols_df = ols_analysis(metrics_df, attrs_df, rcp)
    out = OUTPUT_DIR / f"ols_{rcp}.parquet"
    ols_df.to_parquet(out, index=False)
    print(f"  Written: {out.name}", flush=True)

    # Print adjusted R² per metric
    print("\n  OLS adjusted R² per delta metric:")
    for delta in DELTA_METRICS:
        sub = ols_df[ols_df["delta_metric"] == delta]
        if not sub.empty:
            r2 = sub["r2_adj"].iloc[0]
            print(f"    {delta:25s} adj-R²={r2:.3f}")

    # Random forest
    print("\n  Running random forest importance ...", flush=True)
    rf_df = rf_importance_analysis(metrics_df, attrs_df, rcp)
    out = OUTPUT_DIR / f"rf_importance_{rcp}.parquet"
    rf_df.to_parquet(out, index=False)
    print(f"  Written: {out.name}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regress transition change metrics against CAMELS-GB v2 attributes."
    )
    parser.add_argument(
        "--rcp",
        choices=RCPS,
        default=None,
        help="Single RCP to process. Default: all four.",
    )
    parser.add_argument(
        "--attrs-dir",
        type=Path,
        default=ATTRS_DIR,
        help="Directory containing CAMELS-GB v2 attribute CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CAMELS-GB v2 attributes ...", flush=True)
    attrs_df = load_attributes(args.attrs_dir)

    # Save combined attribute table once
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
