# analyze_dynamic_snow.py
# --------------------------
# Tests the independent, mechanistic complement to the static frac_snow
# attribute (Methods 2.7): a genuinely DYNAMIC snow-change metric, built
# from HBV's simulated SP state (available for both baseline and future
# periods, under every RCP), and its correlation with transition-gap
# change. Unlike the static frac_snow attribute (fixed historical
# climatology, the same value at every RCP), this dynamic metric can
# respond to RCP-specific warming -- providing a mechanistically
# stronger test of the snow relationship reported in Results 3.4.
#
# Steps:
#   1. For each of 16 RCP/ensemble combinations, compute baseline and
#      future mean_sp and max_sp per catchment (median across ensemble).
#   2. delta_mean_sp = future - baseline (snowpack decline, typically
#      negative under warming), same for delta_max_sp.
#   3. Correlate delta_snow metrics against delta_gap in BOTH the FTD and
#      DTF catchment_metrics tables, across all four RCPs.
#
# Requires:
#   chess_scape_output/<rcp>_<ens>_hbv_sp.csv
#   aggregated_output/catchment_metrics_<rcp>.parquet (FTD)
#   d2f_aggregated_output/d2f_catchment_metrics_<rcp>.parquet (DTF)
#
# Usage
# -----
#   python analyze_dynamic_snow.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
F2D_DIR = Path("./aggregated_output")
D2F_DIR = Path("./d2f_aggregated_output")
OUTPUT_DIR = Path("./spatial_output")

RCPS = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]

BASELINE_START, BASELINE_END = "1981-10-01", "2010-09-30"
FUTURE_START, FUTURE_END     = "2050-10-01", "2080-09-30"

ALPHA = 0.05


def extract_period_sp(rcp: str, ens: str, start: str, end: str) -> pd.DataFrame:
    path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_sp.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"date": str})
    date_only = df["date"].str.slice(0, 10)
    mask = (date_only >= start) & (date_only <= end)
    return df.loc[mask].drop(columns=["date"])


def summarise(period_sp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid in period_sp.columns:
        vals = period_sp[cid].to_numpy(dtype=np.float64)
        rows.append({"gauge_id": cid, "mean_sp": float(np.nanmean(vals)),
                     "max_sp": float(np.nanmax(vals))})
    return pd.DataFrame(rows).set_index("gauge_id")


def build_snow_change_table(rcp: str) -> pd.DataFrame:
    """Ensemble-median delta_mean_sp / delta_max_sp for one RCP."""
    baseline_frames, future_frames = [], []
    for ens in ENSEMBLES:
        b = extract_period_sp(rcp, ens, BASELINE_START, BASELINE_END)
        f = extract_period_sp(rcp, ens, FUTURE_START, FUTURE_END)
        if b is None or f is None:
            print(f"  WARNING: missing snow data for {rcp}/{ens}", flush=True)
            continue
        baseline_frames.append(summarise(b))
        future_frames.append(summarise(f))

    if not baseline_frames:
        return pd.DataFrame()

    common_idx = baseline_frames[0].index
    for s in baseline_frames[1:] + future_frames:
        common_idx = common_idx.intersection(s.index)

    b_stack = {m: np.stack([s.loc[common_idx, m].values for s in baseline_frames], axis=1)
               for m in ["mean_sp", "max_sp"]}
    f_stack = {m: np.stack([s.loc[common_idx, m].values for s in future_frames], axis=1)
               for m in ["mean_sp", "max_sp"]}

    result = pd.DataFrame(index=common_idx)
    for m in ["mean_sp", "max_sp"]:
        b_med = np.nanmedian(b_stack[m], axis=1)
        f_med = np.nanmedian(f_stack[m], axis=1)
        result[f"{m}_baseline"] = b_med
        result[f"{m}_future"] = f_med
        result[f"delta_{m}"] = f_med - b_med

    result.index.name = "gauge_id"
    result.index = result.index.astype(int)
    return result


def correlate_with_target(snow_change: pd.DataFrame, target_df: pd.DataFrame,
                            target_name: str, rcp: str) -> list:
    common = snow_change.index.intersection(target_df.index)
    rows = []
    for snow_metric in ["delta_mean_sp", "delta_max_sp"]:
        if "delta_gap" not in target_df.columns:
            continue
        x = snow_change.loc[common, snow_metric].values
        y = target_df.loc[common, "delta_gap"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 30:
            continue
        rho, pval = stats.spearmanr(x[valid], y[valid])
        rows.append({"target": target_name, "rcp": rcp, "snow_metric": snow_metric,
                     "rho": rho, "pvalue": pval, "n": int(valid.sum()), "significant": pval < ALPHA})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_snow_tables = []

    for rcp in RCPS:
        print(f"\n{rcp}: building snow change table...", flush=True)
        snow_change = build_snow_change_table(rcp)
        if snow_change.empty:
            continue
        print(f"  {len(snow_change)} catchments  "
              f"median delta_mean_sp={snow_change['delta_mean_sp'].median():+.2f} mm", flush=True)
        snow_change["rcp"] = rcp
        all_snow_tables.append(snow_change.reset_index())

        f2d_path = F2D_DIR / f"catchment_metrics_{rcp}.parquet"
        if f2d_path.exists():
            f2d = pd.read_parquet(f2d_path)
            f2d.index = f2d.index.astype(int)
            all_results.extend(correlate_with_target(snow_change, f2d, "f2d", rcp))

        d2f_path = D2F_DIR / f"d2f_catchment_metrics_{rcp}.parquet"
        if d2f_path.exists():
            d2f = pd.read_parquet(d2f_path)
            d2f.index = d2f.index.astype(int)
            all_results.extend(correlate_with_target(snow_change, d2f, "d2f", rcp))

    combined_snow = pd.concat(all_snow_tables, ignore_index=True)
    combined_snow.to_csv(OUTPUT_DIR / "dynamic_snow_change_all_rcps.csv", index=False)
    print(f"\nWritten: {OUTPUT_DIR / 'dynamic_snow_change_all_rcps.csv'}")

    print("\n" + "=" * 90)
    print("DYNAMIC SNOW CHANGE vs delta_gap -- f2d and d2f, all RCPs")
    print("=" * 90)
    result_df = pd.DataFrame(all_results)
    print(result_df.to_string(index=False))
    result_df.to_csv(OUTPUT_DIR / "dynamic_snow_correlation.csv", index=False)
    print(f"\nWritten: {OUTPUT_DIR / 'dynamic_snow_correlation.csv'}")


if __name__ == "__main__":
    main()
