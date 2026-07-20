# aggregate_ensemble_d2f.py
# ---------------------------
# d2f equivalent of aggregate_ensemble.py. Aggregates drought-to-flood
# transition metrics across ensemble members and computes baseline vs
# future change signals per catchment per RCP -- same structure as the
# f2d aggregation, so the two are directly comparable.
#
# Inputs
# ------
#   drought_to_flood_exploratory/d2f_<rcp>_<ens>_<period>_w090.parquet
#       (from explore_drought_to_flood.py --stage pairing, 90-day window
#        to match FTD's MAX_TRANSITION_GAP and existing literature
#        precedent; change DEFAULT_WINDOW below to use one of the other
#        widths supported by explore_drought_to_flood.py's --sensitivity
#        mode instead, if a different window is preferred)
#   calibrated_parameters.csv (via catchment_filter.py, same filter as f2d)
#
# Outputs (in d2f_aggregated_output/)
# --------------------------------
#   d2f_catchment_metrics_<rcp>.parquet
#       One row per catchment. Columns:
#
#       Transition frequency (events per year):
#           freq_baseline_med, freq_baseline_iqr
#           freq_future_med,   freq_future_iqr
#           delta_freq
#
#       Recovery gap (days from drought end to flood onset):
#           gap_baseline_med, gap_baseline_iqr
#           gap_future_med,   gap_future_iqr
#           delta_gap
#
#       SM recovery rate (mm/day during transition):
#           recovery_rate_baseline_med, recovery_rate_baseline_iqr
#           recovery_rate_future_med,   recovery_rate_future_iqr
#           delta_recovery_rate
#
#       SM at drought end / flood onset (mm):
#           sm_drought_end_baseline_med, sm_drought_end_future_med, delta_sm_drought_end
#           sm_flood_onset_baseline_med, sm_flood_onset_future_med, delta_sm_flood_onset
#
#       Flood characteristics:
#           flood_peak_baseline_med, flood_peak_future_med, delta_flood_peak
#
#       Drought characteristics:
#           drought_dur_baseline_med, drought_dur_future_med, delta_drought_dur
#
#   d2f_catchment_metrics_all_rcps.parquet
#       All four RCPs stacked with an rcp column.
#
# Usage
# -----
#   python aggregate_ensemble_d2f.py
#   python aggregate_ensemble_d2f.py --window 60   # use a different sensitivity window

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from catchment_filter import get_valid_catchments

TRANSITIONS_DIR   = Path("./drought_to_flood_exploratory")
OUTPUT_DIR        = Path("./d2f_aggregated_output")

KGE_RESULTS_PATH  = "./calibrated_parameters.csv"
KGE_THRESHOLD     = 0.5

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]

BASELINE_YEARS = 29.0   # WY1982-WY2010, same as f2d
FUTURE_YEARS   = 30.0   # WY2051-WY2080, same as f2d

DEFAULT_WINDOW = 90     # max_gap window, matches f2d's MAX_TRANSITION_GAP


# ---------------------------------------------------------------------------
# Load good catchment list (same filter as f2d, for direct comparability)
# ---------------------------------------------------------------------------

def load_good_catchments() -> list:
    valid_int = get_valid_catchments(
        kge_path=KGE_RESULTS_PATH,
        kge_threshold=KGE_THRESHOLD,
        verbose=True,
    )
    valid_str = [str(i) for i in valid_int]
    print(f"  {len(valid_str)} catchments retained as strings for filtering", flush=True)
    return valid_str


# ---------------------------------------------------------------------------
# Per-catchment summary from one d2f transition parquet
# ---------------------------------------------------------------------------

def summarise_transitions(df: pd.DataFrame, period_years: float, good_catchments) -> pd.DataFrame:
    """
    For each catchment in df, compute median of each metric across all
    d2f transitions in the period. Returns one row per catchment.
    """
    if good_catchments is not None:
        df = df[df["gauge_id"].isin(good_catchments)]

    if df.empty:
        return pd.DataFrame()

    metrics = {
        "gap":            "recovery_gap_days",
        "recovery_rate":  "sm_recovery_rate",
        "sm_drought_end": "sm_at_drought_end",
        "sm_flood_onset": "sm_at_flood_onset",
        "flood_peak":     "flood_peak_flow",
        "drought_dur":    "drought_duration",
    }

    rows = []
    for cid, grp in df.groupby("gauge_id"):
        row = {"gauge_id": cid}
        row["freq"] = len(grp) / period_years
        for name, col in metrics.items():
            if col in grp.columns:
                row[name] = float(np.nanmedian(grp[col].values))
            else:
                row[name] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("gauge_id")


# ---------------------------------------------------------------------------
# Ensemble aggregation: median + IQR across 4 members (identical logic to f2d)
# ---------------------------------------------------------------------------

def ensemble_stats(frames: list, suffix: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    common_idx = frames[0].index
    for f in frames[1:]:
        common_idx = common_idx.intersection(f.index)

    aligned = [f.loc[common_idx] for f in frames]
    stacked = np.stack([f.values for f in aligned], axis=2)

    col_names = aligned[0].columns.tolist()
    result = pd.DataFrame(index=common_idx)

    for i, col in enumerate(col_names):
        vals = stacked[:, i, :]
        result[f"{col}_{suffix}_med"] = np.nanmedian(vals, axis=1)
        q75 = np.nanpercentile(vals, 75, axis=1)
        q25 = np.nanpercentile(vals, 25, axis=1)
        result[f"{col}_{suffix}_iqr"] = q75 - q25

    return result


# ---------------------------------------------------------------------------
# Main aggregation for one RCP
# ---------------------------------------------------------------------------

def aggregate_rcp(rcp: str, window: int, good_catchments) -> pd.DataFrame:
    print(f"\n  RCP: {rcp}", flush=True)

    baseline_frames = []
    future_frames = []

    for ens in ENSEMBLES:
        b_path = TRANSITIONS_DIR / f"d2f_{rcp}_{ens}_baseline_w{window:03d}.parquet"
        f_path = TRANSITIONS_DIR / f"d2f_{rcp}_{ens}_future_w{window:03d}.parquet"

        if not b_path.exists():
            print(f"    WARNING: missing {b_path.name}", flush=True)
            continue
        if not f_path.exists():
            print(f"    WARNING: missing {f_path.name}", flush=True)
            continue

        b_df = pd.read_parquet(b_path)
        f_df = pd.read_parquet(f_path)

        baseline_frames.append(summarise_transitions(b_df, BASELINE_YEARS, good_catchments))
        future_frames.append(summarise_transitions(f_df, FUTURE_YEARS, good_catchments))
        print(f"    Loaded {ens}: baseline={len(b_df)}, future={len(f_df)}", flush=True)

    print("    Computing ensemble stats ...", flush=True)
    baseline_stats = ensemble_stats(baseline_frames, "baseline")
    future_stats   = ensemble_stats(future_frames,   "future")

    common = baseline_stats.index.intersection(future_stats.index)
    baseline_stats = baseline_stats.loc[common]
    future_stats   = future_stats.loc[common]

    metrics_df = pd.concat([baseline_stats, future_stats], axis=1)

    metric_names = ["freq", "gap", "recovery_rate", "sm_drought_end",
                     "sm_flood_onset", "flood_peak", "drought_dur"]
    for name in metric_names:
        b_col = f"{name}_baseline_med"
        f_col = f"{name}_future_med"
        if b_col in metrics_df.columns and f_col in metrics_df.columns:
            metrics_df[f"delta_{name}"] = metrics_df[f_col] - metrics_df[b_col]

    metrics_df.index.name = "gauge_id"
    print(f"    Result: {len(metrics_df)} catchments, {len(metrics_df.columns)} columns", flush=True)
    return metrics_df


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble aggregation for d2f transition metrics.")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Max-gap window to aggregate (default: {DEFAULT_WINDOW}, matching f2d).")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    good_catchments = load_good_catchments()

    all_rcp_frames = []

    for rcp in RCPS:
        out_path = OUTPUT_DIR / f"d2f_catchment_metrics_{rcp}.parquet"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists", flush=True)
            df = pd.read_parquet(out_path)
        else:
            df = aggregate_rcp(rcp, args.window, good_catchments)
            df.to_parquet(out_path)
            print(f"  Written: {out_path.name}", flush=True)

        df["rcp"] = rcp
        all_rcp_frames.append(df.reset_index())

    combined_path = OUTPUT_DIR / "d2f_catchment_metrics_all_rcps.parquet"
    if combined_path.exists():
        print(f"  [skip] {combined_path.name} already exists", flush=True)
    else:
        combined = pd.concat(all_rcp_frames, ignore_index=True)
        combined.to_parquet(combined_path, index=False)
        print(f"  Written: {combined_path.name}  shape={combined.shape}", flush=True)

    print("\n--- Summary (RCP8.5) ---")
    rcp85 = pd.read_parquet(OUTPUT_DIR / "d2f_catchment_metrics_rcp85.parquet")
    for col in ["delta_freq", "delta_gap", "delta_recovery_rate"]:
        if col in rcp85.columns:
            print(
                f"  {col}: median={rcp85[col].median():.3f}  "
                f"IQR=[{rcp85[col].quantile(0.25):.3f}, "
                f"{rcp85[col].quantile(0.75):.3f}]"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
