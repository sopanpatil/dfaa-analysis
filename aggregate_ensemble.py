# aggregate_ensemble.py
# ---------------------
# Aggregate flood-to-drought transition metrics across ensemble members
# and compute baseline vs future change signals per catchment per RCP.
#
# Also extracts whiplash frequency from SSI parquets and merges into
# the same output table.
#
# Inputs
# ------
#   transitions_output/transitions_<rcp>_<ens>_<period>.parquet
#   ssi_output/ssi_baseline_<rcp>_<ens>.parquet
#   ssi_output/ssi_future_<rcp>_<ens>.parquet
#   thresholds/thresholds_ensemble_median.parquet  (for catchment list)
#
# Outputs (in aggregated_output/)
# --------------------------------
#   catchment_metrics_<rcp>.parquet
#       One row per catchment. Columns:
#
#       Transition frequency (events per year):
#           freq_baseline_med, freq_baseline_iqr
#           freq_future_med,   freq_future_iqr
#           delta_freq
#
#       Gap days (days from flood end to drought onset):
#           gap_baseline_med, gap_baseline_iqr
#           gap_future_med,   gap_future_iqr
#           delta_gap
#
#       SM depletion rate (mm/day during transition):
#           smdr_baseline_med, smdr_baseline_iqr
#           smdr_future_med,   smdr_future_iqr
#           delta_smdr
#
#       SM at flood peak (mm):
#           sm_peak_baseline_med, sm_peak_future_med, delta_sm_peak
#
#       SM at drought onset (mm):
#           sm_drought_baseline_med, sm_drought_future_med, delta_sm_drought
#
#       Flood characteristics:
#           flood_peak_baseline_med, flood_peak_future_med, delta_flood_peak
#           flood_dur_baseline_med,  flood_dur_future_med,  delta_flood_dur
#
#       Drought characteristics:
#           drought_sev_baseline_med, drought_sev_future_med, delta_drought_sev
#           drought_dur_baseline_med, drought_dur_future_med, delta_drought_dur
#
#       Whiplash frequency (wet-to-dry SSI crossings per 30-year period):
#           whiplash_baseline_med, whiplash_baseline_iqr
#           whiplash_future_med,   whiplash_future_iqr
#           delta_whiplash
#
#   catchment_metrics_all_rcps.parquet
#       All four RCPs stacked with an rcp column, for cross-RCP plots.
#
# Usage
# -----
#   python aggregate_ensemble.py
#
# Notes
# -----
# Ensemble aggregation order:
#   1. For each catchment × period × ensemble member:
#      compute per-catchment summary (median across all transitions)
#   2. Across the 4 ensemble members: median (central) + IQR (uncertainty)
#   3. Delta = future_med - baseline_med
#
# Whiplash definition (He et al. 2026 compatible):
#   A wet-to-dry whiplash event occurs when SSI-1 transitions from
#   >= 1.0 (wet) in month t to <= -1.0 (dry) in month t+1.
#   Count per catchment per 30-year period.
#
# Catchment filter:
#   Uses catchment_filter.py (get_valid_catchments) to restrict analysis
#   to catchments with validation KGE >= KGE_THRESHOLD (default 0.5).
#   Requires calibrated_parameters.csv in the working directory.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from catchment_filter import get_valid_catchments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSITIONS_DIR  = Path("./transitions_output")
SSI_DIR          = Path("./ssi_output")
OUTPUT_DIR       = Path("./aggregated_output")

# Path to calibrated_parameters.csv -- used by catchment_filter.py
KGE_RESULTS_PATH = "./calibrated_parameters.csv"
KGE_THRESHOLD    = 0.5

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]

# Baseline and future period lengths in years (for per-year frequency)
BASELINE_YEARS = 29.0   # WY1982-WY2010
FUTURE_YEARS   = 30.0   # WY2051-WY2080

# SSI thresholds for whiplash detection
SSI_WET_THRESHOLD = 1.0
SSI_DRY_THRESHOLD = -1.0


# ---------------------------------------------------------------------------
# Load good catchment list via catchment_filter.py
# ---------------------------------------------------------------------------

def load_good_catchments() -> list[str]:
    """
    Load valid catchment IDs using catchment_filter.get_valid_catchments().
    Returns a list of catchment ID strings (e.g. ["15006", "39001", ...])
    to match the gauge_id column format in the transition parquets.

    catchment_filter returns integer IDs; we convert to str here so that
    filtering against gauge_id columns works without type mismatches.
    """
    valid_int = get_valid_catchments(
        kge_path=KGE_RESULTS_PATH,
        kge_threshold=KGE_THRESHOLD,
        verbose=True,
    )
    # Convert to strings to match gauge_id format in transition parquets
    valid_str = [str(i) for i in valid_int]
    print(f"  {len(valid_str)} catchments retained as strings for filtering", flush=True)
    return valid_str


# ---------------------------------------------------------------------------
# Per-catchment summary from one transition parquet
# ---------------------------------------------------------------------------

def summarise_transitions(
    df: pd.DataFrame,
    period_years: float,
    good_catchments: list[str] | None,
) -> pd.DataFrame:
    """
    For each catchment in df, compute median of each metric across all
    transitions in the period. Returns one row per catchment.

    Metrics computed:
        freq         : transitions per year
        gap          : median gap_days
        smdr         : median sm_depletion_rate
        sm_peak      : median sm_at_flood_peak
        sm_drought   : median sm_at_drought_start
        flood_peak   : median flood_peak_flow
        flood_dur    : median flood_duration
        drought_sev  : median drought_severity
        drought_dur  : median drought_duration
    """
    if good_catchments is not None:
        df = df[df["gauge_id"].isin(good_catchments)]

    if df.empty:
        return pd.DataFrame()

    metrics = {
        "gap":        "gap_days",
        "smdr":       "sm_depletion_rate",
        "sm_peak":    "sm_at_flood_peak",
        "sm_drought": "sm_at_drought_start",
        "flood_peak": "flood_peak_flow",
        "flood_dur":  "flood_duration",
        "drought_sev":"drought_severity",
        "drought_dur":"drought_duration",
    }

    rows = []
    for cid, grp in df.groupby("gauge_id"):
        row = {"gauge_id": cid}
        # Frequency: events per year
        row["freq"] = len(grp) / period_years
        # Median of each metric across all transitions
        for name, col in metrics.items():
            if col in grp.columns:
                row[name] = float(np.nanmedian(grp[col].values))
            else:
                row[name] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("gauge_id")


# ---------------------------------------------------------------------------
# Ensemble aggregation: median + IQR across 4 members
# ---------------------------------------------------------------------------

def ensemble_stats(
    frames: list[pd.DataFrame],
    suffix: str,
) -> pd.DataFrame:
    """
    Given a list of per-catchment summary DataFrames (one per ensemble
    member), compute median and IQR across members for each metric.

    suffix: "baseline" or "future" -- appended to output column names.
    """
    if not frames:
        return pd.DataFrame()

    # Align on common catchments
    common_idx = frames[0].index
    for f in frames[1:]:
        common_idx = common_idx.intersection(f.index)

    aligned = [f.loc[common_idx] for f in frames]
    stacked = np.stack([f.values for f in aligned], axis=2)
    # stacked shape: (n_catchments, n_metrics, n_members)

    col_names = aligned[0].columns.tolist()
    result = pd.DataFrame(index=common_idx)

    for i, col in enumerate(col_names):
        vals = stacked[:, i, :]   # (n_catchments, n_members)
        result[f"{col}_{suffix}_med"] = np.nanmedian(vals, axis=1)
        q75 = np.nanpercentile(vals, 75, axis=1)
        q25 = np.nanpercentile(vals, 25, axis=1)
        result[f"{col}_{suffix}_iqr"] = q75 - q25

    return result


# ---------------------------------------------------------------------------
# Whiplash frequency from SSI parquets
# ---------------------------------------------------------------------------

def count_whiplash(ssi_df: pd.DataFrame, good_catchments: list[str] | None,
                    direction: str = "wet_to_dry") -> pd.Series:
    """
    Count whiplash events per catchment from an SSI parquet.

    direction="wet_to_dry": SSI[t] >= SSI_WET_THRESHOLD and SSI[t+1] <= SSI_DRY_THRESHOLD
    direction="dry_to_wet": SSI[t] <= SSI_DRY_THRESHOLD and SSI[t+1] >= SSI_WET_THRESHOLD

    Returns Series indexed by gauge_id with whiplash event count.
    """
    if direction not in ("wet_to_dry", "dry_to_wet"):
        raise ValueError(f"Unknown direction: {direction}")

    catchment_cols = [c for c in ssi_df.columns if c != "year_month"]
    if good_catchments is not None:
        catchment_cols = [c for c in catchment_cols if c in good_catchments]

    counts = {}
    for cid in catchment_cols:
        ssi = ssi_df[cid].to_numpy(dtype=np.float64)
        if direction == "wet_to_dry":
            start = ssi[:-1] >= SSI_WET_THRESHOLD
            end   = ssi[1:]  <= SSI_DRY_THRESHOLD
        else:
            start = ssi[:-1] <= SSI_DRY_THRESHOLD
            end   = ssi[1:]  >= SSI_WET_THRESHOLD
        counts[cid] = int(np.sum(start & end))

    return pd.Series(counts, name=f"whiplash_{direction}")


def aggregate_whiplash(
    rcp: str,
    period: str,
    good_catchments: list[str] | None,
    direction: str = "wet_to_dry",
) -> pd.DataFrame:
    """
    Load SSI parquets for all ensemble members, count whiplash per member
    for the given direction, then compute median and IQR across members.
    """
    member_counts = []
    for ens in ENSEMBLES:
        fpath = SSI_DIR / f"ssi_{period}_{rcp}_{ens}.parquet"
        if not fpath.exists():
            print(f"  WARNING: missing {fpath.name}", flush=True)
            continue
        ssi_df = pd.read_parquet(fpath)
        counts = count_whiplash(ssi_df, good_catchments, direction=direction)
        member_counts.append(counts)

    if not member_counts:
        return pd.DataFrame()

    stacked = pd.concat(member_counts, axis=1)   # (n_catchments, n_members)
    result = pd.DataFrame(index=stacked.index)
    result[f"whiplash_{direction}_{period}_med"] = stacked.median(axis=1)
    result[f"whiplash_{direction}_{period}_iqr"] = stacked.quantile(0.75, axis=1) - \
                                        stacked.quantile(0.25, axis=1)
    return result


# ---------------------------------------------------------------------------
# Main aggregation for one RCP
# ---------------------------------------------------------------------------

def aggregate_rcp(rcp: str, good_catchments: list[str] | None) -> pd.DataFrame:
    print(f"\n  RCP: {rcp}", flush=True)

    # ------------------------------------------------------------------
    # Transition metrics
    # ------------------------------------------------------------------
    baseline_frames = []
    future_frames   = []

    for ens in ENSEMBLES:
        b_path = TRANSITIONS_DIR / f"transitions_{rcp}_{ens}_baseline.parquet"
        f_path = TRANSITIONS_DIR / f"transitions_{rcp}_{ens}_future.parquet"

        if not b_path.exists():
            print(f"    WARNING: missing {b_path.name}", flush=True)
            continue
        if not f_path.exists():
            print(f"    WARNING: missing {f_path.name}", flush=True)
            continue

        b_df = pd.read_parquet(b_path)
        f_df = pd.read_parquet(f_path)

        baseline_frames.append(
            summarise_transitions(b_df, BASELINE_YEARS, good_catchments)
        )
        future_frames.append(
            summarise_transitions(f_df, FUTURE_YEARS, good_catchments)
        )
        print(f"    Loaded {ens}: baseline={len(b_df)}, future={len(f_df)}", flush=True)

    print("    Computing ensemble stats ...", flush=True)
    baseline_stats = ensemble_stats(baseline_frames, "baseline")
    future_stats   = ensemble_stats(future_frames,   "future")

    # Align on common catchments
    common = baseline_stats.index.intersection(future_stats.index)
    baseline_stats = baseline_stats.loc[common]
    future_stats   = future_stats.loc[common]

    # Merge baseline and future stats
    metrics_df = pd.concat([baseline_stats, future_stats], axis=1)

    # Compute delta columns (future median - baseline median)
    metric_names = [
        "freq", "gap", "smdr", "sm_peak", "sm_drought",
        "flood_peak", "flood_dur", "drought_sev", "drought_dur",
    ]
    for name in metric_names:
        b_col = f"{name}_baseline_med"
        f_col = f"{name}_future_med"
        if b_col in metrics_df.columns and f_col in metrics_df.columns:
            metrics_df[f"delta_{name}"] = (
                metrics_df[f_col] - metrics_df[b_col]
            )

    # ------------------------------------------------------------------
    # Whiplash frequency from SSI (both directions)
    # ------------------------------------------------------------------
    print("    Computing whiplash frequency ...", flush=True)
    for direction in ["wet_to_dry", "dry_to_wet"]:
        whiplash_b = aggregate_whiplash(rcp, "baseline", good_catchments, direction=direction)
        whiplash_f = aggregate_whiplash(rcp, "future",   good_catchments, direction=direction)

        if not whiplash_b.empty and not whiplash_f.empty:
            whiplash_common = whiplash_b.index.intersection(whiplash_f.index)
            whiplash_df = pd.concat(
                [whiplash_b.loc[whiplash_common],
                 whiplash_f.loc[whiplash_common]],
                axis=1,
            )
            whiplash_df[f"delta_whiplash_{direction}"] = (
                whiplash_df[f"whiplash_{direction}_future_med"] -
                whiplash_df[f"whiplash_{direction}_baseline_med"]
            )
            metrics_df = metrics_df.join(whiplash_df, how="left")

    # Backward-compatible alias: delta_whiplash == wet-to-dry (the original,
    # already-published metric name), so existing downstream scripts and
    # figures referencing delta_whiplash are unaffected.
    if "delta_whiplash_wet_to_dry" in metrics_df.columns:
        metrics_df["delta_whiplash"] = metrics_df["delta_whiplash_wet_to_dry"]

    metrics_df.index.name = "gauge_id"
    print(
        f"    Result: {len(metrics_df)} catchments, "
        f"{len(metrics_df.columns)} columns",
        flush=True,
    )
    return metrics_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    good_catchments = load_good_catchments()  # always returns list[str]

    all_rcp_frames = []

    for rcp in RCPS:
        out_path = OUTPUT_DIR / f"catchment_metrics_{rcp}.parquet"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists", flush=True)
            df = pd.read_parquet(out_path)
        else:
            df = aggregate_rcp(rcp, good_catchments)
            df.to_parquet(out_path)
            print(f"  Written: {out_path.name}", flush=True)

        df["rcp"] = rcp
        all_rcp_frames.append(df.reset_index())

    # Combined file with all RCPs
    combined_path = OUTPUT_DIR / "catchment_metrics_all_rcps.parquet"
    if combined_path.exists():
        print(f"  [skip] {combined_path.name} already exists", flush=True)
    else:
        combined = pd.concat(all_rcp_frames, ignore_index=True)
        combined.to_parquet(combined_path, index=False)
        print(f"  Written: {combined_path.name}  shape={combined.shape}", flush=True)

    # Summary printout
    print("\n--- Summary (RCP8.5) ---")
    rcp85 = pd.read_parquet(OUTPUT_DIR / "catchment_metrics_rcp85.parquet")
    for col in ["delta_freq", "delta_gap", "delta_smdr", "delta_whiplash"]:
        if col in rcp85.columns:
            print(
                f"  {col}: median={rcp85[col].median():.3f}  "
                f"IQR=[{rcp85[col].quantile(0.25):.3f}, "
                f"{rcp85[col].quantile(0.75):.3f}]"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
