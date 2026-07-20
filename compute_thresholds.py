# compute_thresholds.py
# ---------------------
# Compute fixed Q5 (flood) and Q80 (drought) thresholds for each catchment
# from the baseline period (WY1982-WY2010, water year starting 1 October).
#
# Thresholds are computed from HBV simulated discharge (not observed),
# consistent with the projection runs. Using simulated baseline flow ensures
# model bias does not affect the future/baseline comparison -- both periods
# use the same model with the same parameter set.
#
# Inputs
# ------
# Baseline HBV discharge CSVs (wide format):
#     <CHESS_SCAPE_ROOT>/<rcp>_<ens>_hbv_discharge.csv
#     Columns: date, <catchment_id>, <catchment_id>, ...
#     Date format: YYYY-MM-DD (360-day calendar, kept as opaque strings)
#
# Outputs
# -------
#     <OUTPUT_DIR>/thresholds_<rcp>_<ens>.parquet
#         gauge_id | q5 | q80
#     <OUTPUT_DIR>/thresholds_ensemble_median.parquet
#         gauge_id | q5_median | q80_median
#
# Usage
# -----
#   python compute_thresholds.py --rcp rcp26 --ensemble 01   # single combination
#   python compute_thresholds.py --all                        # all 16 + median
#
# Notes
# -----
# Q5: flow exceeded 5% of the time  -> flood/high-flow threshold
# Q80: flow exceeded 80% of the time -> drought/low-flow threshold
# numpy.percentile(x, 95) gives Q5; numpy.percentile(x, 20) gives Q80
# (Q80 is a LOW-flow statistic -- 80% of days sit ABOVE it -- so it
#  corresponds to the 20th percentile of the magnitude distribution,
#  not the 80th.)
#
# Water year Oct-Sep. Month extracted from date string (YYYY-MM-DD)
# without any datetime parsing -- safe for 360-day CHESS-SCAPE calendar.
#
# Baseline: WY1982 (starts 1 Oct 1981) to WY2010 (ends 30 Sep 2010)
#   BASELINE_START = "1981-10-01"
#   BASELINE_END   = "2010-09-30"
# Expected ~10440 daily timesteps (29 years x 360 days).

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
OUTPUT_DIR       = Path("./thresholds")

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]

# Baseline window as date strings (lexicographic comparison is safe for
# ISO 8601 format with the 360-day CHESS-SCAPE calendar).
# WY1982 starts 1 Oct 1981; WY2010 ends 30 Sep 2010.
BASELINE_START = "1981-10-01"
BASELINE_END   = "2010-09-30"


# ---------------------------------------------------------------------------
# Core threshold computation
# ---------------------------------------------------------------------------

def extract_baseline_mask(dates: pd.Series) -> np.ndarray:
    """
    Return boolean mask selecting the baseline water-year period.

    Dates are opaque strings. In practice the CHESS-SCAPE-derived CSVs carry
    a " 12:00:00" time suffix (e.g. "2010-09-30 12:00:00"), so a plain
    string comparison against a bare "YYYY-MM-DD" end date silently drops
    the final day of the baseline window (the longer string with an
    identical date prefix sorts as greater, so "<= BASELINE_END" excludes
    it). Truncating to the first 10 characters before comparing avoids
    that off-by-one regardless of whether a time suffix is present.
    """
    date_only = dates.str.slice(0, 10)
    return (date_only >= BASELINE_START) & (date_only <= BASELINE_END)


def compute_thresholds_one(discharge_csv: Path) -> pd.DataFrame:
    """
    Read one wide-format discharge CSV, extract the baseline period,
    and compute Q5 and Q80 for every catchment column.

    Returns DataFrame with columns: gauge_id, q5, q80.
    """
    df = pd.read_csv(discharge_csv, dtype={"date": str})

    catchment_ids = [c for c in df.columns if c != "date"]
    if not catchment_ids:
        raise ValueError(f"No catchment columns found in {discharge_csv.name}")

    mask = extract_baseline_mask(df["date"])
    n_baseline = int(mask.sum())

    if n_baseline == 0:
        raise ValueError(
            f"No baseline timesteps found in {discharge_csv.name}. "
            f"Check BASELINE_START/BASELINE_END and date column format. "
            f"First date in file: {df['date'].iloc[0]}, "
            f"last date: {df['date'].iloc[-1]}"
        )

    print(
        f"    Baseline timesteps: {n_baseline} "
        f"(expected ~10440 for 29 water years x 360 days)",
        flush=True,
    )

    # Extract baseline slice as numpy array: shape (n_timesteps, n_catchments)
    baseline = df.loc[mask, catchment_ids].to_numpy(dtype=np.float64)

    # np.percentile with axis=0 gives one value per catchment column
    #
    # Q5 = flow exceeded 5% of the time  -> the 95th percentile of magnitude
    #       (only 5% of days exceed it), so np.percentile(x, 95) is correct.
    # Q80 = flow exceeded 80% of the time -> this is a LOW-flow statistic:
    #       80% of days have flow ABOVE it, so it corresponds to the 20th
    #       percentile of magnitude, i.e. np.percentile(x, 20).
    q5_arr = np.percentile(baseline, 95, axis=0)
    q80_arr = np.percentile(baseline, 20, axis=0)

    return pd.DataFrame({
        "gauge_id": catchment_ids,
        "q5":      q5_arr,
        "q80":      q80_arr,
    })


# ---------------------------------------------------------------------------
# Ensemble median across all 16 combinations
# ---------------------------------------------------------------------------

def compute_ensemble_median(threshold_dir: Path) -> pd.DataFrame:
    """
    Load all per-combination threshold parquets and compute the median
    Q5 and Q80 across all available combinations per catchment.

    Returns DataFrame with columns: gauge_id, q5_median, q80_median.
    """
    frames = []
    for rcp in RCPS:
        for ens in ENSEMBLES:
            fpath = threshold_dir / f"thresholds_{rcp}_{ens}.parquet"
            if fpath.exists():
                frames.append(pd.read_parquet(fpath).set_index("gauge_id"))
            else:
                print(f"  WARNING: missing {fpath.name}, excluded from median", flush=True)

    if not frames:
        raise RuntimeError("No threshold parquets found -- run --all first.")

    n = len(frames)
    print(f"  Averaging across {n} combinations", flush=True)

    gauge_ids = frames[0].index.values
    q5_stack = np.stack([f["q5"].values for f in frames], axis=1)  # (n_catch, n_combos)
    q80_stack = np.stack([f["q80"].values for f in frames], axis=1)

    return pd.DataFrame({
        "gauge_id":   gauge_ids,
        "q5_median": np.median(q5_stack, axis=1),
        "q80_median": np.median(q80_stack, axis=1),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def process_one(rcp: str, ens: str, output_dir: Path) -> None:
    out_path = output_dir / f"thresholds_{rcp}_{ens}.parquet"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists", flush=True)
        return

    csv_path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_discharge.csv"
    if not csv_path.exists():
        print(f"  ERROR: discharge CSV not found: {csv_path}", flush=True)
        return

    print(f"  Computing thresholds: {rcp}/{ens} ...", flush=True)
    thresholds = compute_thresholds_one(csv_path)
    thresholds.to_parquet(out_path, index=False)

    print(f"  Written: {out_path.name}  ({len(thresholds)} catchments)", flush=True)
    print(
        f"  Q5 range: {thresholds['q5'].min():.4f} - "
        f"{thresholds['q5'].max():.4f} mm/day",
        flush=True,
    )
    print(
        f"  Q80 range: {thresholds['q80'].min():.4f} - "
        f"{thresholds['q80'].max():.4f} mm/day",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Q5/Q80 thresholds from HBV baseline discharge."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all 16 RCP/ensemble combinations and compute ensemble median.",
    )
    group.add_argument(
        "--rcp",
        choices=RCPS,
        help="Single RCP to process (requires --ensemble).",
    )
    parser.add_argument(
        "--ensemble",
        choices=ENSEMBLES,
        help="Single ensemble member (required with --rcp).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        print(f"Processing all {len(RCPS) * len(ENSEMBLES)} combinations...", flush=True)
        for rcp in RCPS:
            for ens in ENSEMBLES:
                process_one(rcp, ens, OUTPUT_DIR)

        print("\nComputing ensemble median...", flush=True)
        median_path = OUTPUT_DIR / "thresholds_ensemble_median.parquet"
        if median_path.exists():
            print(f"  [skip] {median_path.name} already exists", flush=True)
        else:
            median_df = compute_ensemble_median(OUTPUT_DIR)
            median_df.to_parquet(median_path, index=False)
            print(
                f"  Written: {median_path.name}  ({len(median_df)} catchments)",
                flush=True,
            )
            print(
                f"  Q5 median range: {median_df['q5_median'].min():.4f} - "
                f"{median_df['q5_median'].max():.4f} mm/day",
                flush=True,
            )
            print(
                f"  Q80 median range: {median_df['q80_median'].min():.4f} - "
                f"{median_df['q80_median'].max():.4f} mm/day",
                flush=True,
            )

    else:
        if not args.ensemble:
            print("ERROR: --ensemble is required when using --rcp")
            sys.exit(1)
        process_one(args.rcp, args.ensemble, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
