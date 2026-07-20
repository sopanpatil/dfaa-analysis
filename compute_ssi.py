# compute_ssi.py
# --------------
# Compute the Standardised Streamflow Index at 1-month aggregation (SSI-1)
# for baseline and future periods across all 621 HBV catchments.
#
# Method: nonparametric (empirical) SSI following Tijdeman et al. (2020, WRR).
# The empirical CDF is fitted to baseline monthly flows by calendar month,
# then future monthly flows are mapped to standard normal deviates via the
# same empirical CDF. This avoids the Tweedie distribution fitting used by
# He et al. (2026), which requires R. The nonparametric approach is
# defensible for transition detection (as opposed to extreme drought return
# period estimation) and is less sensitive to sample size at the tails.
#
# Reference:
#   Tijdeman et al. (2020) Drought Characteristics Derived Based on the
#   Standardized Streamflow Index: A Large Sample Comparison for Parametric
#   and Nonparametric Methods. WRR. doi:10.1029/2019WR026315
#
# Inputs
# ------
# Wide-format HBV discharge CSVs (one per RCP/ensemble):
#   <CHESS_SCAPE_ROOT>/<rcp>_<ens>_hbv_discharge.csv
#   Columns: date (YYYY-MM-DD, opaque string), <catchment_id>, ...
#
# Outputs (in <OUTPUT_DIR>/ssi/)
# ------
#   ssi_baseline_<rcp>_<ens>.parquet  -- SSI-1 for baseline period
#   ssi_future_<rcp>_<ens>.parquet    -- SSI-1 for future period
#   Columns: year_month (YYYY-MM str), <catchment_id>, ...
#
# Periods
# -------
#   Baseline: WY1982-WY2010  (1981-10-01 to 2010-09-30)
#   Future:   WY2051-WY2080  (2050-10-01 to 2080-09-30)
#
# Usage
# -----
#   python compute_ssi.py --rcp rcp26 --ensemble 01   # single combination
#   python compute_ssi.py --all                        # all 16 combinations
#
# Calendar note
# -------------
# CHESS-SCAPE uses a 360-day calendar (12 months x 30 days). Dates are kept
# as opaque YYYY-MM-DD strings throughout. Month is extracted from the string
# directly (characters 5-6). No datetime parsing is used.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
OUTPUT_DIR       = Path("./ssi_output")

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]

# Period boundaries (lexicographic string comparison, ISO 8601)
BASELINE_START = "1981-10-01"
BASELINE_END   = "2010-09-30"
FUTURE_START   = "2050-10-01"
FUTURE_END     = "2080-09-30"

# Plotting position parameter for empirical CDF (Weibull: a=0)
# P(x_i) = (i - a) / (n + 1 - 2a), i = 1..n sorted ascending
# Weibull (a=0): P = i / (n+1)  -- unbiased, standard for SSI
PLOTTING_POSITION_A = 0.0

# Clamp SSI to avoid infinite values from exact 0 or 1 probabilities
SSI_CLAMP = 1e-6


# ---------------------------------------------------------------------------
# Utility: extract month string from date string
# ---------------------------------------------------------------------------

def month_from_date(date_str: str) -> str:
    """Extract 'YYYY-MM' from 'YYYY-MM-DD' without datetime parsing."""
    return date_str[:7]


def month_number_from_date(date_str: str) -> int:
    """Extract integer month (1-12) from 'YYYY-MM-DD' without datetime parsing."""
    return int(date_str[5:7])


# ---------------------------------------------------------------------------
# Aggregate daily flow to monthly means
# ---------------------------------------------------------------------------

def daily_to_monthly(
    daily_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Aggregate daily wide-format discharge to monthly means for a given period.

    Parameters
    ----------
    daily_df : DataFrame
        Columns: date (str YYYY-MM-DD), catchment_id columns (float)
    start, end : str
        Period boundaries as YYYY-MM-DD strings (inclusive).

    Returns
    -------
    DataFrame with columns: year_month (str YYYY-MM), catchment columns.
    Rows ordered chronologically.
    """
    # Dates carry a " 12:00:00" time suffix in the source CSVs, so a plain
    # string comparison against a bare "YYYY-MM-DD" end date silently
    # drops the final day of the period (a longer string with an
    # identical date prefix sorts as greater, so "<= end" excludes it).
    # Truncate to the date portion before comparing to avoid that.
    date_only = daily_df["date"].str.slice(0, 10)
    mask = (date_only >= start) & (date_only <= end)
    period = daily_df.loc[mask].copy()

    catchment_cols = [c for c in period.columns if c != "date"]

    # Extract YYYY-MM from date string without datetime parsing
    period["year_month"] = period["date"].str[:7]

    monthly = (
        period.groupby("year_month", sort=True)[catchment_cols]
        .mean()
        .reset_index()
    )
    return monthly


# ---------------------------------------------------------------------------
# Nonparametric SSI computation
# ---------------------------------------------------------------------------

def empirical_cdf_prob(
    value: float,
    baseline_values: np.ndarray,
    a: float = PLOTTING_POSITION_A,
) -> float:
    """
    Map a single value to its empirical CDF probability using the
    Weibull plotting position fitted to baseline_values.

    baseline_values must be sorted ascending before calling (or sort here).
    Uses linear interpolation between plotting positions.

    If value < min(baseline): return plotting_position of rank 1
    If value > max(baseline): return plotting_position of rank n
    """
    n = len(baseline_values)
    sorted_vals = np.sort(baseline_values)

    # Plotting positions: P_i = (i - a) / (n + 1 - 2a), i = 1..n
    ranks = np.arange(1, n + 1, dtype=float)
    probs = (ranks - a) / (n + 1 - 2 * a)

    # Linear interpolation
    prob = np.interp(value, sorted_vals, probs)
    return float(prob)


def compute_ssi_for_catchment(
    baseline_monthly: np.ndarray,   # shape (n_baseline_months,)
    baseline_months:  np.ndarray,   # shape (n_baseline_months,) int 1-12
    target_monthly:   np.ndarray,   # shape (n_target_months,)
    target_months:    np.ndarray,   # shape (n_target_months,) int 1-12
) -> np.ndarray:
    """
    Compute SSI-1 for one catchment.

    For each calendar month (1-12), fit an empirical CDF to the baseline
    flows for that month, then map target flows to probabilities, then
    transform to standard normal deviates.

    Returns SSI array of shape (n_target_months,).
    """
    n_target = len(target_monthly)
    ssi = np.full(n_target, np.nan)

    for m in range(1, 13):
        # Baseline flows for this calendar month
        baseline_mask = baseline_months == m
        baseline_m    = baseline_monthly[baseline_mask]

        if len(baseline_m) < 3:
            # Insufficient baseline data for this month -- leave as NaN
            continue

        # Target indices for this calendar month
        target_mask = target_months == m
        target_idx  = np.where(target_mask)[0]

        for idx in target_idx:
            prob = empirical_cdf_prob(target_monthly[idx], baseline_m)
            # Clamp to avoid inf from norm.ppf(0) or norm.ppf(1)
            prob = np.clip(prob, SSI_CLAMP, 1.0 - SSI_CLAMP)
            ssi[idx] = norm.ppf(prob)

    return ssi


# ---------------------------------------------------------------------------
# Main computation for one RCP/ensemble combination
# ---------------------------------------------------------------------------

def compute_ssi_one(
    rcp: str,
    ens: str,
    output_dir: Path,
) -> None:
    baseline_out = output_dir / f"ssi_baseline_{rcp}_{ens}.parquet"
    future_out   = output_dir / f"ssi_future_{rcp}_{ens}.parquet"

    if baseline_out.exists() and future_out.exists():
        print(f"  [skip] SSI outputs already exist for {rcp}/{ens}", flush=True)
        return

    csv_path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_discharge.csv"
    if not csv_path.exists():
        print(f"  ERROR: discharge CSV not found: {csv_path}", flush=True)
        return

    print(f"  Computing SSI: {rcp}/{ens} ...", flush=True)

    # Read daily discharge -- dates as strings
    daily_df = pd.read_csv(csv_path, dtype={"date": str})
    catchment_ids = [c for c in daily_df.columns if c != "date"]
    n_catch = len(catchment_ids)

    # Aggregate to monthly means
    print("    Aggregating daily -> monthly ...", flush=True)
    baseline_monthly = daily_to_monthly(daily_df, BASELINE_START, BASELINE_END)
    future_monthly   = daily_to_monthly(daily_df, FUTURE_START,   FUTURE_END)

    n_baseline_months = len(baseline_monthly)
    n_future_months   = len(future_monthly)
    print(
        f"    Baseline months: {n_baseline_months} "
        f"(expected 348 for 29 WYs x 12 months)",
        flush=True,
    )
    print(
        f"    Future months:   {n_future_months} "
        f"(expected 360 for 30 WYs x 12 months)",
        flush=True,
    )

    # Extract integer month arrays (no datetime parsing)
    baseline_month_nums = baseline_monthly["year_month"].str[5:7].astype(int).values
    future_month_nums   = future_monthly["year_month"].str[5:7].astype(int).values

    # Compute SSI for every catchment
    ssi_baseline_dict: dict[str, np.ndarray] = {
        "year_month": baseline_monthly["year_month"].values
    }
    ssi_future_dict: dict[str, np.ndarray] = {
        "year_month": future_monthly["year_month"].values
    }

    for i, cid in enumerate(catchment_ids):
        if (i + 1) % 100 == 0 or (i + 1) == n_catch:
            print(f"    ...{i+1}/{n_catch} catchments", flush=True)

        baseline_flow = baseline_monthly[cid].to_numpy(dtype=np.float64)
        future_flow   = future_monthly[cid].to_numpy(dtype=np.float64)

        # SSI for baseline period (self-fitted: baseline flow against
        # itself -- by definition centred near zero)
        ssi_b = compute_ssi_for_catchment(
            baseline_flow, baseline_month_nums,
            baseline_flow, baseline_month_nums,
        )

        # SSI for future period (fitted against baseline CDF)
        ssi_f = compute_ssi_for_catchment(
            baseline_flow, baseline_month_nums,
            future_flow,   future_month_nums,
        )

        ssi_baseline_dict[cid] = ssi_b
        ssi_future_dict[cid]   = ssi_f

    # Write outputs
    pd.DataFrame(ssi_baseline_dict).to_parquet(baseline_out, index=False)
    pd.DataFrame(ssi_future_dict).to_parquet(future_out, index=False)

    print(f"  Written: {baseline_out.name}", flush=True)
    print(f"  Written: {future_out.name}", flush=True)

    # Sanity check: baseline SSI should be near zero mean
    sample_cid = catchment_ids[0]
    b_mean = np.nanmean(ssi_baseline_dict[sample_cid])
    print(
        f"  Sanity check ({sample_cid}): baseline SSI mean = {b_mean:.3f} "
        f"(expected ~0.0)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute nonparametric SSI-1 for baseline and future periods."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all 16 RCP/ensemble combinations.",
    )
    group.add_argument(
        "--rcp",
        choices=RCPS,
        help="Single RCP (requires --ensemble).",
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
        total = len(RCPS) * len(ENSEMBLES)
        print(f"Processing all {total} combinations ...", flush=True)
        for rcp in RCPS:
            for ens in ENSEMBLES:
                compute_ssi_one(rcp, ens, OUTPUT_DIR)
    else:
        if not args.ensemble:
            print("ERROR: --ensemble is required when using --rcp")
            sys.exit(1)
        compute_ssi_one(args.rcp, args.ensemble, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
