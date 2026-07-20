"""
catchment_filter.py
===================

Utility for filtering CAMELS-GB v2 catchments by HBV model performance.

Reads validation KGE from the hbv-model repository's
calibrated_parameters.csv (columns: gauge_id, ..., validation_kge,
used_in_analysis), rather than maintaining a separate, redundant KGE
file in this repository.

Usage
-----
Import this module in any script that needs to restrict analysis to
well-performing catchments:

    from catchment_filter import get_valid_catchments, filter_dataframe

    valid_ids = get_valid_catchments()          # default KGE_val >= 0.5
    valid_ids = get_valid_catchments(kge_threshold=0.6)   # stricter

    # Filter a DataFrame whose index or a column contains catchment IDs
    filtered = filter_dataframe(some_df, valid_ids)

The filter applies to validation KGE. Catchments with NaN validation
KGE are always excluded regardless of threshold.

Configuration
-------------
Set KGE_PATH to the path of hbv-model's calibrated_parameters.csv.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — point this at hbv-model's calibrated_parameters.csv
# ---------------------------------------------------------------------------
KGE_PATH = "../hbv-model/calibrated_parameters.csv"

# Default threshold — validation KGE >= this value required for inclusion
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_valid_catchments(
    kge_path: str = KGE_PATH,
    kge_threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = True
) -> set:
    """
    Return a set of catchment IDs with validation KGE >= kge_threshold.

    NaN validation KGE values are always excluded.

    Parameters
    ----------
    kge_path      : path to hbv-model's calibrated_parameters.csv
    kge_threshold : minimum acceptable validation KGE (default 0.5)
    verbose       : print a summary of included/excluded counts

    Returns
    -------
    set of integer catchment IDs
    """
    if not os.path.exists(kge_path):
        raise FileNotFoundError(
            f"Calibrated parameters file not found: {kge_path}\n"
            "Update KGE_PATH in catchment_filter.py or pass kge_path explicitly. "
            "This should point at hbv-model's calibrated_parameters.csv."
        )

    kge = pd.read_csv(kge_path)

    # Validate expected columns
    required = {"gauge_id", "validation_kge"}
    if not required.issubset(kge.columns):
        raise ValueError(
            f"File must contain columns {required} (from hbv-model's "
            f"calibrated_parameters.csv). Found: {set(kge.columns)}"
        )

    n_total   = len(kge)
    n_nan     = kge["validation_kge"].isna().sum()
    kge_valid = kge.dropna(subset=["validation_kge"])
    n_below   = (kge_valid["validation_kge"] < kge_threshold).sum()
    valid_ids = set(
        kge_valid.loc[kge_valid["validation_kge"] >= kge_threshold, "gauge_id"]
        .astype(int)
    )

    if verbose:
        print(f"Catchment filter (validation KGE >= {kge_threshold}):")
        print(f"  Total catchments:          {n_total}")
        print(f"  Excluded — NaN val KGE:    {n_nan}")
        print(f"  Excluded — below threshold:{n_below}")
        print(f"  Retained:                  {len(valid_ids)} "
              f"({100 * len(valid_ids) / n_total:.1f}%)")

    return valid_ids


def filter_dataframe(
    df: pd.DataFrame,
    valid_ids: set,
    id_col: str = None
) -> pd.DataFrame:
    """
    Filter a DataFrame to retain only valid catchment IDs.

    Works in two modes:
    - If id_col is None: assumes the DataFrame index contains catchment IDs
    - If id_col is a column name: filters on that column

    Parameters
    ----------
    df        : DataFrame to filter
    valid_ids : set of valid catchment IDs (from get_valid_catchments)
    id_col    : column name containing catchment IDs, or None to use index

    Returns
    -------
    Filtered DataFrame (copy)
    """
    if id_col is None:
        mask = df.index.astype(int).isin(valid_ids)
        return df.loc[mask].copy()
    else:
        mask = df[id_col].astype(int).isin(valid_ids)
        return df.loc[mask].copy()


def filter_array_columns(
    df: pd.DataFrame,
    valid_ids: set
) -> pd.DataFrame:
    """
    Filter a wide-format discharge DataFrame where columns are catchment IDs.

    Used when loading ensemble mean discharge CSVs (rows = dates,
    columns = catchment IDs).

    Parameters
    ----------
    df        : wide-format DataFrame with catchment IDs as column names
    valid_ids : set of valid catchment IDs

    Returns
    -------
    DataFrame with only valid catchment columns retained
    """
    valid_cols = [
        c for c in df.columns
        if str(c) != "date" and int(c) in valid_ids
    ]
    if "date" in df.columns:
        return df[["date"] + valid_cols].copy()
    return df[valid_cols].copy()


# ---------------------------------------------------------------------------
# Convenience: print summary without returning IDs
# ---------------------------------------------------------------------------
def print_exclusion_summary(
    kge_path: str = KGE_PATH,
    kge_threshold: float = DEFAULT_THRESHOLD
):
    """Print a detailed breakdown of excluded catchments."""
    kge = pd.read_csv(kge_path)
    excluded = kge[
        kge["validation_kge"].isna() | (kge["validation_kge"] < kge_threshold)
    ].copy()
    excluded = excluded.sort_values("validation_kge")
    print(f"\nExcluded catchments (validation KGE < {kge_threshold} or NaN):")
    print(excluded[["gauge_id", "calibration_kge", "validation_kge"]].to_string(index=False))
    print(f"\nTotal excluded: {len(excluded)}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    valid = get_valid_catchments(verbose=True)
    print(f"\nFirst 10 valid IDs: {sorted(valid)[:10]}")
    print_exclusion_summary()
