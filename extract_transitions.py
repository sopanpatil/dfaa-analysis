# extract_transitions.py
# ----------------------
# Daily flood event, drought event, and flood-to-drought transition
# extraction across all 621 HBV catchments for baseline and future
# periods, all four RCPs and four ensemble members.
#
# Thresholds: ensemble median Q5 (flood) and Q80 (drought) from
#   thresholds/thresholds_ensemble_median.parquet
# Applied identically to baseline and future periods across all
# RCP/ensemble combinations to enable clean cross-scenario comparison.
#
# Notes
# -----
#   - thresholds_ensemble_median.parquet (from compute_thresholds.py) must
#     be regenerated before rerunning this script whenever the baseline
#     period changes, since Q5/Q80 are computed from that period.
#   - Dates carry a " 12:00:00" time suffix in the source CSVs;
#     _extract_period truncates to the date portion before comparing
#     against period boundaries (see its docstring below).
#
# Inputs
# ------
#   chess_scape_output/<rcp>_<ens>_hbv_discharge.csv
#   chess_scape_output/<rcp>_<ens>_hbv_sm.csv
#   chess_scape_output/<rcp>_<ens>_hbv_uz.csv
#   chess_scape_output/<rcp>_<ens>_hbv_lz.csv
#   chess_scape_output/<rcp>_<ens>_hbv_sp.csv
#   thresholds/thresholds_ensemble_median.parquet
#
# Outputs (in transitions_output/)
# -------
# One parquet per RCP/ensemble/period combination:
#   transitions_<rcp>_<ens>_baseline.parquet
#   transitions_<rcp>_<ens>_future.parquet
#
# Columns per output file:
#   gauge_id            : catchment identifier
#   flood_start_idx     : integer index into period flow array
#   flood_end_idx
#   flood_peak_idx
#   flood_peak_flow     : mm/day
#   flood_duration      : days
#   drought_start_idx
#   drought_end_idx
#   drought_min_idx
#   drought_min_flow    : mm/day
#   drought_duration    : days
#   drought_severity    : cumulative deficit mm
#   gap_days            : days from flood end to drought start
#   sm_at_flood_peak    : HBV SM store (mm)
#   sm_at_flood_end     : HBV SM store (mm)
#   sm_at_drought_start : HBV SM store (mm)
#   sm_depletion_rate   : (sm_flood_end - sm_drought_start) / gap_days (mm/day)
#   uz_at_flood_peak    : HBV UZ store (mm)
#   lz_at_flood_peak    : HBV LZ store (mm)
#   sp_at_flood_peak    : HBV SP (snowpack) store (mm)
#   sp_at_drought_start : HBV SP (snowpack) store (mm)
#   flood_start_date    : date string from original CSV
#   drought_start_date  : date string from original CSV
#
# Periods
# -------
#   Baseline: WY1982-WY2010  1981-10-01 to 2010-09-30
#   Future:   WY2051-WY2080  2050-10-01 to 2080-09-30
#
# Usage
# -----
#   # Single combination, both periods (good for testing)
#   python extract_transitions.py --rcp rcp26 --ensemble 01
#
#   # All 16 combinations (submit as Slurm array on LOTUS)
#   python extract_transitions.py --all
#
#   # Single combination, one period only
#   python extract_transitions.py --rcp rcp26 --ensemble 01 --period future
#
# LOTUS array usage
# -----------------
#   --task-id below selects one RCP/ensemble combination (0-15) from the
#   flat list in ALL_COMBINATIONS, for use in a Slurm array job submission
#   script on JASMIN's LOTUS cluster (one array task per combination).

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
THRESHOLD_DIR    = Path("./thresholds")
OUTPUT_DIR       = Path("./transitions_output")

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]

# All 16 combinations in a flat list (used for Slurm array indexing)
ALL_COMBINATIONS = [(r, e) for r in RCPS for e in ENSEMBLES]

BASELINE_START = "1981-10-01"
BASELINE_END   = "2010-09-30"
FUTURE_START   = "2050-10-01"
FUTURE_END     = "2080-09-30"

# Event extraction parameters
FLOOD_MIN_DURATION  = 1    # days; minimum flood event length
FLOOD_POOL_GAP      = 5    # days; merge flood events closer than this
DROUGHT_MIN_DUR     = 5    # days; minimum drought spell length (sole
                           # independence criterion -- no inter-event pooling)
MAX_TRANSITION_GAP  = 90   # days; max gap from flood end to drought start

N_WORKERS = 8


# ---------------------------------------------------------------------------
# Event extraction functions (self-contained, picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _extract_flood_events(flow, q5):
    above = flow > q5
    events = []
    in_event = False
    start = 0
    for i in range(len(flow)):
        if above[i] and not in_event:
            start = i
            in_event = True
        elif not above[i] and in_event:
            events.append([start, i - 1])
            in_event = False
    if in_event:
        events.append([start, len(flow) - 1])

    # Pool events separated by fewer than FLOOD_POOL_GAP days
    if len(events) > 1:
        pooled = [events[0]]
        for ev in events[1:]:
            if ev[0] - pooled[-1][1] <= FLOOD_POOL_GAP:
                pooled[-1][1] = ev[1]
            else:
                pooled.append(ev)
        events = pooled

    result = []
    for s, e in events:
        dur = e - s + 1
        if dur >= FLOOD_MIN_DURATION:
            seg = flow[s:e + 1]
            peak_idx = s + int(np.argmax(seg))
            result.append({
                "flood_start_idx":  s,
                "flood_end_idx":    e,
                "flood_peak_idx":   peak_idx,
                "flood_peak_flow":  float(flow[peak_idx]),
                "flood_duration":   dur,
            })
    return result


def _extract_drought_events(flow, q80):
    below = flow < q80
    events = []
    in_event = False
    start = 0
    for i in range(len(flow)):
        if below[i] and not in_event:
            start = i
            in_event = True
        elif not below[i] and in_event:
            events.append([start, i - 1])
            in_event = False
    if in_event:
        events.append([start, len(flow) - 1])

    # No inter-event pooling: each maximal run below Q80 is retained as a
    # distinct event. The DROUGHT_MIN_DUR filter below is the sole
    # independence criterion (Methods 2.2).

    result = []
    for s, e in events:
        dur = e - s + 1
        if dur >= DROUGHT_MIN_DUR:
            seg = flow[s:e + 1]
            min_idx  = s + int(np.argmin(seg))
            severity = float(np.sum(q80 - seg))
            result.append({
                "drought_start_idx": s,
                "drought_end_idx":   e,
                "drought_min_idx":   min_idx,
                "drought_min_flow":  float(flow[min_idx]),
                "drought_duration":  dur,
                "drought_severity":  severity,
            })
    return result


def _find_transitions(floods, droughts):
    """
    One-to-one, temporally-ordered pairing: process flood-end and
    drought-start events in a single chronological pass. Each new flood
    supersedes any earlier unresolved flood (so a flood immediately
    followed by another flood, before any drought occurs, is not paired
    with a later, non-adjacent drought). Each drought consumes whichever
    flood is currently pending -- if any -- and that flood is then
    unavailable to any other drought. This guarantees every flood and
    every drought participates in at most one transition, and that each
    transition represents an uninterrupted flood-to-drought episode with
    no other flood or drought event intervening.
    """
    if not floods or not droughts:
        return []

    events = []
    for f in floods:
        events.append((f["flood_end_idx"], 0, "flood", f))
    for d in droughts:
        events.append((d["drought_start_idx"], 1, "drought", d))
    events.sort(key=lambda e: (e[0], e[1]))

    transitions = []
    pending_flood = None
    for _, _, kind, obj in events:
        if kind == "flood":
            pending_flood = obj
        else:
            if pending_flood is not None:
                gap_days = obj["drought_start_idx"] - pending_flood["flood_end_idx"] - 1
                if gap_days <= MAX_TRANSITION_GAP:
                    row = {}
                    row.update(pending_flood)
                    row.update(obj)
                    row["gap_days"] = gap_days
                    transitions.append(row)
            pending_flood = None
    return transitions


def _attach_states(transitions, sm, uz, lz, sp):
    for t in transitions:
        fp  = t["flood_peak_idx"]
        fe  = t["flood_end_idx"]
        ds  = t["drought_start_idx"]
        gap = t["gap_days"]

        t["sm_at_flood_peak"]    = float(sm[fp])
        t["sm_at_flood_end"]     = float(sm[fe])
        t["sm_at_drought_start"] = float(sm[ds])
        t["uz_at_flood_peak"]    = float(uz[fp])
        t["lz_at_flood_peak"]    = float(lz[fp])
        t["sp_at_flood_peak"]    = float(sp[fp])
        t["sp_at_drought_start"] = float(sp[ds])

        if gap > 0:
            t["sm_depletion_rate"] = (float(sm[fe]) - float(sm[ds])) / gap
        else:
            t["sm_depletion_rate"] = np.nan
    return transitions


# ---------------------------------------------------------------------------
# Per-catchment worker (runs in subprocess -- must be picklable)
# ---------------------------------------------------------------------------

def _process_catchment(
    cid:      str,
    flow:     np.ndarray,
    sm:       np.ndarray,
    uz:       np.ndarray,
    lz:       np.ndarray,
    sp:       np.ndarray,
    dates:    np.ndarray,   # string array, same length as flow
    q5:      float,
    q80:      float,
) -> tuple[str, list]:
    """
    Extract transitions for one catchment. Returns (cid, list_of_dicts).
    Each dict is one transition event with all columns.
    """
    try:
        floods      = _extract_flood_events(flow, q5)
        droughts    = _extract_drought_events(flow, q80)
        transitions = _find_transitions(floods, droughts)
        transitions = _attach_states(transitions, sm, uz, lz, sp)

        # Attach date strings for interpretability
        for t in transitions:
            t["gauge_id"]           = cid
            t["flood_start_date"]   = dates[t["flood_start_idx"]]
            t["drought_start_date"] = dates[t["drought_start_idx"]]

        return cid, transitions

    except Exception as exc:
        print(f"  ERROR: {cid}: {exc}", flush=True)
        return cid, []


# ---------------------------------------------------------------------------
# Period extraction helper
# ---------------------------------------------------------------------------

def _extract_period(
    q_df:  pd.DataFrame,
    sm_df: pd.DataFrame,
    uz_df: pd.DataFrame,
    lz_df: pd.DataFrame,
    sp_df: pd.DataFrame,
    start: str,
    end:   str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Slice all five wide DataFrames to the requested period.
    Returns (dates_array, q_slice, sm_slice, uz_slice, lz_slice, sp_slice).

    Dates carry a " 12:00:00" time suffix in the source CSVs, so we
    truncate to the date portion before comparing -- a plain string
    comparison against a bare "YYYY-MM-DD" end date would otherwise
    silently drop the final day of the period (a longer string with an
    identical date prefix sorts as greater, so "<= end" excludes it).
    """
    date_only = q_df["date"].str.slice(0, 10)
    mask = (date_only >= start) & (date_only <= end)
    dates = q_df.loc[mask, "date"].to_numpy(dtype=str)

    catchment_cols = [c for c in q_df.columns if c != "date"]

    q_slice  = q_df.loc[mask,  catchment_cols]
    sm_slice = sm_df.loc[mask, catchment_cols]
    uz_slice = uz_df.loc[mask, catchment_cols]
    lz_slice = lz_df.loc[mask, catchment_cols]
    sp_slice = sp_df.loc[mask, catchment_cols]

    n = mask.sum()
    print(
        f"    Period {start} to {end}: {n} timesteps "
        f"(expected ~10440 baseline / ~10800 future)",
        flush=True,
    )
    return dates, q_slice, sm_slice, uz_slice, lz_slice, sp_slice


# ---------------------------------------------------------------------------
# Main extraction for one RCP/ensemble/period
# ---------------------------------------------------------------------------

def extract_one_period(
    rcp:        str,
    ens:        str,
    period:     str,       # "baseline" or "future"
    thresholds: pd.DataFrame,
    output_dir: Path,
) -> None:
    out_path = output_dir / f"transitions_{rcp}_{ens}_{period}.parquet"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists", flush=True)
        return

    start = BASELINE_START if period == "baseline" else FUTURE_START
    end   = BASELINE_END   if period == "baseline" else FUTURE_END

    # Load discharge and state CSVs
    def _csv(suffix):
        return CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_{suffix}.csv"

    for suffix in ["discharge", "sm", "uz", "lz", "sp"]:
        if not _csv(suffix).exists():
            print(f"  ERROR: missing {_csv(suffix).name}", flush=True)
            return

    print(f"  Loading CSVs for {rcp}/{ens} ...", flush=True)
    q_df  = pd.read_csv(_csv("discharge"), dtype={"date": str})
    sp_df = pd.read_csv(_csv("sp"),        dtype={"date": str})
    sm_df = pd.read_csv(_csv("sm"),                  dtype={"date": str})
    uz_df = pd.read_csv(_csv("uz"),                  dtype={"date": str})
    lz_df = pd.read_csv(_csv("lz"),                  dtype={"date": str})

    # Align index across DataFrames (all should be identical but verify)
    assert list(q_df["date"]) == list(sm_df["date"]), \
        "Date mismatch between discharge and SM CSVs"
    assert list(q_df["date"]) == list(sp_df["date"]), \
        "Date mismatch between discharge and SP CSVs"

    dates, q_sl, sm_sl, uz_sl, lz_sl, sp_sl = _extract_period(
        q_df, sm_df, uz_df, lz_df, sp_df, start, end
    )

    catchment_ids = [c for c in q_df.columns if c != "date"]

    # Build threshold lookup: gauge_id -> (q5, q80)
    thresh_lookup = thresholds.set_index("gauge_id")

    # Build work items -- only catchments present in both data and thresholds
    work = []
    skipped = []
    for cid in catchment_ids:
        if cid not in thresh_lookup.index:
            skipped.append(cid)
            continue
        q5 = float(thresh_lookup.loc[cid, "q5_median"])
        q80 = float(thresh_lookup.loc[cid, "q80_median"])
        work.append((
            cid,
            q_sl[cid].to_numpy(dtype=np.float64),
            sm_sl[cid].to_numpy(dtype=np.float64),
            uz_sl[cid].to_numpy(dtype=np.float64),
            lz_sl[cid].to_numpy(dtype=np.float64),
            sp_sl[cid].to_numpy(dtype=np.float64),
            dates,
            q5,
            q80,
        ))

    if skipped:
        print(
            f"  WARNING: {len(skipped)} catchments not in threshold file, skipped",
            flush=True,
        )

    print(
        f"  Extracting transitions: {len(work)} catchments, "
        f"period={period} ...",
        flush=True,
    )

    # Run in parallel across catchments
    all_rows = []
    completed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_process_catchment, *item): item[0]
            for item in work
        }
        for future in as_completed(futures):
            cid, rows = future.result()
            all_rows.extend(rows)
            completed += 1
            if completed % 100 == 0 or completed == len(work):
                print(f"    ...{completed}/{len(work)} done", flush=True)

    if not all_rows:
        print(f"  WARNING: no transitions found for {rcp}/{ens}/{period}", flush=True)
        # Write empty parquet so skip logic works on re-run
        pd.DataFrame().to_parquet(out_path, index=False)
        return

    result_df = pd.DataFrame(all_rows)

    # Reorder columns for clarity
    col_order = [
        "gauge_id",
        "flood_start_date", "drought_start_date",
        "flood_start_idx", "flood_end_idx", "flood_peak_idx",
        "flood_peak_flow", "flood_duration",
        "drought_start_idx", "drought_end_idx", "drought_min_idx",
        "drought_min_flow", "drought_duration", "drought_severity",
        "gap_days",
        "sm_at_flood_peak", "sm_at_flood_end", "sm_at_drought_start",
        "sm_depletion_rate",
        "uz_at_flood_peak", "lz_at_flood_peak",
        "sp_at_flood_peak", "sp_at_drought_start",
    ]
    result_df = result_df[[c for c in col_order if c in result_df.columns]]

    result_df.to_parquet(out_path, index=False)

    n_transitions = len(result_df)
    n_catchments  = result_df["gauge_id"].nunique()
    print(
        f"  Written: {out_path.name}  "
        f"({n_transitions} transitions across {n_catchments} catchments)",
        flush=True,
    )
    print(
        f"  Gap days: median={result_df['gap_days'].median():.1f}  "
        f"mean={result_df['gap_days'].mean():.1f}  "
        f"max={result_df['gap_days'].max()}",
        flush=True,
    )
    print(
        f"  SM depletion rate: median={result_df['sm_depletion_rate'].median():.4f} "
        f"mm/day",
        flush=True,
    )


def extract_one_combination(
    rcp:        str,
    ens:        str,
    periods:    list[str],
    thresholds: pd.DataFrame,
    output_dir: Path,
) -> None:
    print(f"\n{'='*55}", flush=True)
    print(f"  {rcp} / {ens}", flush=True)
    print(f"{'='*55}", flush=True)
    for period in periods:
        extract_one_period(rcp, ens, period, thresholds, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract flood-to-drought transitions from HBV discharge "
            "and internal state CSVs."
        )
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
    group.add_argument(
        "--task-id",
        type=int,
        metavar="N",
        help=(
            "Slurm array task ID (0-15). Selects one RCP/ensemble combination "
            "from the fixed order: rcp26/01, rcp26/04, rcp26/06, rcp26/15, "
            "rcp45/01, ... rcp85/15."
        ),
    )
    parser.add_argument(
        "--ensemble",
        choices=ENSEMBLES,
        help="Ensemble member (required with --rcp).",
    )
    parser.add_argument(
        "--period",
        choices=["baseline", "future", "both"],
        default="both",
        help="Which period(s) to process (default: both).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"Parallel workers per combination (default: {N_WORKERS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ensemble median thresholds (used for all combinations)
    threshold_path = THRESHOLD_DIR / "thresholds_ensemble_median.parquet"
    if not threshold_path.exists():
        print(f"ERROR: threshold file not found: {threshold_path}")
        print("Run compute_thresholds.py --all first.")
        sys.exit(1)
    thresholds = pd.read_parquet(threshold_path)
    print(
        f"Loaded thresholds for {len(thresholds)} catchments "
        f"from {threshold_path.name}",
        flush=True,
    )

    periods = ["baseline", "future"] if args.period == "both" else [args.period]

    global N_WORKERS
    N_WORKERS = args.workers

    if args.all:
        for rcp, ens in ALL_COMBINATIONS:
            extract_one_combination(rcp, ens, periods, thresholds, OUTPUT_DIR)

    elif args.task_id is not None:
        # Slurm array mode: task ID 0-15 maps to one combination
        if args.task_id < 0 or args.task_id >= len(ALL_COMBINATIONS):
            print(
                f"ERROR: --task-id must be 0-{len(ALL_COMBINATIONS)-1}, "
                f"got {args.task_id}"
            )
            sys.exit(1)
        rcp, ens = ALL_COMBINATIONS[args.task_id]
        extract_one_combination(rcp, ens, periods, thresholds, OUTPUT_DIR)

    else:
        if not args.ensemble:
            print("ERROR: --ensemble is required with --rcp")
            sys.exit(1)
        extract_one_combination(
            args.rcp, args.ensemble, periods, thresholds, OUTPUT_DIR
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
