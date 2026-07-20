# quantify_pairing_supersession.py
# ---------------------------------
# Standalone diagnostic quantifying how often the one-to-one, temporally
# ordered pairing rule (Methods 2.2) SUPERSEDES (drops) a precursor
# flood/drought event, versus CONSUMES it into a registered transition.
# Backs the pairing-rule statistics reported in Methods 2.2 and the
# Discussion (75.1% of flood events and 59.5% of drought events
# superseded, pooled across all catchments/RCPs/ensembles/periods).
#
# Re-derives flood/drought events independently, with the identical
# detection logic used in extract_transitions.py / explore_drought_to_flood.py,
# so it has no import dependency on either pipeline script and never
# modifies their outputs -- it independently re-reads the same discharge/
# state CSVs and the same threshold file, re-runs event detection with
# the IDENTICAL logic, and re-runs pairing with instrumentation added,
# writing its own diagnostic CSVs to ./pairing_diagnostics_output/.
#
# Requires (same as extract_transitions.py / explore_drought_to_flood.py):
#   chess_scape_output/<rcp>_<ens>_hbv_discharge.csv
#   chess_scape_output/<rcp>_<ens>_hbv_sm.csv
#   thresholds/thresholds_ensemble_median.parquet
#
# Usage
# -----
#   # Single combination, both periods (good for testing)
#   python quantify_pairing_supersession.py --rcp rcp85 --ensemble 01
#
#   # All 16 combinations
#   python quantify_pairing_supersession.py --all
#
#   # Slurm array (one task per RCP/ensemble combination, same convention
#   # as extract_transitions.py --task-id)
#   python quantify_pairing_supersession.py --task-id 0
#
#   # After all combinations have been run once, pool everything into
#   # the final headline number (safe to re-run any time; just reads the
#   # per-combination CSVs already written):
#   python quantify_pairing_supersession.py --summarise-only
#
# Output
# ------
#   pairing_diagnostics_output/diagnostics_<rcp>_<ens>_<period>.csv
#       One row per catchment per direction (ftd/dtf) with:
#       gauge_id, direction, total_precursor_events, superseded,
#       consumed, consumed_but_gap_excluded
#   pairing_diagnostics_output/pairing_supersession_summary.csv
#       Single pooled summary row per direction: the % superseded number
#       for the manuscript sentence.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
THRESHOLD_DIR    = Path("./thresholds")
OUTPUT_DIR       = Path("./pairing_diagnostics_output")

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]
ALL_COMBINATIONS = [(r, e) for r in RCPS for e in ENSEMBLES]

BASELINE_START, BASELINE_END = "1981-10-01", "2010-09-30"
FUTURE_START,   FUTURE_END   = "2050-10-01", "2080-09-30"

# Identical to extract_transitions.py / explore_drought_to_flood.py
FLOOD_MIN_DURATION = 1
FLOOD_POOL_GAP     = 5
DROUGHT_MIN_DUR    = 5
DROUGHT_POOL_GAP   = 5
DROUGHT_POOL_REC   = 0.9
MAX_TRANSITION_GAP = 90   # FTD gap window (Methods 2.2)
MAX_RECOVERY_GAP   = 90   # DTF gap window (Methods 2.2)


# ---------------------------------------------------------------------------
# Event extraction -- identical logic to extract_transitions.py /
# explore_drought_to_flood.py, copied here so this script has no import
# dependency on either file.
# ---------------------------------------------------------------------------

def _extract_flood_events(flow, q5):
    above = flow > q5
    events, in_event, start = [], False, 0
    for i in range(len(flow)):
        if above[i] and not in_event:
            start, in_event = i, True
        elif not above[i] and in_event:
            events.append([start, i - 1]); in_event = False
    if in_event:
        events.append([start, len(flow) - 1])
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
        if (e - s + 1) >= FLOOD_MIN_DURATION:
            result.append({"flood_start_idx": s, "flood_end_idx": e})
    return result


def _extract_drought_events(flow, q80):
    below = flow < q80
    events, in_event, start = [], False, 0
    for i in range(len(flow)):
        if below[i] and not in_event:
            start, in_event = i, True
        elif not below[i] and in_event:
            events.append([start, i - 1]); in_event = False
    if in_event:
        events.append([start, len(flow) - 1])
    if len(events) > 1:
        pooled = [events[0]]
        for ev in events[1:]:
            gap_start, gap_end = pooled[-1][1] + 1, ev[0] - 1
            gap_len = gap_end - gap_start + 1
            if gap_len <= DROUGHT_POOL_GAP:
                inter_max = float(np.max(flow[gap_start:gap_end + 1])) if gap_len > 0 else 0.0
                if inter_max < q80 * DROUGHT_POOL_REC:
                    pooled[-1][1] = ev[1]
                else:
                    pooled.append(ev)
            else:
                pooled.append(ev)
        events = pooled
    result = []
    for s, e in events:
        if (e - s + 1) >= DROUGHT_MIN_DUR:
            result.append({"drought_start_idx": s, "drought_end_idx": e})
    return result


# ---------------------------------------------------------------------------
# Instrumented pairing -- same rule as _find_transitions() /
# _find_drought_to_flood_transitions(), with counters added.
# ---------------------------------------------------------------------------

def _count_ftd_pairing(floods, droughts) -> dict:
    counts = {"total_precursor_events": 0, "superseded": 0,
              "consumed": 0, "consumed_but_gap_excluded": 0}
    if not floods or not droughts:
        counts["total_precursor_events"] = len(floods)
        return counts

    events = [(f["flood_end_idx"], 0, "flood", f) for f in floods]
    events += [(d["drought_start_idx"], 1, "drought", d) for d in droughts]
    events.sort(key=lambda e: (e[0], e[1]))

    pending_flood = None
    for _, _, kind, obj in events:
        if kind == "flood":
            counts["total_precursor_events"] += 1
            if pending_flood is not None:
                counts["superseded"] += 1
            pending_flood = obj
        else:
            if pending_flood is not None:
                gap_days = obj["drought_start_idx"] - pending_flood["flood_end_idx"] - 1
                counts["consumed"] += 1
                if gap_days > MAX_TRANSITION_GAP:
                    counts["consumed_but_gap_excluded"] += 1
            pending_flood = None
    return counts


def _count_dtf_pairing(droughts, floods) -> dict:
    counts = {"total_precursor_events": 0, "superseded": 0,
              "consumed": 0, "consumed_but_gap_excluded": 0}
    if not droughts or not floods:
        counts["total_precursor_events"] = len(droughts)
        return counts

    events = [(d["drought_end_idx"], 0, "drought", d) for d in droughts]
    events += [(f["flood_start_idx"], 1, "flood", f) for f in floods]
    events.sort(key=lambda e: (e[0], e[1]))

    pending_drought = None
    for _, _, kind, obj in events:
        if kind == "drought":
            counts["total_precursor_events"] += 1
            if pending_drought is not None:
                counts["superseded"] += 1
            pending_drought = obj
        else:
            if pending_drought is not None:
                gap_days = obj["flood_start_idx"] - pending_drought["drought_end_idx"] - 1
                counts["consumed"] += 1
                if gap_days > MAX_RECOVERY_GAP:
                    counts["consumed_but_gap_excluded"] += 1
            pending_drought = None
    return counts


# ---------------------------------------------------------------------------
# Period slicing -- identical approach to extract_transitions.py
# ---------------------------------------------------------------------------

def _extract_period(q_df, sm_df, start, end):
    date_only = q_df["date"].str.slice(0, 10)
    mask = (date_only >= start) & (date_only <= end)
    catchment_cols = [c for c in q_df.columns if c != "date"]
    return q_df.loc[mask, catchment_cols], sm_df.loc[mask, catchment_cols]


# ---------------------------------------------------------------------------
# One RCP/ensemble/period combination
# ---------------------------------------------------------------------------

def run_one_period(rcp: str, ens: str, period: str, thresholds: pd.DataFrame) -> Path:
    out_path = OUTPUT_DIR / f"diagnostics_{rcp}_{ens}_{period}.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists", flush=True)
        return out_path

    start = BASELINE_START if period == "baseline" else FUTURE_START
    end   = BASELINE_END   if period == "baseline" else FUTURE_END

    q_path  = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_discharge.csv"
    sm_path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_sm.csv"
    for p in (q_path, sm_path):
        if not p.exists():
            print(f"  ERROR: missing {p}", flush=True)
            sys.exit(1)

    print(f"  Loading CSVs for {rcp}/{ens}/{period} ...", flush=True)
    q_df  = pd.read_csv(q_path,  dtype={"date": str})
    sm_df = pd.read_csv(sm_path, dtype={"date": str})  # loaded for parity; not used in counts

    q_sl, _ = _extract_period(q_df, sm_df, start, end)
    catchment_ids = [c for c in q_df.columns if c != "date"]
    thresh_lookup = thresholds.set_index("gauge_id")

    rows = []
    n_done = 0
    for cid in catchment_ids:
        if cid not in thresh_lookup.index:
            continue
        q5  = float(thresh_lookup.loc[cid, "q5_median"])
        q80 = float(thresh_lookup.loc[cid, "q80_median"])
        flow = q_sl[cid].to_numpy(dtype=np.float64)

        floods   = _extract_flood_events(flow, q5)
        droughts = _extract_drought_events(flow, q80)

        ftd = _count_ftd_pairing(floods, droughts)
        dtf = _count_dtf_pairing(droughts, floods)

        rows.append({"gauge_id": cid, "direction": "ftd", **ftd})
        rows.append({"gauge_id": cid, "direction": "dtf", **dtf})

        n_done += 1
        if n_done % 100 == 0:
            print(f"    ...{n_done}/{len(catchment_ids)} catchments done", flush=True)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Written: {out_path}", flush=True)
    return out_path


def run_one_combination(rcp: str, ens: str, thresholds: pd.DataFrame) -> None:
    print(f"\n{'='*55}\n  {rcp} / {ens}\n{'='*55}", flush=True)
    for period in ["baseline", "future"]:
        run_one_period(rcp, ens, period, thresholds)


# ---------------------------------------------------------------------------
# Summarise across every diagnostics_*.csv written so far
# ---------------------------------------------------------------------------

def summarise() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("diagnostics_*.csv"))
    if not files:
        print(f"No diagnostics_*.csv files found in {OUTPUT_DIR}/ -- "
              f"run this script with --all or --rcp/--ensemble first.")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rows = []
    for direction in ["ftd", "dtf"]:
        sub = df[df["direction"] == direction]
        total = sub["total_precursor_events"].sum()
        superseded = sub["superseded"].sum()
        consumed = sub["consumed"].sum()
        gap_excluded = sub["consumed_but_gap_excluded"].sum()
        rows.append({
            "direction": direction,
            "total_precursor_events": total,
            "superseded": superseded,
            "pct_superseded": 100 * superseded / total if total else np.nan,
            "consumed": consumed,
            "pct_consumed": 100 * consumed / total if total else np.nan,
            "consumed_but_gap_excluded": gap_excluded,
            "pct_gap_excluded": 100 * gap_excluded / total if total else np.nan,
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("PAIRING SUPERSESSION SUMMARY (pooled across all catchments/RCPs/ensembles/periods)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    out_path = OUTPUT_DIR / "pairing_supersession_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantify how often the FTD/DTF pairing rule supersedes vs consumes precursor events."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all 16 RCP/ensemble combinations.")
    group.add_argument("--rcp", choices=RCPS, help="Single RCP (requires --ensemble).")
    group.add_argument("--task-id", type=int, metavar="N",
                        help="Slurm array task ID (0-15), same convention as extract_transitions.py.")
    group.add_argument("--summarise-only", action="store_true",
                        help="Skip processing; just pool existing diagnostics_*.csv files into the final summary.")
    parser.add_argument("--ensemble", choices=ENSEMBLES, help="Ensemble member (required with --rcp).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.summarise_only:
        summarise()
        return

    threshold_path = THRESHOLD_DIR / "thresholds_ensemble_median.parquet"
    if not threshold_path.exists():
        print(f"ERROR: threshold file not found: {threshold_path}")
        sys.exit(1)
    thresholds = pd.read_parquet(threshold_path)
    print(f"Loaded thresholds for {len(thresholds)} catchments", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        for rcp, ens in ALL_COMBINATIONS:
            run_one_combination(rcp, ens, thresholds)
    elif args.task_id is not None:
        if not (0 <= args.task_id < len(ALL_COMBINATIONS)):
            print(f"ERROR: --task-id must be 0-{len(ALL_COMBINATIONS)-1}")
            sys.exit(1)
        rcp, ens = ALL_COMBINATIONS[args.task_id]
        run_one_combination(rcp, ens, thresholds)
    else:
        if not args.ensemble:
            print("ERROR: --ensemble is required with --rcp")
            sys.exit(1)
        run_one_combination(args.rcp, args.ensemble, thresholds)

    print("\nAll done. Once every combination has been run, call:")
    print("  python quantify_pairing_supersession.py --summarise-only")


if __name__ == "__main__":
    main()
