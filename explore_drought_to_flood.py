# explore_drought_to_flood.py
# ----------------------------
# Drought-to-flood (DTF) transition extraction, mirroring the
# flood-to-drought (FTD) pipeline in extract_transitions.py.
#
# Reuses the SAME flood and drought event extraction logic (Q5/Q80,
# 5-day flood pooling, no drought pooling) from
# extract_transitions.py -- only the PAIRING DIRECTION and the METRICS
# computed at the transition are different.
#
# Key DTF-specific metrics (mirroring but not identical to FTD):
#   recovery_gap_days   : days from drought end to flood onset
#   sm_at_drought_end    : SM store value at drought recession (Q80 crossing)
#   sm_at_flood_onset     : SM store value when flow first exceeds Q5
#   sm_recovery_rate      : (sm_at_flood_onset - sm_at_drought_end) / gap
#
# ---------------------------------------------------------------------------
# TWO-STAGE DESIGN
# ---------------------------------------------------------------------------
# MAX_RECOVERY_GAP only affects which drought/flood events get PAIRED -- it
# has no bearing on event DETECTION (_extract_flood_events /
# _extract_drought_events never reference it). So event extraction is
# window-independent and only needs to run once per (rcp, ensemble, period);
# pairing is cheap and can be re-run per window value without re-touching
# the raw discharge/SM series. This separation is what makes the
# --max-gap sensitivity check (see Usage below) practical to run across
# multiple candidate window widths.
#
#   Stage "events"  : read CSVs, run flood/drought extraction per catchment,
#                      cache to events_cache/events_{rcp}_{ens}_{period}.pkl
#   Stage "pairing" : load cached events, pair drought->flood for a given
#                      max_gap, write d2f_{rcp}_{ens}_{period}_w{gap:03d}.parquet
#   Stage "all"     : run both stages back to back (default)
#
# The 90-day max gap used throughout the paper (Methods 2.2) is the
# --max-gap default below; other widths can be tested via --sensitivity
# for robustness checking, but are not used in any reported result.
#
# Usage
# -----
#   # Single combination, default window (90d), both stages
#   python explore_drought_to_flood.py --rcp rcp85 --ensemble 01
#
#   # Cache events only (e.g. as a JASMIN array job over 16 rcp/ens combos)
#   python explore_drought_to_flood.py --stage events --rcp rcp85 --ensemble 01
#
#   # Pairing only, for one window, using already-cached events
#   python explore_drought_to_flood.py --stage pairing --rcp rcp85 --ensemble 01 --max-gap 60
#
#   # Full 16-combination robustness check at the default window
#   python explore_drought_to_flood.py --all
#
#   # Sensitivity sweep across window values, all 16 combinations
#   python explore_drought_to_flood.py --all --max-gap 30 60 90 120 180

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
THRESHOLD_DIR    = Path("./thresholds")
OUTPUT_DIR       = Path("./drought_to_flood_exploratory")
EVENTS_DIR       = OUTPUT_DIR / "events_cache"

BASELINE_START = "1981-10-01"
BASELINE_END   = "2010-09-30"
FUTURE_START   = "2050-10-01"
FUTURE_END     = "2080-09-30"

FLOOD_POOL_GAP   = 5
DROUGHT_MIN_DUR  = 5    # NOTE: shorter than Anderson et al. (2025)'s 30-day
                          # minimum. Sole independence criterion -- drought
                          # spells are not pooled (Methods 2.2).
                          # Matches the FTD pipeline's choice (extract_transitions.py)
                          # for direct cross-direction comparability. Held fixed
                          # across the sensitivity sweep below -- only
                          # MAX_RECOVERY_GAP is varied.

DEFAULT_MAX_RECOVERY_GAP = 90    # matches Götte & Brunner (2024) and
                                    # Anderson et al. (2025) precedent
SENSITIVITY_WINDOWS = [30, 60, 90, 120, 180]

RCPS      = ["rcp26", "rcp45", "rcp60", "rcp85"]
ENSEMBLES = ["01", "04", "06", "15"]


# ---------------------------------------------------------------------------
# Event extraction (identical logic to extract_transitions.py)
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
        seg = flow[s:e + 1]
        peak_idx = s + int(np.argmax(seg))
        result.append({"flood_start_idx": s, "flood_end_idx": e,
                       "flood_peak_idx": peak_idx, "flood_peak_flow": float(flow[peak_idx])})
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
    # No inter-event pooling -- see extract_transitions.py / Methods 2.2.
    result = []
    for s, e in events:
        dur = e - s + 1
        if dur >= DROUGHT_MIN_DUR:
            seg = flow[s:e + 1]
            min_idx = s + int(np.argmin(seg))
            result.append({"drought_start_idx": s, "drought_end_idx": e,
                           "drought_min_idx": min_idx, "drought_min_flow": float(flow[min_idx]),
                           "drought_duration": dur})
    return result


# ---------------------------------------------------------------------------
# REVERSED pairing: drought -> next flood (mirror of flood -> next drought)
# max_gap is a parameter (not a module constant) so the same cached events
# can be re-paired at different window values.
# ---------------------------------------------------------------------------

def _find_drought_to_flood_transitions(droughts, floods, max_gap):
    """
    One-to-one, temporally-ordered pairing (mirror image of
    find_transitions() in extract_transitions.py). Processes drought-end
    and flood-start events in a single chronological pass. Each new
    drought supersedes any earlier unresolved drought (so a drought
    immediately followed by another drought, before any flood occurs, is
    not paired with a later, non-adjacent flood). Each flood consumes
    whichever drought is currently pending -- if any -- and that drought
    is then unavailable to any other flood. This guarantees every drought
    and every flood participates in at most one transition, and that each
    transition represents an uninterrupted drought-to-flood episode with
    no other drought or flood event intervening.
    """
    if not droughts or not floods:
        return []

    events = []
    for d in droughts:
        events.append((d["drought_end_idx"], 0, "drought", d))
    for f in floods:
        events.append((f["flood_start_idx"], 1, "flood", f))
    events.sort(key=lambda e: (e[0], e[1]))

    transitions = []
    pending_drought = None
    for _, _, kind, obj in events:
        if kind == "drought":
            pending_drought = obj
        else:
            if pending_drought is not None:
                gap_days = obj["flood_start_idx"] - pending_drought["drought_end_idx"] - 1
                if gap_days <= max_gap:
                    row = {}
                    row.update(pending_drought)
                    row.update(obj)
                    row["recovery_gap_days"] = gap_days
                    transitions.append(row)
            pending_drought = None
    return transitions


def _attach_recovery_states(transitions, sm, sp):
    """
    Mirror of attach_states() but for recovery direction:
      sm_at_drought_end : SM when drought recedes (Q80 crossing, exit)
      sm_at_flood_onset : SM when flow first crosses above Q5
      sm_recovery_rate  : how fast storage rebuilds during the gap
      sp_at_drought_end : HBV SP (snowpack) at drought recession
      sp_at_flood_onset : HBV SP (snowpack) when flow first crosses above Q5
    """
    for t in transitions:
        de = t["drought_end_idx"]
        fo = t["flood_start_idx"]
        gap = t["recovery_gap_days"]

        t["sm_at_drought_end"] = float(sm[de])
        t["sm_at_flood_onset"] = float(sm[fo])
        t["sp_at_drought_end"] = float(sp[de])
        t["sp_at_flood_onset"] = float(sp[fo])

        if gap > 0:
            t["sm_recovery_rate"] = (float(sm[fo]) - float(sm[de])) / gap
        else:
            t["sm_recovery_rate"] = np.nan
    return transitions


# ---------------------------------------------------------------------------
# Period extraction helper (same logic as extract_transitions.py)
# ---------------------------------------------------------------------------

def _extract_period(q_df, sm_df, sp_df, start, end):
    # NOTE: raw date strings carry a " 12:00:00" time suffix (e.g.
    # "2010-09-30 12:00:00"), so a plain string comparison against a
    # bare "YYYY-MM-DD" end date silently drops the final day of every
    # period ("2010-09-30 12:00:00" <= "2010-09-30" is False, since the
    # longer string with an identical prefix sorts greater). Truncating
    # to the date portion before comparing avoids that off-by-one.
    date_only = q_df["date"].str.slice(0, 10)
    mask = (date_only >= start) & (date_only <= end)
    dates = q_df.loc[mask, "date"].to_numpy(dtype=str)
    catchment_cols = [c for c in q_df.columns if c != "date"]
    return (dates, q_df.loc[mask, catchment_cols], sm_df.loc[mask, catchment_cols],
            sp_df.loc[mask, catchment_cols])


# ---------------------------------------------------------------------------
# Stage 1: event extraction (window-independent, cached)
# ---------------------------------------------------------------------------

def _events_cache_path(rcp: str, ens: str, period: str) -> Path:
    return EVENTS_DIR / f"events_{rcp}_{ens}_{period}.pkl"


def extract_events_for_combination(rcp: str, ens: str, period: str,
                                    thresholds: pd.DataFrame, force: bool = False) -> dict:
    """
    Run flood/drought event extraction for every catchment in one
    (rcp, ensemble, period) combination and cache the result.

    Returns dict: catchment_id -> {"floods": [...], "droughts": [...], "sm": np.ndarray}
    sm is cached alongside the events because _attach_recovery_states needs
    it at pairing time, indexed by the same absolute position within the
    period slice.
    """
    cache_path = _events_cache_path(rcp, ens, period)
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    start = BASELINE_START if period == "baseline" else FUTURE_START
    end   = BASELINE_END   if period == "baseline" else FUTURE_END

    q_path  = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_discharge.csv"
    sm_path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_sm.csv"
    sp_path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_sp.csv"

    q_df  = pd.read_csv(q_path,  dtype={"date": str})
    sm_df = pd.read_csv(sm_path, dtype={"date": str})
    sp_df = pd.read_csv(sp_path, dtype={"date": str})

    _, q_sl, sm_sl, sp_sl = _extract_period(q_df, sm_df, sp_df, start, end)
    catchment_ids = [c for c in q_df.columns if c != "date"]
    thresh_lookup = thresholds.set_index("gauge_id")

    events_by_catchment = {}
    for cid in catchment_ids:
        if cid not in thresh_lookup.index:
            continue
        q5 = float(thresh_lookup.loc[cid, "q5_median"])
        q80 = float(thresh_lookup.loc[cid, "q80_median"])

        flow = q_sl[cid].to_numpy(dtype=np.float64)
        sm   = sm_sl[cid].to_numpy(dtype=np.float64)
        sp   = sp_sl[cid].to_numpy(dtype=np.float64)

        floods   = _extract_flood_events(flow, q5)
        droughts = _extract_drought_events(flow, q80)

        events_by_catchment[cid] = {"floods": floods, "droughts": droughts, "sm": sm, "sp": sp}

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(events_by_catchment, f)

    return events_by_catchment


# ---------------------------------------------------------------------------
# Stage 2: pairing (window-dependent, cheap -- reuses cached events)
# ---------------------------------------------------------------------------

def pair_transitions(events_by_catchment: dict, max_gap: int) -> pd.DataFrame:
    all_rows = []
    for cid, ev in events_by_catchment.items():
        transitions = _find_drought_to_flood_transitions(ev["droughts"], ev["floods"], max_gap)
        transitions = _attach_recovery_states(transitions, ev["sm"], ev["sp"])
        for t in transitions:
            t["gauge_id"] = cid
        all_rows.extend(transitions)
    return pd.DataFrame(all_rows)


def _pairing_output_path(rcp: str, ens: str, period: str, max_gap: int) -> Path:
    return OUTPUT_DIR / f"d2f_{rcp}_{ens}_{period}_w{max_gap:03d}.parquet"


def run_combination(rcp: str, ens: str, thresholds: pd.DataFrame,
                     max_gaps: list, stage: str, force: bool = False) -> list:
    """
    Run stage(s) for one (rcp, ensemble) across both periods and (for the
    pairing stage) all requested max_gap values.

    Returns a list of summary dicts, one per (period, max_gap) combination
    actually paired (empty if stage == "events").
    """
    summaries = []
    events_by_period = {}

    if stage in ("events", "all"):
        for period in ["baseline", "future"]:
            print(f"  [events]  {rcp}/{ens}/{period} ...", flush=True)
            events_by_period[period] = extract_events_for_combination(
                rcp, ens, period, thresholds, force=force)

    if stage in ("pairing", "all"):
        for period in ["baseline", "future"]:
            if period not in events_by_period:
                events_by_period[period] = extract_events_for_combination(
                    rcp, ens, period, thresholds, force=False)

        for max_gap in max_gaps:
            summary = {"rcp": rcp, "ensemble": ens, "max_gap": max_gap}
            for period in ["baseline", "future"]:
                out = _pairing_output_path(rcp, ens, period, max_gap)
                if out.exists() and not force:
                    df = pd.read_parquet(out)
                else:
                    df = pair_transitions(events_by_period[period], max_gap)
                    df.to_parquet(out, index=False)

                summary[f"{period}_n_transitions"] = len(df)
                summary[f"{period}_n_catchments"]  = df["gauge_id"].nunique() if len(df) else 0
                summary[f"{period}_gap_median"]    = df["recovery_gap_days"].median() if len(df) else np.nan
                summary[f"{period}_recovery_rate_median"] = df["sm_recovery_rate"].median() if len(df) else np.nan

            summary["delta_gap"] = summary["future_gap_median"] - summary["baseline_gap_median"]
            summary["delta_recovery_rate"] = (
                summary["future_recovery_rate_median"] - summary["baseline_recovery_rate_median"]
            )
            summaries.append(summary)
            print(f"  [pairing] {rcp}/{ens} w={max_gap:3d}d: "
                  f"baseline n={summary['baseline_n_transitions']}  "
                  f"future n={summary['future_n_transitions']}  "
                  f"delta_gap={summary['delta_gap']:+.1f}d", flush=True)

    return summaries


def parse_args():
    p = argparse.ArgumentParser(description="Drought-to-flood transition analysis (events + pairing stages).")
    p.add_argument("--rcp", default=None, help="Single RCP, e.g. rcp85. Omit with --all.")
    p.add_argument("--ensemble", default=None, help="Single ensemble member, e.g. 01. Omit with --all.")
    p.add_argument("--all", action="store_true",
                   help="Run all 16 RCP/ensemble combinations.")
    p.add_argument("--stage", choices=["events", "pairing", "all"], default="all",
                   help="Which stage to run. 'events' caches event extraction only "
                        "(window-independent); 'pairing' re-pairs from cached events "
                        "for the given --max-gap value(s); 'all' does both.")
    p.add_argument("--max-gap", type=int, nargs="+", default=[DEFAULT_MAX_RECOVERY_GAP],
                   help="Max recovery gap (days) for pairing. Accepts multiple values "
                        "for a sensitivity sweep, e.g. --max-gap 30 60 90 120 180")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cached/output files exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be run without touching any files.")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = [(r, e) for r in RCPS for e in ENSEMBLES] if args.all else [(args.rcp or "rcp85", args.ensemble or "01")]

    if args.dry_run:
        print("DRY RUN -- no files will be written.\n")
        print(f"Stage:        {args.stage}")
        print(f"Max gap(s):   {args.max_gap}")
        print(f"Combinations: {len(combos)}")
        for rcp, ens in combos:
            print(f"  {rcp}/{ens}")
        return

    threshold_path = THRESHOLD_DIR / "thresholds_ensemble_median.parquet"
    thresholds = pd.read_parquet(threshold_path)

    all_summaries = []
    for rcp, ens in combos:
        print(f"\n{rcp}/{ens}", flush=True)
        all_summaries.extend(
            run_combination(rcp, ens, thresholds, args.max_gap, args.stage, force=args.force)
        )

    if args.stage in ("pairing", "all") and all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        tag = "sensitivity" if len(args.max_gap) > 1 else f"w{args.max_gap[0]:03d}"
        summary_path = OUTPUT_DIR / f"d2f_summary_{tag}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nWritten: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
