"""
run_hbv_chess_scape.py
-----------------------
Run the calibrated HBV model (from the separate hbv-model repository/
package) across CHESS-SCAPE forcing data for a given RCP and ensemble
member, saving simulated discharge and the internal SM, UZ, LZ states
needed for this paper's soil-moisture-based FTD/DTF metrics.

Calibrated parameters are read from a single consolidated CSV
(calibrated_parameters.csv), produced alongside the HBV model itself
in the hbv-model repository.

Requires the hbv_model package to be installed or importable
(pip install, or on the Python path) -- see hbv-model repository.

Output files:
    <DATA_DIR>/<rcp>_<ensemble>_hbv_discharge.csv
    <DATA_DIR>/<rcp>_<ensemble>_hbv_sm.csv
    <DATA_DIR>/<rcp>_<ensemble>_hbv_uz.csv
    <DATA_DIR>/<rcp>_<ensemble>_hbv_lz.csv
    <DATA_DIR>/<rcp>_<ensemble>_hbv_sp.csv   (snowpack, needed for Section 2.7 snow validation)

Usage
-----
python run_hbv_chess_scape.py --rcp rcp26 --ensemble 01 \\
    --params-csv /path/to/hbv-model/calibrated_parameters.csv
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hbv_model.hbv import HBVModel

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
N_WORKERS = 8


def _run_catchment(cid, params):
    try:
        model = HBVModel(params)
        Q_routed, states_full = model.run(
            _run_catchment.precip[cid], _run_catchment.temp[cid], _run_catchment.evap[cid]
        )
        states_out = {
            "SM": states_full["SM"], "UZ": states_full["UZ"],
            "LZ": states_full["LZ"], "SP": states_full["SP"],
        }
        return cid, Q_routed, states_out
    except Exception as exc:
        print(f"  ERROR: HBV / {cid}: {exc}", flush=True)
        return cid, None, None


def _init_worker(precip, temp, evap):
    _run_catchment.precip = precip
    _run_catchment.temp = temp
    _run_catchment.evap = evap


def load_params_csv(params_csv_path):
    """Load hbv-model's consolidated calibrated_parameters.csv into a
    dict of {gauge_id (str): param dict}, using only the 13 HBV
    parameter columns (ignoring calibration_kge/validation_kge/
    used_in_analysis, which are metadata, not model inputs)."""
    df = pd.read_csv(params_csv_path, dtype={"gauge_id": str})
    param_cols = ['TT', 'CFMAX', 'CFR', 'CWH', 'FC', 'LP', 'BETA',
                  'K0', 'K1', 'K2', 'UZL', 'PERC', 'MAXBAS']
    return {row["gauge_id"]: {p: row[p] for p in param_cols} for _, row in df.iterrows()}


def load_forcing(data_dir, rcp, ensemble):
    def _read(var):
        path = data_dir / f"{rcp}_{ensemble}_{var}_catchment_means_combined.csv"
        return pd.read_csv(path)
    pr_df, tas_df, pet_df = _read("pr"), _read("tas"), _read("pet")
    dates = pr_df["date"]
    for name, df in [("tas", tas_df), ("pet", pet_df)]:
        if not df["date"].equals(dates):
            raise ValueError(f"Date mismatch between pr and {name}")
        if list(df.columns) != list(pr_df.columns):
            raise ValueError(f"Column mismatch between pr and {name}")
    return pr_df, tas_df, pet_df


def run_hbv(rcp, ensemble, pr_df, tas_df, pet_df, params_by_catchment, output_dir, n_workers=N_WORKERS):
    dates = pr_df["date"]
    catchment_ids = [c for c in pr_df.columns if c != "date"]

    q_file = output_dir / f"{rcp}_{ensemble}_hbv_discharge.csv"
    sm_file = output_dir / f"{rcp}_{ensemble}_hbv_sm.csv"
    uz_file = output_dir / f"{rcp}_{ensemble}_hbv_uz.csv"
    lz_file = output_dir / f"{rcp}_{ensemble}_hbv_lz.csv"
    sp_file = output_dir / f"{rcp}_{ensemble}_hbv_sp.csv"

    if all(f.exists() for f in [q_file, sm_file, uz_file, lz_file, sp_file]):
        print(f"  [skip] all HBV outputs already exist for {rcp}/{ensemble}", flush=True)
        return

    print(f"\n  Model: HBV | {len(catchment_ids)} catchments", flush=True)

    precip = {cid: pr_df[cid].to_numpy() for cid in catchment_ids}
    temp = {cid: tas_df[cid].to_numpy() for cid in catchment_ids}
    evap = {cid: pet_df[cid].to_numpy() for cid in catchment_ids}

    work = []
    skipped_params = []
    for cid in catchment_ids:
        params = params_by_catchment.get(cid)
        if params is None:
            skipped_params.append(cid)
            continue
        work.append((cid, params))

    if skipped_params:
        print(f"  WARNING: no parameters for {len(skipped_params)} catchments, skipping", flush=True)

    q_results, sm_results, uz_results, lz_results, sp_results = {}, {}, {}, {}, {}
    failed = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=(precip, temp, evap)) as pool:
        futures = {pool.submit(_run_catchment, cid, params): cid for cid, params in work}
        for future in as_completed(futures):
            cid, Q_routed, states_out = future.result()
            if Q_routed is not None:
                q_results[cid] = Q_routed
                sm_results[cid] = states_out["SM"]
                uz_results[cid] = states_out["UZ"]
                lz_results[cid] = states_out["LZ"]
                sp_results[cid] = states_out["SP"]
            else:
                failed.append(cid)
            completed += 1
            if completed % 100 == 0 or completed == len(work):
                print(f"    ...{completed}/{len(work)} done", flush=True)

    q_out = {"date": dates}
    for cid in catchment_ids:
        if cid in q_results:
            q_out[cid] = q_results[cid]
    q_df = pd.DataFrame(q_out)
    q_df.to_csv(q_file, index=False, float_format="%.4f")
    sim_cols = [c for c in q_df.columns if c != "date"]
    print(f"  Written: {q_file.name}  shape={q_df.shape}", flush=True)
    if sim_cols:
        print(f"  Q range: {q_df[sim_cols].min().min():.3f} -- "
              f"{q_df[sim_cols].max().max():.3f} mm/day", flush=True)

    for state_name, results, out_path in [
        ("SM", sm_results, sm_file), ("UZ", uz_results, uz_file),
        ("LZ", lz_results, lz_file), ("SP", sp_results, sp_file),
    ]:
        state_out = {"date": dates}
        for cid in catchment_ids:
            if cid in results:
                state_out[cid] = results[cid]
        state_df = pd.DataFrame(state_out)
        state_df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"  Written: {out_path.name}  shape={state_df.shape}", flush=True)

    if failed:
        print(f"  FAILED (runtime error): {len(failed)} catchments: {failed}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibrated HBV on CHESS-SCAPE forcing data.")
    parser.add_argument("--rcp", required=True, choices=["rcp26", "rcp45", "rcp60", "rcp85"])
    parser.add_argument("--ensemble", required=True, help="Ensemble member, e.g. 01 or 15.")
    parser.add_argument("--params-csv", required=True,
                        help="Path to hbv-model's calibrated_parameters.csv")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = CHESS_SCAPE_ROOT

    print(f"{'='*60}\nRCP: {args.rcp}  |  Ensemble: {args.ensemble}\n{'='*60}")

    print("Loading calibrated parameters...", flush=True)
    params_by_catchment = load_params_csv(args.params_csv)
    print(f"  {len(params_by_catchment)} catchments with parameters", flush=True)

    print("Loading forcing data...", flush=True)
    try:
        pr_df, tas_df, pet_df = load_forcing(data_dir, args.rcp, args.ensemble)
    except FileNotFoundError as e:
        print(f"ERROR: forcing file not found: {e}")
        sys.exit(1)

    n_dates = len(pr_df["date"])
    n_catch = len([c for c in pr_df.columns if c != "date"])
    print(f"  {n_dates} timesteps, {n_catch} catchments", flush=True)

    run_hbv(args.rcp, args.ensemble, pr_df, tas_df, pet_df,
            params_by_catchment, data_dir, n_workers=args.workers)

    print(f"\nAll done for {args.rcp} / {args.ensemble}.")


if __name__ == "__main__":
    main()
