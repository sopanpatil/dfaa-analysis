# validate_snowpack_fracsnow.py
# --------------------------------
# Snow-validation figure cited in Results 3.4 / Discussion: correlates
# HBV-simulated BASELINE mean snowpack (SP state, ensemble-median across
# the 4 members) against the CAMELS-GB v2 static frac_snow attribute --
# an independently derived quantity (observed temperature + precipitation),
# providing corroborating evidence that HBV's snow states are physically
# credible rather than an artefact of discharge-only calibration.
#
# This is DELIBERATELY run on the full available catchment set (not the
# KGE-filtered or delta-metric-valid subsamples used elsewhere), since it
# is a validation of the model's internal snow state itself, not of the
# downstream FTD/DTF transition metrics -- the two questions ("is the
# calibrated model good enough to use for transitions?" vs "is the snow
# state physically credible?") use different, appropriately-scoped
# samples. State this explicitly in Methods 2.7.
#
# Requires:
#   chess_scape_output/<rcp>_<ens>_hbv_sp.csv   (any single RCP suffices;
#       baseline period is identical forcing/model run across RCPs, so
#       use rcp85 or whichever was run first -- do NOT average across RCPs,
#       that would pseudo-replicate the same baseline signal 4x)
#   camels_gb_v2_climatic_attributes.csv        (frac_snow)
#
# Usage
# -----
#   python validate_snowpack_fracsnow.py --rcp rcp85

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHESS_SCAPE_ROOT = Path("./chess_scape_output")
CLIMATIC_ATTR_FILE = Path("./camels_gb_v2_climatic_attributes.csv")
OUTPUT_DIR = Path("./spatial_output")

ENSEMBLES = ["01", "04", "06", "15"]
BASELINE_START, BASELINE_END = "1981-10-01", "2010-09-30"


def load_baseline_sp(rcp: str) -> pd.DataFrame:
    """Ensemble-median baseline mean/peak snowpack per catchment."""
    frames = []
    for ens in ENSEMBLES:
        path = CHESS_SCAPE_ROOT / f"{rcp}_{ens}_hbv_sp.csv"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        df = pd.read_csv(path, dtype={"date": str})
        date_only = df["date"].str.slice(0, 10)
        mask = (date_only >= BASELINE_START) & (date_only <= BASELINE_END)
        sub = df.loc[mask].drop(columns=["date"])
        rows = []
        for cid in sub.columns:
            vals = sub[cid].to_numpy(dtype=np.float64)
            rows.append({"gauge_id": int(cid), "mean_sp": float(np.nanmean(vals)),
                         "max_sp": float(np.nanmax(vals)),
                         "frac_days_with_snow": float(np.mean(vals > 0))})
        frames.append(pd.DataFrame(rows).set_index("gauge_id"))
    if not frames:
        return pd.DataFrame()
    stacked_mean = np.stack([f["mean_sp"].reindex(frames[0].index).values for f in frames], axis=1)
    stacked_max = np.stack([f["max_sp"].reindex(frames[0].index).values for f in frames], axis=1)
    stacked_frac = np.stack([f["frac_days_with_snow"].reindex(frames[0].index).values for f in frames], axis=1)
    out = pd.DataFrame(index=frames[0].index)
    out["mean_sp"] = np.nanmedian(stacked_mean, axis=1)
    out["max_sp"] = np.nanmedian(stacked_max, axis=1)
    out["frac_days_with_snow"] = np.nanmedian(stacked_frac, axis=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rcp", default="rcp85")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline simulated snowpack ({args.rcp})...")
    sp = load_baseline_sp(args.rcp)
    print(f"  {len(sp)} catchments with simulated snow states")

    print("Loading frac_snow (independent, observation-derived)...")
    frac_snow = pd.read_csv(CLIMATIC_ATTR_FILE, index_col="gauge_id")["frac_snow"]
    print(f"  {len(frac_snow)} catchments in CAMELS-GB v2 climatic attributes")

    common = sp.index.intersection(frac_snow.index)
    print(f"  {len(common)} catchments common to both -- THIS is the N to report")

    results = []
    for metric in ["mean_sp", "max_sp", "frac_days_with_snow"]:
        x = sp.loc[common, metric].values
        y = frac_snow.loc[common].values
        valid = ~(np.isnan(x) | np.isnan(y))
        rho, pval = stats.spearmanr(x[valid], y[valid])
        results.append({"metric": metric, "rho": rho, "pvalue": pval, "n": int(valid.sum())})
        print(f"  {metric} vs frac_snow: rho={rho:.3f}, p={pval:.3e}, n={valid.sum()}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "snowpack_fracsnow_validation.csv", index=False)
    print(f"\nWritten: {OUTPUT_DIR / 'snowpack_fracsnow_validation.csv'}")


if __name__ == "__main__":
    main()
