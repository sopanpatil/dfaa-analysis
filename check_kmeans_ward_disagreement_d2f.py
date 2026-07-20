# check_kmeans_ward_disagreement_d2f.py
# -----------------------------------------
# d2f equivalent of check_kmeans_ward_disagreement.py. Identical method,
# pointed at d2f_typology_output and d2f's cluster variable names.
#
# d2f's typology uses k=2 (silhouette-selected, matching f2d's own
# preference -- see catchment_typology_d2f.py / catchment_typology.py).
# ARI=0.460 (k-means vs Ward) -- below the 0.7 "good agreement" bar,
# though somewhat higher than f2d's own result (ARI=0.409). This
# identifies WHICH catchments the two methods disagree on, and checks
# whether disagreement is driven by extreme-value outliers or
# borderline/ambiguous cases.
#
# Requires d2f_typology_output/cluster_assignments.parquet
# (run catchment_typology_d2f.py first).
#
# Usage
# -----
#   python check_kmeans_ward_disagreement_d2f.py

from pathlib import Path

import numpy as np
import pandas as pd

TYPOLOGY_DIR = Path("./d2f_typology_output")
CLUSTER_VARS = ["delta_gap", "delta_sm_drought_end", "delta_freq", "delta_recovery_rate"]


def main():
    path = TYPOLOGY_DIR / "cluster_assignments.parquet"
    df = pd.read_parquet(path)
    df.index.name = "gauge_id"

    n = len(df)
    agree_mask = df["cluster_kmeans"] == df["cluster_ward"]
    n_agree = int(agree_mask.sum())
    n_disagree = n - n_agree

    print("=" * 90)
    print(f"D2F K-MEANS vs WARD AGREEMENT  (n={n} catchments)")
    print("=" * 90)
    print(f"  Agree:    {n_agree}  ({100*n_agree/n:.1f}%)")
    print(f"  Disagree: {n_disagree}  ({100*n_disagree/n:.1f}%)")

    X = df[CLUSTER_VARS].to_numpy(dtype=np.float64)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    extremity = np.sum(z ** 2, axis=1)
    df = df.copy()
    df["extremity"] = extremity

    print(f"\n{'-'*90}")
    print("EXTREMITY COMPARISON (sum of squared z-scores across the 4 cluster vars)")
    print(f"{'-'*90}")
    print(f"  Agreeing catchments:    mean extremity = {df.loc[agree_mask, 'extremity'].mean():.3f}  "
          f"median = {df.loc[agree_mask, 'extremity'].median():.3f}")
    print(f"  Disagreeing catchments: mean extremity = {df.loc[~agree_mask, 'extremity'].mean():.3f}  "
          f"median = {df.loc[~agree_mask, 'extremity'].median():.3f}")

    ratio = df.loc[~agree_mask, "extremity"].mean() / df.loc[agree_mask, "extremity"].mean()
    print(f"\n  Disagreeing/agreeing mean extremity ratio: {ratio:.2f}x")
    if ratio > 1.3:
        print("  -> Disagreeing catchments are notably MORE extreme than agreeing ones.")
        print("     Consistent with extreme-tail catchments driving method disagreement")
        print("     (same pattern found for f2d's typology).")
    elif ratio < 0.77:
        print("  -> Disagreeing catchments are notably LESS extreme (closer to the boundary")
        print("     between clusters) -- consistent with genuine ambiguous/borderline cases")
        print("     rather than outlier-driven instability.")
    else:
        print("  -> No strong difference -- disagreement doesn't map cleanly onto extremity.")

    print(f"\n{'-'*90}")
    print(f"ALL {n_disagree} DISAGREEING CATCHMENTS (sorted by extremity, most extreme first)")
    print(f"{'-'*90}")
    disagree_df = df.loc[~agree_mask, ["cluster_kmeans", "cluster_ward", "extremity"] + CLUSTER_VARS]
    disagree_df = disagree_df.sort_values("extremity", ascending=False)
    print(disagree_df.to_string())

    out_path = TYPOLOGY_DIR / "kmeans_ward_disagreement_d2f.csv"
    disagree_df.to_csv(out_path)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
