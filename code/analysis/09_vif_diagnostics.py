#!/usr/bin/env python3
"""
09_vif_diagnostics.py
========================
Variance Inflation Factors for the continuous main-model predictors (the
non-sector, non-reference numeric/binary regressors of spec A in
07_main_model.py), matching csv_exports/Analysis_Ready/vif_table.csv.

O'Brien (2007), cited in Appendix B of the dissertation, cautions against
treating VIF thresholds as decision rules; these are reported purely as
diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, ref  # noqa: E402

PREDICTORS = [
    "years_since_first_investment",
    "flag_founding_pre2000",
    "first_round_syndicate_size",
    "first_round_foreign_investor_share",
    "first_round_corporate_vc_flag",
    "first_round_late_stage_pe_flag",
    "log_investor_experience_mean",
]


def main() -> None:
    sample_path = Path(__file__).resolve().parent / "outputs" / "model_sample_main_with_ids.csv"
    if not sample_path.exists():
        raise SystemExit("Run 06_sample_construction.py first")
    df = pd.read_csv(sample_path)

    X = sm.add_constant(df[PREDICTORS].astype(float))
    vif = pd.DataFrame(
        {
            "variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    ).set_index("variable")

    out = outputs_dir()
    vif.to_csv(out / "vif_table.csv")
    print(f"Wrote {out / 'vif_table.csv'}")
    print(vif)

    gt_path = ref("Analysis_Ready", "vif_table.csv")
    if gt_path.exists():
        gt = pd.read_csv(gt_path, index_col=0)
        common = vif.index.intersection(gt.index)
        print("\n=== VALIDATION vs csv_exports/Analysis_Ready/vif_table.csv ===")
        for v in common:
            print(f"  {v}: reconstructed={vif.loc[v, 'VIF']:.3f}  ground_truth={gt.loc[v, 'VIF']:.3f}")
        mae = (vif.loc[common, "VIF"] - gt.loc[common, "VIF"]).abs().mean()
        print(f"  mean |VIF diff| = {mae:.4f}")


if __name__ == "__main__":
    main()
