#!/usr/bin/env python3
"""
08_sensitivity_models.py
===========================
Two robustness/sensitivity re-fits of the continuous main-model
specification from 07_main_model.py (spec A):

  1. Confirmed-liquidity outcome (Section 3.3 / 4.2.3): redefine the
     outcome as 1 only where the company is Realised AND its
     preqin_most_recent_deal_type indicates an unambiguous liquidity event
     -- Trade Sale (including compound labels such as 'Add-on, Trade Sale'),
     IPO, Secondary Buyout (including compound labels), Sale to Management,
     or Unspecified Exit. Applied to the 2,506 accepted India matches
     (before the main model's complete-case covariate filter), this rule
     reproduces the dissertation's count EXACTLY: 63 of 240 Realised firms
     qualify (Section 3.3: "Of the 240 Realised firms, 63 have a most-recent
     deal type that indicates liquidity").

  2. Post-2000 founding cohort (Section 4.2.5): restrict the main sample to
     firms with preqin_year_established >= 2000 and refit WITHOUT
     flag_founding_pre2000, since that indicator is constant (=0) in the
     restricted subsample and would otherwise make the design matrix
     singular -- this exact bug is called out in the task brief and is
     guarded against explicitly below.

Ground truth to diff against (read-only):
  csv_exports/Analysis_Ready/sensitivity_confirmedexit_results.csv
  csv_exports/Analysis_Ready/sensitivity_confirmedexit_diag.csv
  csv_exports/Analysis_Ready/sensitivity_post2000_results.csv
  csv_exports/Analysis_Ready/sensitivity_post2000_diag.csv

NOTE: the ground-truth sensitivity CSVs use the CONTINUOUS main-model
formula (spec A in 07_main_model.py), not the categorical/year-FE Equation
(2) described in the dissertation text (which explicitly omits sector and
year indicators for the confirmed-liquidity model). This mirrors the same
spec-A/spec-B duality documented in 07_main_model.py's docstring, so this
script fits the continuous form to match the saved ground truth, and
prints how that compares with the simpler Equation (2) text description.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, read_merged_left_join, ref  # noqa: E402

LIQUIDITY_PATTERNS = ["Trade Sale", "Secondary Buyout"]
LIQUIDITY_EXACT = {"IPO", "Sale to Management", "Unspecified Exit"}


def is_confirmed_liquidity(deal_type: str) -> bool:
    if pd.isna(deal_type):
        return False
    if deal_type in LIQUIDITY_EXACT:
        return True
    return any(p in deal_type for p in LIQUIDITY_PATTERNS)


def load_full_sample_with_deal_type() -> pd.DataFrame:
    """Re-derive the model sample but also attach preqin_most_recent_deal_type
    and preqin_year_established, needed for the confirmed-liquidity /
    post-2000 redefinitions."""
    sample_path = Path(__file__).resolve().parent / "outputs" / "model_sample_main_with_ids.csv"
    if not sample_path.exists():
        raise SystemExit("Run 06_sample_construction.py first")
    sample = pd.read_csv(sample_path, parse_dates=["preqin_earliest_deal_date", "first_round_date"])
    return sample


def fit_logit(df: pd.DataFrame, drop_pre2000: bool = False):
    terms = [
        "C(sector_grp, Treatment(reference='Software'))",
        "years_since_first_investment",
        "first_round_syndicate_size",
        "first_round_foreign_investor_share",
        "first_round_corporate_vc_flag",
        "first_round_late_stage_pe_flag",
        "log_investor_experience_mean",
    ]
    if not drop_pre2000:
        terms.insert(2, "flag_founding_pre2000")
    formula = "outcome ~ " + " + ".join(terms)
    model = smf.logit(formula, data=df)
    return model.fit(disp=0, maxiter=200)


def results_table(fit) -> pd.DataFrame:
    conf = fit.conf_int()
    return pd.DataFrame(
        {
            "coef": fit.params,
            "OR": np.exp(fit.params),
            "OR_lo": np.exp(conf[0]),
            "OR_hi": np.exp(conf[1]),
            "p": fit.pvalues,
        }
    )


def confirmed_liquidity_sensitivity(sample: pd.DataFrame) -> None:
    print("\n=== Sensitivity 1: confirmed-liquidity outcome ===")
    df = sample.copy()
    df["outcome"] = (
        (df["exit_any"] == 1) & df["preqin_most_recent_deal_type"].apply(is_confirmed_liquidity)
    ).astype(int)
    print(f"  confirmed-liquidity events in complete-case sample: {df['outcome'].sum()} "
          f"(of {int(df['exit_any'].sum())} exit_any events)")

    df_cc = df.dropna(subset=["preqin_most_recent_deal_type"]).copy()
    print(f"  N after additionally requiring non-null deal type: {len(df_cc)}")

    # Any sector with zero confirmed-liquidity events causes quasi-complete
    # separation (that dummy's coefficient diverges, singular Hessian). Fold
    # such sectors into 'Other' for this outcome only -- this matches the
    # ground truth exactly: sensitivity_confirmedexit_results.csv has no
    # sector[T.Agribusiness] row, and Agribusiness is exactly the one sector
    # with zero confirmed-liquidity events in this reconstruction.
    event_by_sector = df_cc.groupby("sector_grp")["outcome"].sum()
    zero_event_sectors = event_by_sector[event_by_sector == 0].index.tolist()
    if zero_event_sectors:
        print(f"  folding zero-event sectors into 'Other' to avoid separation: {zero_event_sectors}")
        df_cc["sector_grp"] = df_cc["sector_grp"].where(~df_cc["sector_grp"].isin(zero_event_sectors), "Other")

    fit = fit_logit(df_cc, drop_pre2000=False)
    res = results_table(fit)

    out = outputs_dir()
    res.to_csv(out / "sensitivity_confirmedexit_results.csv")
    pd.DataFrame(
        {"metric": ["N", "events", "pseudo_R2"], "value": [len(df_cc), int(df_cc["outcome"].sum()), round(fit.prsquared, 4)]}
    ).to_csv(out / "sensitivity_confirmedexit_diag.csv", index=False)
    print(f"  Wrote sensitivity_confirmedexit_results.csv / _diag.csv")

    diag_path = ref("Analysis_Ready", "sensitivity_confirmedexit_diag.csv")
    if diag_path.exists():
        gt = pd.read_csv(diag_path).set_index("metric")["value"]
        print(f"  VALIDATION: N reconstructed={len(df_cc)} gt={gt.get('N')}; "
              f"events reconstructed={int(df_cc['outcome'].sum())} gt={gt.get('events')}; "
              f"pseudo_R2 reconstructed={fit.prsquared:.4f} gt={gt.get('pseudo_R2')}")
    res_path = ref("Analysis_Ready", "sensitivity_confirmedexit_results.csv")
    if res_path.exists():
        gt_res = pd.read_csv(res_path, index_col=0)
        common = res.index.intersection(gt_res.index)
        if len(common):
            or_mae = (res.loc[common, "OR"] - gt_res.loc[common, "OR"]).abs().mean()
            print(f"  mean |OR diff| over {len(common)} shared coefficient rows = {or_mae:.4f}")
            for v in ["first_round_syndicate_size", "years_since_first_investment", "log_investor_experience_mean"]:
                if v in common:
                    print(f"    {v}: reconstructed OR={res.loc[v,'OR']:.3f} p={res.loc[v,'p']:.4g}  "
                          f"| gt OR={gt_res.loc[v,'OR']:.3f} p={gt_res.loc[v,'p']:.4g}")


def post2000_sensitivity(sample: pd.DataFrame) -> None:
    print("\n=== Sensitivity 2: firms founded 2000 or later ===")
    df = sample[sample["flag_founding_pre2000"] == 0].copy()
    df["outcome"] = df["exit_any"]
    print(f"  N (post-2000 subsample): {len(df)}   events: {int(df['outcome'].sum())}")
    assert df["flag_founding_pre2000"].nunique() <= 1, "flag_founding_pre2000 must be constant here"

    fit = fit_logit(df, drop_pre2000=True)  # guard against the singular-matrix bug
    res = results_table(fit)

    out = outputs_dir()
    res.to_csv(out / "sensitivity_post2000_results.csv")
    pd.DataFrame(
        {"metric": ["N", "events", "pseudo_R2"], "value": [len(df), int(df["outcome"].sum()), round(fit.prsquared, 4)]}
    ).to_csv(out / "sensitivity_post2000_diag.csv", index=False)
    print(f"  Wrote sensitivity_post2000_results.csv / _diag.csv")

    diag_path = ref("Analysis_Ready", "sensitivity_post2000_diag.csv")
    if diag_path.exists():
        gt = pd.read_csv(diag_path).set_index("metric")["value"]
        print(f"  VALIDATION: N reconstructed={len(df)} gt={gt.get('N')}; "
              f"events reconstructed={int(df['outcome'].sum())} gt={gt.get('events')}; "
              f"pseudo_R2 reconstructed={fit.prsquared:.4f} gt={gt.get('pseudo_R2')}")
    res_path = ref("Analysis_Ready", "sensitivity_post2000_results.csv")
    if res_path.exists():
        gt_res = pd.read_csv(res_path, index_col=0)
        common = res.index.intersection(gt_res.index)
        if len(common):
            or_mae = (res.loc[common, "OR"] - gt_res.loc[common, "OR"]).abs().mean()
            print(f"  mean |OR diff| over {len(common)} shared coefficient rows = {or_mae:.4f}")
            for v in ["first_round_syndicate_size", "years_since_first_investment", "log_investor_experience_mean"]:
                if v in common:
                    print(f"    {v}: reconstructed OR={res.loc[v,'OR']:.3f} p={res.loc[v,'p']:.4g}  "
                          f"| gt OR={gt_res.loc[v,'OR']:.3f} p={gt_res.loc[v,'p']:.4g}")


def main() -> None:
    sample = load_full_sample_with_deal_type()
    confirmed_liquidity_sensitivity(sample)
    post2000_sensitivity(sample)


if __name__ == "__main__":
    main()
