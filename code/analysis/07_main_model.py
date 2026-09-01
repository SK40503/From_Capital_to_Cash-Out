#!/usr/bin/env python3
"""
07_main_model.py
==================
Fits the primary logistic-regression model of exit incidence on the sample
built by 06_sample_construction.py.

IMPORTANT: TWO DIFFERENT HISTORICAL SPECIFICATIONS EXIST, AND THIS SCRIPT
REPRODUCES BOTH
--------------------------------------------------------------------------
Cross-referencing csv_exports/Analysis_Ready/main_model_results.csv against
the dissertation text (Section 3.5, Table 3/Table 4 in
drafts/Dissertation_Restructured.docx) reveals that they describe two
DIFFERENT parameterisations of "the main model", not the same model twice:

  (A) The CSV (main_model_results.csv / main_model_diagnostics.csv /
      vif_table.csv / the sensitivity_*.csv files) uses first_round_syndicate_
      size and years_since_first_investment as CONTINUOUS covariates, and
      also includes first_round_corporate_vc_flag and
      first_round_late_stage_pe_flag directly in the regression. This
      specification has 27 estimated parameters (N=2,434, 228 events,
      McFadden pseudo-R^2 = 0.0882, AUC = 0.7162).

  (B) The dissertation prose (Section 3.5, Equation (1)) describes syndicate
      size entered as 1/2/3/4+ CATEGORICAL dummies and first-investment-year
      FIXED EFFECTS for 2015-2024 (2015 = reference), explicitly EXCLUDING
      the corporate-VC and late-stage-PE proxies from the confirmatory
      model (they are reported only in Appendix C as exploratory). This is
      the model behind Table 4 / Table 3 in the docx, with 31 parameters
      (N=2,433, 228 events, LL=-687.70, AIC=1,437.41, BIC=1,617.11, pseudo-
      R^2=0.091, in-sample AUC=0.721).

The dissertation itself explains why these differ (Section 4.5): "Because
the exact estimation script behind Table 4 was not available for direct
modification, the checks below use an independently rebuilt replication
drawn from the same underlying Preqin and LSEG files" -- i.e. even at
dissertation-writing time, the ORIGINAL script behind Table 4 was already
lost, exactly as described in this reconstruction task's brief, and a
second, continuous-form replication was built alongside it. The CSV files
in Analysis_Ready/ are artifacts of that continuous-form replication family
(confirmed by vif_table.csv and the sensitivity CSVs sharing its exact
variable set), not of the original Table-4 script.

This script therefore fits BOTH specifications and validates each against
its own ground truth:
  * fit_continuous_spec()  -> validated numerically against
    csv_exports/Analysis_Ready/main_model_results.csv and
    main_model_diagnostics.csv (this is the PRIMARY validated deliverable).
  * fit_categorical_yearfe_spec() -> a best-effort reproduction of Equation
    (1) as described in the dissertation text, validated against the
    handful of numbers quoted in the text/tables (LL, AIC, BIC, pseudo-R2,
    AUC, and the three syndicate-size odds ratios) since no separate
    machine-readable ground-truth CSV for this exact parameterisation
    exists in csv_exports/.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, ref  # noqa: E402


def load_sample() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "outputs" / "model_sample_main_with_ids.csv"
    if not path.exists():
        raise SystemExit("Run 06_sample_construction.py first")
    return pd.read_csv(path, parse_dates=["preqin_earliest_deal_date", "first_round_date"])


def results_table(fit) -> pd.DataFrame:
    conf = fit.conf_int()
    out = pd.DataFrame(
        {
            "coef": fit.params,
            "se": fit.bse,
            "OR": np.exp(fit.params),
            "OR_lo": np.exp(conf[0]),
            "OR_hi": np.exp(conf[1]),
            "p": fit.pvalues,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Specification (A): continuous form, matches Analysis_Ready/main_model_results.csv
# ---------------------------------------------------------------------------
def fit_continuous_spec(df: pd.DataFrame):
    formula = (
        "exit_any ~ C(sector_grp, Treatment(reference='Software')) "
        "+ years_since_first_investment + flag_founding_pre2000 "
        "+ first_round_syndicate_size + first_round_foreign_investor_share "
        "+ first_round_corporate_vc_flag + first_round_late_stage_pe_flag "
        "+ log_investor_experience_mean"
    )
    model = smf.logit(formula, data=df)
    fit = model.fit(disp=0)
    return fit


def validate_continuous_spec(fit, df: pd.DataFrame) -> None:
    print("\n=== VALIDATION (A): continuous spec vs Analysis_Ready/main_model_results.csv ===")
    res = results_table(fit)
    gt_path = ref("Analysis_Ready", "main_model_results.csv")
    diag_path = ref("Analysis_Ready", "main_model_diagnostics.csv")
    if gt_path.exists():
        gt = pd.read_csv(gt_path, index_col=0)
        common_idx = res.index.intersection(gt.index)
        print(f"  matched coefficient rows: {len(common_idx)} / {len(gt)}")
        coef_mae = (res.loc[common_idx, "coef"] - gt.loc[common_idx, "coef"]).abs().mean()
        or_mae = (res.loc[common_idx, "OR"] - gt.loc[common_idx, "OR"]).abs().mean()
        print(f"  mean |coef diff| = {coef_mae:.4f}   mean |OR diff| = {or_mae:.4f}")
        key_vars = ["first_round_syndicate_size", "log_investor_experience_mean",
                    "first_round_foreign_investor_share", "years_since_first_investment"]
        for v in key_vars:
            if v in res.index and v in gt.index:
                print(f"  {v}: reconstructed OR={res.loc[v, 'OR']:.3f} (gt={gt.loc[v, 'OR']:.3f}), "
                      f"p={res.loc[v, 'p']:.4g} (gt={gt.loc[v, 'p']:.4g})")
    if diag_path.exists():
        diag = pd.read_csv(diag_path).set_index("metric")["value"]
        n = int(df.shape[0])
        events = int(df["exit_any"].sum())
        pseudo_r2 = fit.prsquared
        auc = roc_auc_score(df["exit_any"], fit.predict(df))
        print(f"  N: reconstructed={n}  gt={diag.get('N')}")
        print(f"  events: reconstructed={events}  gt={diag.get('events')}")
        print(f"  pseudo_R2_McFadden: reconstructed={pseudo_r2:.4f}  gt={diag.get('pseudo_R2_McFadden')}")
        print(f"  AUC (in-sample): reconstructed={auc:.4f}  gt={diag.get('AUC')}")


# ---------------------------------------------------------------------------
# Specification (B): categorical syndicate + first-investment-year FE,
# best-effort reproduction of Equation (1) / Table 4 as described in text.
# ---------------------------------------------------------------------------
def fit_categorical_yearfe_spec(df: pd.DataFrame):
    df = df.copy()

    def bucket(n):
        if n >= 4:
            return "4+"
        return str(int(n))

    df["syndicate_cat"] = df["first_round_syndicate_size"].apply(bucket)
    df["first_investment_year"] = df["preqin_earliest_deal_date"].dt.year
    # Restrict to the 2015-2024 range described in the text; earlier/later
    # years get folded to the nearest edge to avoid dropping observations
    # outright (documented best-effort choice -- the text does not specify
    # a treatment for out-of-range years).
    df["first_investment_year"] = df["first_investment_year"].clip(lower=2015, upper=2024)

    formula = (
        "exit_any ~ C(sector_grp, Treatment(reference='Software')) "
        "+ C(syndicate_cat, Treatment(reference='1')) "
        "+ flag_founding_pre2000 + first_round_foreign_investor_share "
        "+ log_investor_experience_mean "
        "+ C(first_investment_year, Treatment(reference=2015))"
    )
    model = smf.logit(formula, data=df)
    fit = model.fit(disp=0, maxiter=200)
    return fit, df


def validate_categorical_yearfe_spec(fit, df: pd.DataFrame) -> None:
    print("\n=== VALIDATION (B): categorical + year-FE spec vs dissertation text (Table 3/4) ===")
    res = results_table(fit)
    n_params = len(fit.params)
    print(f"  N parameters: reconstructed={n_params}  dissertation text=31")
    n = int(df.shape[0])
    events = int(df["exit_any"].sum())
    auc = roc_auc_score(df["exit_any"], fit.predict(df))
    pseudo_r2 = fit.prsquared
    print(f"  N={n} (text: 2,433)   events={events} (text: 228)")
    print(f"  LL={fit.llf:.2f} (text: -687.70)   AIC={fit.aic:.2f} (text: 1,437.41)   "
          f"BIC={fit.bic:.2f} (text: 1,617.11)")
    print(f"  pseudo_R2_McFadden={pseudo_r2:.3f} (text: 0.091)   in-sample AUC={auc:.3f} (text: 0.721)")
    for level, gt_or in [("2", 0.38), ("3", 0.14), ("4+", 0.04)]:
        key = f"C(syndicate_cat, Treatment(reference='1'))[T.{level}]"
        if key in res.index:
            print(f"  syndicate {level} vs 1: reconstructed OR={res.loc[key, 'OR']:.3f} "
                  f"(text: {gt_or})")
    if "flag_founding_pre2000" in res.index:
        print(f"  founded pre-2000: reconstructed OR={res.loc['flag_founding_pre2000', 'OR']:.3f} (text: 1.21)")
    if "first_round_foreign_investor_share" in res.index:
        print(f"  foreign investor share: reconstructed OR="
              f"{res.loc['first_round_foreign_investor_share', 'OR']:.3f} (text: 1.07)")
    if "log_investor_experience_mean" in res.index:
        print(f"  log mean experience: reconstructed OR="
              f"{res.loc['log_investor_experience_mean', 'OR']:.3f} (text: 1.10)")


def main() -> None:
    df = load_sample()
    out = outputs_dir()

    print("Fitting specification (A): continuous form ...")
    fit_a = fit_continuous_spec(df)
    res_a = results_table(fit_a)
    res_a.to_csv(out / "main_model_results.csv")
    pd.DataFrame(
        {
            "metric": ["N", "events", "pseudo_R2_McFadden", "AUC", "LLR_pvalue"],
            "value": [
                len(df),
                int(df["exit_any"].sum()),
                round(fit_a.prsquared, 4),
                roc_auc_score(df["exit_any"], fit_a.predict(df)),
                fit_a.llr_pvalue,
            ],
        }
    ).to_csv(out / "main_model_diagnostics.csv", index=False)
    with open(out / "main_model.pickle", "wb") as f:
        pickle.dump(fit_a, f)
    print(f"Wrote {out / 'main_model_results.csv'}, main_model_diagnostics.csv, main_model.pickle")
    validate_continuous_spec(fit_a, df)

    print("\nFitting specification (B): categorical syndicate + year FE ...")
    fit_b, df_b = fit_categorical_yearfe_spec(df)
    results_table(fit_b).to_csv(out / "main_model_categorical_yearfe_results.csv")
    print(f"Wrote {out / 'main_model_categorical_yearfe_results.csv'}")
    validate_categorical_yearfe_spec(fit_b, df_b)


if __name__ == "__main__":
    main()
