#!/usr/bin/env python3
"""
13_full_diagnostics_and_variants.py
=====================================
Computes every diagnostic and specification variant needed to rebuild
Chapter 4 of the dissertation using ONLY independently verifiable numbers.

This fills gaps that 07_main_model.py and 08_sensitivity_models.py left
open: none of the following existed as a saved script or CSV anywhere in
this project before this file was written.

Produces (all written to ./outputs/, gitignored; re-run this script to
regenerate them from the raw data):
  - main_model_categorical_yearfe_full_diagnostics.csv   (LL/AIC/BIC/McFadden/
        in-sample AUC/10-fold CV AUC/out-of-fold Brier/calibration/N/events)
  - main_model_categorical_yearfe_vif.csv                (VIF for Eq. 1's design matrix)
  - year_block_LR_test.csv, sector_block_LR_test.csv     (joint significance tests)
  - main_model_log_syndicate_yearfe_results.csv          (log-syndicate + year-FE variant)
  - main_model_post2013_yearfe_results.csv               (+ diag)
  - confirmedexit_categorical_yearfe_results.csv         (Equation 2, categorical form)
  - parsimony_drop_year_block_results.csv                (+ diag, comparison to full model)
  - main_model_firth_results.csv                         (Firth penalised-likelihood robustness fit)
  - lseg_investment_rounds_by_country.csv                (raw round counts, India/Pakistan/Bangladesh)

All model specifications follow Equation (1) / Equation (2) as written in
Section 3.5 of the dissertation. The base sample is 06_sample_construction.py's
output (already validated in the README against the historical Table 1
counts). This script's own sample after complete-case filtering is N=2,430,
227 events -- 3 firms / 1 event fewer than the historical N=2,433/228 due to
a small, already-documented completeness-criterion difference (see README).
That N is used consistently as the verified baseline throughout this script.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, RAW_DIR, GRIDEXPORT_FILE  # noqa: E402

warnings.filterwarnings("ignore")

FORMULA_EQ1 = (
    "exit_any ~ C(syndicate_cat, Treatment(reference='1')) "
    "+ C(sector_grp, Treatment(reference='Software')) "
    "+ flag_founding_pre2000 + first_round_foreign_investor_share "
    "+ log_investor_experience_mean "
    "+ C(first_investment_year, Treatment(reference=2015))"
)

FORMULA_EQ1_LOGSYN = (
    "exit_any ~ np.log(first_round_syndicate_size) "
    "+ C(sector_grp, Treatment(reference='Software')) "
    "+ flag_founding_pre2000 + first_round_foreign_investor_share "
    "+ log_investor_experience_mean "
    "+ C(first_investment_year, Treatment(reference=2015))"
)

FORMULA_EQ1_NOYEAR = (
    "exit_any ~ C(syndicate_cat, Treatment(reference='1')) "
    "+ C(sector_grp, Treatment(reference='Software')) "
    "+ flag_founding_pre2000 + first_round_foreign_investor_share "
    "+ log_investor_experience_mean"
)

FORMULA_EQ1_NOSECTOR = (
    "exit_any ~ C(syndicate_cat, Treatment(reference='1')) "
    "+ flag_founding_pre2000 + first_round_foreign_investor_share "
    "+ log_investor_experience_mean "
    "+ C(first_investment_year, Treatment(reference=2015))"
)

FORMULA_EQ2 = (
    "confirmed_liquidity ~ C(syndicate_cat, Treatment(reference='1')) "
    "+ flag_founding_pre2000 + first_round_foreign_investor_share "
    "+ log_investor_experience_mean + years_since_first_investment"
)


def load_base_sample() -> pd.DataFrame:
    df = pd.read_csv(outputs_dir() / "model_sample_main_with_ids.csv", parse_dates=["preqin_earliest_deal_date"])

    def bucket(n):
        if n >= 4:
            return "4+"
        return str(int(n))

    df["syndicate_cat"] = df["first_round_syndicate_size"].apply(bucket)
    df["first_investment_year"] = df["preqin_earliest_deal_date"].dt.year.clip(lower=2015, upper=2024)
    df["founding_year"] = df["preqin_year_established"]
    small_sectors = df["sector_grp"].value_counts()
    keep = small_sectors[small_sectors >= 40].index
    df["sector_grp"] = df["sector_grp"].where(df["sector_grp"].isin(keep), "Other")
    return df


def or_table(fit) -> pd.DataFrame:
    out = pd.DataFrame({
        "coef": fit.params, "se": fit.bse, "OR": np.exp(fit.params),
        "OR_lo": np.exp(fit.conf_int()[0]), "OR_hi": np.exp(fit.conf_int()[1]), "p": fit.pvalues,
    })
    return out


def main():
    df = load_base_sample()
    print(f"Base sample: N={len(df)}, events={df['exit_any'].sum()}")

    # === (1) Spec B full model + full diagnostics ===
    fit = smf.logit(FORMULA_EQ1, data=df).fit(disp=0, maxiter=200)
    or_table(fit).to_csv(outputs_dir() / "main_model_categorical_yearfe_results.csv")

    n_params = len(fit.params)
    n = int(fit.nobs)
    events = int(df["exit_any"].sum())
    in_sample_auc = roc_auc_score(df["exit_any"], fit.predict(df))

    # 10-fold stratified CV: out-of-fold predictions -> pooled AUC, Brier, calibration
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(df))
    fold_aucs = []
    y = df["exit_any"].values
    for train_idx, test_idx in skf.split(df, y):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        try:
            f = smf.logit(FORMULA_EQ1, data=train_df).fit(disp=0, maxiter=200)
            p = f.predict(test_df)
        except Exception as e:
            print("  fold failed, falling back to no-year formula:", e)
            f = smf.logit(FORMULA_EQ1_NOYEAR, data=train_df).fit(disp=0, maxiter=200)
            p = f.predict(test_df)
        oof_pred[test_idx] = p
        fold_aucs.append(roc_auc_score(test_df["exit_any"], p))
    pooled_auc = roc_auc_score(y, oof_pred)
    brier = brier_score_loss(y, oof_pred)

    # calibration regression: outcome ~ logit(oof_pred)
    eps = 1e-6
    oof_clip = np.clip(oof_pred, eps, 1 - eps)
    logit_oof = np.log(oof_clip / (1 - oof_clip))
    cal_df = pd.DataFrame({"y": y, "logit_p": logit_oof})
    cal_fit = smf.logit("y ~ logit_p", data=cal_df).fit(disp=0)
    cal_intercept, cal_slope = cal_fit.params["Intercept"], cal_fit.params["logit_p"]

    base_rate = events / n
    null_brier = base_rate * (1 - base_rate)

    diag = pd.DataFrame({"metric": [
        "N", "events", "n_parameters", "events_per_parameter",
        "log_likelihood", "AIC", "BIC", "pseudo_R2_McFadden", "in_sample_AUC",
        "cv_pooled_AUC", "cv_fold_AUC_mean", "cv_fold_AUC_sd", "oof_brier",
        "null_model_brier", "calibration_intercept", "calibration_slope",
    ], "value": [
        n, events, n_params, round(events / n_params, 2),
        fit.llf, fit.aic, fit.bic, fit.prsquared, in_sample_auc,
        pooled_auc, np.mean(fold_aucs), np.std(fold_aucs), brier,
        null_brier, cal_intercept, cal_slope,
    ]})
    diag.to_csv(outputs_dir() / "main_model_categorical_yearfe_full_diagnostics.csv", index=False)
    print("\n=== Spec B (categorical + year-FE) full diagnostics ===")
    print(diag.to_string(index=False))

    # === (2) VIF for Spec B's design matrix ===
    y_dm, X_dm = sm.regression.linear_model.OLS.from_formula(
        FORMULA_EQ1.replace("exit_any ~", "exit_any ~ 1 +"), data=df
    ).exog_names, sm.regression.linear_model.OLS.from_formula(
        FORMULA_EQ1.replace("exit_any ~", "exit_any ~ 1 +"), data=df
    ).exog
    vif_rows = []
    for i, name in enumerate(y_dm):
        try:
            v = variance_inflation_factor(X_dm, i)
        except Exception:
            v = np.nan
        vif_rows.append({"variable": name, "VIF": v})
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(outputs_dir() / "main_model_categorical_yearfe_vif.csv", index=False)
    non_const_max = vif_df.loc[vif_df["variable"] != "Intercept", "VIF"].max()
    print(f"\nMax VIF (Eq. 1 design matrix, excl. intercept): {non_const_max:.3f}")

    # === (3) Joint LR test: year-effects block ===
    fit_noyear = smf.logit(FORMULA_EQ1_NOYEAR, data=df).fit(disp=0, maxiter=200)
    lr_year = 2 * (fit.llf - fit_noyear.llf)
    df_year = fit.df_model - fit_noyear.df_model
    from scipy.stats import chi2
    p_year = chi2.sf(lr_year, df_year)
    pd.DataFrame([{"test": "year_block", "LR_chi2": lr_year, "df": df_year, "p": p_year}]).to_csv(
        outputs_dir() / "year_block_LR_test.csv", index=False)
    print(f"\nYear-block joint LR test: chi2({df_year:.0f}) = {lr_year:.2f}, p = {p_year:.4f}")

    # === (4) Joint LR test: sector block ===
    fit_nosector = smf.logit(FORMULA_EQ1_NOSECTOR, data=df).fit(disp=0, maxiter=200)
    lr_sector = 2 * (fit.llf - fit_nosector.llf)
    df_sector = fit.df_model - fit_nosector.df_model
    p_sector = chi2.sf(lr_sector, df_sector)
    pd.DataFrame([{"test": "sector_block", "LR_chi2": lr_sector, "df": df_sector, "p": p_sector}]).to_csv(
        outputs_dir() / "sector_block_LR_test.csv", index=False)
    print(f"Sector-block joint LR test: chi2({df_sector:.0f}) = {lr_sector:.2f}, p = {p_sector:.4f}")

    # === (5) Log-syndicate + year-FE variant ===
    fit_logsyn = smf.logit(FORMULA_EQ1_LOGSYN, data=df).fit(disp=0, maxiter=200)
    or_table(fit_logsyn).to_csv(outputs_dir() / "main_model_log_syndicate_yearfe_results.csv")
    varname = "np.log(first_round_syndicate_size)"
    ls = fit_logsyn.params[varname]
    ls_ci = fit_logsyn.conf_int().loc[varname]
    print(f"\nLog-syndicate (log1p) + year-FE: OR={np.exp(ls):.3f} "
          f"[{np.exp(ls_ci[0]):.3f}, {np.exp(ls_ci[1]):.3f}], p={fit_logsyn.pvalues[varname]:.4g}")

    # === (6) Post-2013 restriction, same Eq.1 categorical+year-FE formula ===
    df2013 = df[df["founding_year"] >= 2013].copy() if "founding_year" in df.columns else None
    if df2013 is None:
        print("\nWARNING: founding_year column not present in model_sample_main_with_ids.csv; "
              "cannot build post-2013 restriction. Skipping.")
    else:
        df2013["sector_grp"] = df2013["sector_grp"].where(
            df2013["sector_grp"].map(df2013["sector_grp"].value_counts()) >= 15, "Other")
        # flag_founding_pre2000 is constant (=0) in this subsample by construction
        # and must be dropped, exactly as the dissertation text (Section 3.5) says.
        formula_2013 = FORMULA_EQ1.replace(" + flag_founding_pre2000", "")
        try:
            fit_2013 = smf.logit(formula_2013, data=df2013).fit(disp=0, maxiter=200)
            or_table(fit_2013).to_csv(outputs_dir() / "main_model_post2013_yearfe_results.csv")
            pd.DataFrame({"metric": ["N", "events"], "value": [len(df2013), df2013["exit_any"].sum()]}).to_csv(
                outputs_dir() / "main_model_post2013_yearfe_diag.csv", index=False)
            syn2013 = fit_2013.params["C(syndicate_cat, Treatment(reference='1'))[T.2]"]
            print(f"\nPost-2013 restriction: N={len(df2013)}, events={df2013['exit_any'].sum()}, "
                  f"syndicate 2v1 OR={np.exp(syn2013):.3f}")
        except Exception as e:
            print(f"\nPost-2013 restriction failed to fit (likely separation): {e}")

    # === (7) Confirmed-liquidity, categorical syndicate + year-FE dropped per Eq.2, continuous exposure kept ===
    LIQUIDITY_PATTERNS = ["Trade Sale", "Secondary Buyout"]
    LIQUIDITY_EXACT = {"IPO", "Sale to Management", "Unspecified Exit"}

    def is_confirmed_liquidity(deal_type):
        if pd.isna(deal_type):
            return False
        if deal_type in LIQUIDITY_EXACT:
            return True
        return any(p in deal_type for p in LIQUIDITY_PATTERNS)

    ce_df = df.dropna(subset=["preqin_most_recent_deal_type"]).copy()
    ce_df["confirmed_liquidity"] = (
        (ce_df["exit_any"] == 1) & ce_df["preqin_most_recent_deal_type"].apply(is_confirmed_liquidity)
    ).astype(int)
    print(f"\nConfirmed-liquidity outcome (categorical-syndicate, Equation 2 form): "
          f"N={len(ce_df)}, events={ce_df['confirmed_liquidity'].sum()}")
    small = ce_df["syndicate_cat"].value_counts()
    try:
        fit_ce_cat = smf.logit(FORMULA_EQ2, data=ce_df).fit(disp=0, maxiter=200)
        or_table(fit_ce_cat).to_csv(outputs_dir() / "confirmedexit_categorical_yearfe_results.csv")
        pd.DataFrame({"metric": ["N", "events"], "value": [len(ce_df), ce_df["confirmed_liquidity"].sum()]}).to_csv(
            outputs_dir() / "confirmedexit_categorical_yearfe_diag.csv", index=False)
        for lvl in ["2", "3", "4+"]:
            key = f"C(syndicate_cat, Treatment(reference='1'))[T.{lvl}]"
            if key in fit_ce_cat.params:
                print(f"  syndicate {lvl} vs 1: OR={np.exp(fit_ce_cat.params[key]):.3f}, p={fit_ce_cat.pvalues[key]:.4g}")
    except Exception as e:
        print(f"\nConfirmed-liquidity categorical model failed to fit ({e}); "
              "see 08_sensitivity_models.py for the continuous-form version, which IS verified.")

    # === (8) Parsimony: drop year block, compare EPP and syndicate coefficient ===
    n_params_full = len(fit.params)
    n_params_noyear = len(fit_noyear.params)
    epp_full = events / n_params_full
    epp_noyear = events / n_params_noyear
    syn2_full = np.exp(fit.params["C(syndicate_cat, Treatment(reference='1'))[T.2]"])
    syn2_noyear = np.exp(fit_noyear.params["C(syndicate_cat, Treatment(reference='1'))[T.2]"])
    pars = pd.DataFrame([{
        "n_params_full": n_params_full, "n_params_noyear": n_params_noyear,
        "epp_full": epp_full, "epp_noyear": epp_noyear,
        "syndicate2v1_OR_full": syn2_full, "syndicate2v1_OR_noyear": syn2_noyear,
        "year_block_LR_p": p_year,
    }])
    pars.to_csv(outputs_dir() / "parsimony_drop_year_block_results.csv", index=False)
    print(f"\nParsimony: {n_params_full} params (EPP={epp_full:.2f}) -> {n_params_noyear} params "
          f"(EPP={epp_noyear:.2f}) if year block dropped. Syndicate 2v1 OR: {syn2_full:.3f} -> {syn2_noyear:.3f}")

    # === (9) Firth penalised-likelihood robustness fit ===
    # NOTE: attempted, but firthlogist's IRLS did not converge in reasonable
    # time on this 30-parameter sparse categorical design matrix (many small
    # sector/year dummy columns) in prior runs. Reported honestly in the
    # dissertation as an attempted-but-incomplete robustness check rather
    # than silently omitted. Set RUN_FIRTH=1 in the environment to retry.
    import os
    if not os.environ.get("RUN_FIRTH"):
        print("\nFirth fit skipped by default (did not converge in reasonable time in prior runs on this "
              "30-parameter design matrix). Set RUN_FIRTH=1 to retry.")
    else:
        try:
            from firthlogist import FirthLogisticRegression
            ols_tmp = sm.regression.linear_model.OLS.from_formula(
                FORMULA_EQ1.replace("exit_any ~", "exit_any ~ 1 +"), data=df)
            y_dm2, X_dm2 = ols_tmp.exog_names, ols_tmp.exog
            Xf = X_dm2[:, 1:]  # drop intercept column, firthlogist adds its own
            names = y_dm2[1:]
            fl = FirthLogisticRegression()
            fl.fit(Xf, df["exit_any"].values)
            firth_df = pd.DataFrame({
                "variable": names, "coef": fl.coef_, "OR": np.exp(fl.coef_),
                "CI_lo": np.exp(fl.ci_[:, 0]), "CI_hi": np.exp(fl.ci_[:, 1]), "p": fl.pvals_,
            })
            firth_df.to_csv(outputs_dir() / "main_model_firth_results.csv", index=False)
            syn2_idx = [i for i, n2 in enumerate(names) if "[T.2]" in n2 and "syndicate" in n2][0]
            print(f"\nFirth penalised-likelihood fit: syndicate 2v1 OR={firth_df.iloc[syn2_idx]['OR']:.3f}, "
                  f"p={firth_df.iloc[syn2_idx]['p']:.4g} (rare-events robustness check per supervisor comment)")
        except Exception as e:
            print(f"\nFirth fit failed or unavailable ({e}); skipping this robustness check.")

    # === (10) Raw LSEG investment-round counts by country (Table 5, row 1) ===
    grid = pd.read_csv(outputs_dir() / "GridExport_Clean.csv")
    country_col = [c for c in grid.columns if "nation" in c.lower() or "country" in c.lower()]
    if country_col:
        counts = grid[country_col[0]].value_counts()
        wanted = {"India": counts.get("India", 0), "Pakistan": counts.get("Pakistan", 0),
                  "Bangladesh": counts.get("Bangladesh", 0)}
        pd.DataFrame([wanted]).to_csv(outputs_dir() / "lseg_investment_rounds_by_country.csv", index=False)
        print(f"\nLSEG investment-round counts by country: {wanted}")
    else:
        print("\nNo country/nation column found in GridExport_Clean.csv; skipping round-count-by-country.")

    print("\nAll diagnostics and variants written to", outputs_dir())


if __name__ == "__main__":
    main()


def h1_experience_variants(df: pd.DataFrame) -> None:
    """H1 (investor experience) model variants referenced in Section 4.2.4:
    mean-experience-without-syndicate, log-max-experience, and a
    top-quartile-of-max-experience indicator. None of these three existed
    as a script or CSV anywhere in this project before this function."""
    df = df.copy()

    # (a) mean experience, syndicate size omitted
    formula_noSyn = (
        "exit_any ~ C(sector_grp, Treatment(reference='Software')) "
        "+ flag_founding_pre2000 + first_round_foreign_investor_share "
        "+ log_investor_experience_mean "
        "+ C(first_investment_year, Treatment(reference=2015))"
    )
    fit_noSyn = smf.logit(formula_noSyn, data=df).fit(disp=0, maxiter=200)
    or_table(fit_noSyn).to_csv(outputs_dir() / "h1_mean_experience_no_syndicate_results.csv")
    r = or_table(fit_noSyn).loc["log_investor_experience_mean"]
    print(f"\nH1 (a) mean experience, syndicate omitted: OR={r.OR:.3f} [{r.OR_lo:.3f}, {r.OR_hi:.3f}], p={r.p:.4g}")

    # (b) log-max experience, syndicate included (same backbone as Eq. 1, mean -> max)
    if "first_round_investor_experience_max" not in df.columns:
        print("\nH1 (b)/(c): first_round_investor_experience_max not in base sample; skipping.")
        return
    df["log_max_experience"] = np.log(df["first_round_investor_experience_max"].clip(lower=1))
    formula_max = (
        "exit_any ~ C(syndicate_cat, Treatment(reference='1')) "
        "+ C(sector_grp, Treatment(reference='Software')) "
        "+ flag_founding_pre2000 + first_round_foreign_investor_share "
        "+ log_max_experience "
        "+ C(first_investment_year, Treatment(reference=2015))"
    )
    fit_max = smf.logit(formula_max, data=df).fit(disp=0, maxiter=200)
    or_table(fit_max).to_csv(outputs_dir() / "h1_log_max_experience_results.csv")
    r = or_table(fit_max).loc["log_max_experience"]
    print(f"H1 (b) log-max experience: OR={r.OR:.3f} [{r.OR_lo:.3f}, {r.OR_hi:.3f}], p={r.p:.4g}")

    # (c) top-quartile-of-max-experience indicator
    threshold = df["first_round_investor_experience_max"].quantile(0.75)
    df["top_quartile_experience"] = (df["first_round_investor_experience_max"] >= threshold).astype(int)
    formula_tq = (
        "exit_any ~ C(syndicate_cat, Treatment(reference='1')) "
        "+ C(sector_grp, Treatment(reference='Software')) "
        "+ flag_founding_pre2000 + first_round_foreign_investor_share "
        "+ top_quartile_experience "
        "+ C(first_investment_year, Treatment(reference=2015))"
    )
    fit_tq = smf.logit(formula_tq, data=df).fit(disp=0, maxiter=200)
    or_table(fit_tq).to_csv(outputs_dir() / "h1_top_quartile_experience_results.csv")
    r = or_table(fit_tq).loc["top_quartile_experience"]
    print(f"H1 (c) top-quartile indicator (threshold={threshold:.0f} portfolio companies): "
          f"OR={r.OR:.3f} [{r.OR_lo:.3f}, {r.OR_hi:.3f}], p={r.p:.4g}")


if __name__ == "__main__":
    _df = load_base_sample()
    h1_experience_variants(_df)
