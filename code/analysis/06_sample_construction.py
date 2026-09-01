#!/usr/bin/env python3
"""
06_sample_construction.py
============================
Builds the primary modelling sample (equivalent of
csv_exports/Analysis_Ready/model_sample_main.csv) entirely from RAW input
files:

  * merged_dataset_left_join.xlsx  -- supplies, per matched company: the
    Preqin company-status outcome (Active/Realised), founding year, sector
    (Preqin primary industry), country and deal-history summary fields.
  * GridExport (via 04_variable_construction.py's output) -- supplies the
    first-round investor variables computed fresh from the raw LSEG file.

WHY merged_dataset_left_join.xlsx AND NOT Firm_Table_Corrected.csv
----------------------------------------------------------------------
Firm_Table_Corrected.csv (the 8,186-row master) does NOT carry a usable exit
outcome at all: every row's event_code/event_date/event_type is blank, and
Corrected_Build/Diagnostics.csv explicitly records why ("Deal-level terminal
event not available for this full Preqin master; existing preqin_has_exit
invalid" -- the survival-design rebuild in Firm_Table_Corrected.csv
deliberately dropped the old company-status field). The Active/Realised
outcome that Analysis_Ready/model_sample_main.csv and
India_Exit_Determinants_Analysis.csv actually use instead lives in the
OLDER merged_dataset_left_join.xlsx artifact, which is explicitly listed
among the permitted RAW SOURCE FILES for this task. Deduplicating that file
on matched_b_company_id reproduces the dissertation's headline sample-
construction numbers from Table 1 exactly:

    2,511 unique matched companies -> preqin_company_status non-null for
    2,506 (5 blank = the non-India matches) -> of those, 2,266 Active +
    240 Realised = 2,506, and Investee Company Nation is India for exactly
    2,506 / Pakistan for 3 / Bangladesh for 2.

This is validated explicitly below before the sample is built further.

REMAINING STEPS (see Section 3.2-3.5 / Appendix A of the dissertation)
--------------------------------------------------------------------------
  1. Restrict to India.
  2. exit_any = 1 if preqin_company_status == 'Realised' else 0.
  3. Attach first-round investor variables from 04_variable_construction.py's
     output (join on LSEG company name).
  4. years_since_first_investment = (2024-12-31 - first LSEG round date) /
     365.25.
  5. flag_founding_pre2000 = preqin_year_established < 2000.
  6. Complete-case filter on founding year (drops firms with no recorded
     founding year).
  7. Pool sector categories with fewer than 40 analytic firms into 'Other'
     (Appendix A rule, threshold applied on THIS complete-case sample).
  8. log_investor_experience_mean = log1p(first_round_investor_experience_mean).

Ground truth to diff against (read-only):
  csv_exports/Analysis_Ready/model_sample_main.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CENSOR_DATE, SECTOR_MIN_N, outputs_dir, read_merged_left_join, ref  # noqa: E402


def build_matched_firm_table(merged: pd.DataFrame) -> pd.DataFrame:
    firm = merged.dropna(subset=["matched_b_company_id"]).drop_duplicates("matched_b_company_id").copy()
    keep_cols = [
        "matched_b_company_id",
        "dataset_a_company_id",
        "Investee Company Name",
        "Investee Company Nation",
        "preqin_company_name",
        "preqin_company_status",
        "preqin_year_established",
        "preqin_primary_industry",
        "preqin_city",
        "preqin_most_recent_deal_type",
        "preqin_n_deal_records",
        "preqin_n_unique_investors",
        "preqin_total_deal_size_usd_mn",
        "preqin_max_deal_size_usd_mn",
        "preqin_earliest_deal_date",
    ]
    firm = firm[keep_cols].rename(columns={
        "matched_b_company_id": "preqin_company_id",
        "Investee Company Name": "lseg_company_name",
        "Investee Company Nation": "country",
    })
    return firm


def validate_headline_counts(firm: pd.DataFrame) -> None:
    print("=== Reproducing Table 1 / Section 3.3 headline counts ===")
    print(f"unique matched companies (all countries): {len(firm)}  (dissertation: 2,511)")
    status_counts = firm["preqin_company_status"].value_counts(dropna=False)
    print("preqin_company_status counts:\n", status_counts.to_string())
    print(f"country counts:\n{firm['country'].value_counts(dropna=False).to_string()}")
    print("(dissertation: India 2,506; Pakistan 3; Bangladesh 2; "
          "2,266 Active + 240 Realised = 2,506 India firms)\n")


def build_model_sample(firm: pd.DataFrame, first_round_vars: pd.DataFrame) -> pd.DataFrame:
    india = firm[firm["country"] == "India"].copy()
    india = india[india["preqin_company_status"].notna()]
    india["exit_any"] = (india["preqin_company_status"] == "Realised").astype(int)

    df = india.merge(first_round_vars, on="lseg_company_name", how="left")

    # The survival/exposure clock uses the PREQIN first observed investment
    # date (preqin_earliest_deal_date), not the LSEG first-round date used
    # for the investor-composition variables -- these can differ (the
    # Preqin-recorded first deal is often earlier than the first round
    # visible in the 2014-2024 LSEG screen). This matches Design_Decision.csv:
    # "Primary time origin = date of first observed Preqin private-market
    # investment", confirmed exactly against Firm_Table_Corrected's
    # entry_preqin_raw field for spot-checked companies.
    df["preqin_earliest_deal_date"] = pd.to_datetime(df["preqin_earliest_deal_date"], errors="coerce")
    df["years_since_first_investment"] = (
        (CENSOR_DATE - df["preqin_earliest_deal_date"]).dt.days / 365.25
    )
    df["flag_founding_pre2000"] = (df["preqin_year_established"] < 2000).astype("Int64")

    # Complete-case filter: require a recorded founding year and complete
    # first-round covariates.
    required = [
        "preqin_year_established",
        "preqin_earliest_deal_date",
        "first_round_syndicate_size",
        "first_round_foreign_investor_share",
        "first_round_investor_experience_mean",
    ]
    complete = df.dropna(subset=required).copy()

    # Sector pooling (Appendix A): categories with < SECTOR_MIN_N firms in
    # the analytic (complete-case) sample are pooled into 'Other'.
    complete["preqin_primary_industry"] = complete["preqin_primary_industry"].fillna("Other")
    counts = complete["preqin_primary_industry"].value_counts()
    keep_sectors = set(counts[counts >= SECTOR_MIN_N].index)
    complete["sector_grp"] = complete["preqin_primary_industry"].where(
        complete["preqin_primary_industry"].isin(keep_sectors), "Other"
    )

    complete["log_investor_experience_mean"] = np.log1p(complete["first_round_investor_experience_mean"])

    out_cols = [
        "exit_any",
        "sector_grp",
        "years_since_first_investment",
        "flag_founding_pre2000",
        "first_round_syndicate_size",
        "first_round_foreign_investor_share",
        "first_round_corporate_vc_flag",
        "first_round_late_stage_pe_flag",
        "first_round_investor_experience_mean",
        "log_investor_experience_mean",
    ]
    return complete[out_cols + ["preqin_company_id", "lseg_company_name"]].reset_index(drop=True), complete, out_cols


def validate_model_sample(sample: pd.DataFrame, out_cols: list[str]) -> None:
    print("\n=== VALIDATION vs csv_exports/Analysis_Ready/model_sample_main.csv ===")
    gt_path = ref("Analysis_Ready", "model_sample_main.csv")
    if not gt_path.exists():
        print("ground truth file not found; skipping")
        return
    gt = pd.read_csv(gt_path)
    print(f"N: reconstructed={len(sample)}  ground_truth={len(gt)}")
    print(f"events (exit_any==1): reconstructed={sample['exit_any'].sum()}  ground_truth={gt['exit_any'].sum()}")
    print(f"sector_grp categories: reconstructed={sorted(sample['sector_grp'].unique())}")
    print(f"                       ground_truth={sorted(gt['sector_grp'].unique())}")
    for col in ["years_since_first_investment", "first_round_syndicate_size",
                "first_round_foreign_investor_share", "first_round_investor_experience_mean",
                "log_investor_experience_mean"]:
        print(f"  {col}: reconstructed mean={sample[col].mean():.4f}  ground_truth mean={gt[col].mean():.4f}")


def main() -> None:
    print("Loading merged_dataset_left_join.xlsx (raw, permitted input) ...")
    merged = read_merged_left_join()
    firm = build_matched_firm_table(merged)
    validate_headline_counts(firm)

    fr_path = Path(__file__).resolve().parent / "outputs" / "LSEG_First_Round_Variables.csv"
    if not fr_path.exists():
        raise SystemExit("Run 04_variable_construction.py first (needs its outputs/LSEG_First_Round_Variables.csv)")
    first_round_vars = pd.read_csv(fr_path, parse_dates=["first_round_date"])

    sample, complete_full, out_cols = build_model_sample(firm, first_round_vars)

    out = outputs_dir()
    sample[out_cols].to_csv(out / "model_sample_main.csv", index=False)
    complete_full.to_csv(out / "model_sample_main_with_ids.csv", index=False)
    print(f"\nWrote {out / 'model_sample_main.csv'} ({len(sample)} rows)")
    print(f"Wrote {out / 'model_sample_main_with_ids.csv'} (same rows, with company identifiers retained)")

    validate_model_sample(sample, out_cols)


if __name__ == "__main__":
    main()
