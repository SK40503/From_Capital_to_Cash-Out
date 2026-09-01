#!/usr/bin/env python3
"""
12_descriptives.py
=====================
Descriptive tables built from the matched India sample (see
06_sample_construction.py for how it is derived from the raw
merged_dataset_left_join.xlsx + GridExport files):

  outputs/descriptive_sector.csv       -- exits/n/exit_rate by (unpooled)
                                           Preqin primary industry, for all
                                           accepted India matches with a
                                           non-missing status.
  outputs/descriptive_continuous.csv   -- mean/median of the continuous
                                           first-round variables, by
                                           exit_any.
  outputs/India_Exit_Determinants_Analysis.csv
                                        -- a wide descriptive/audit-support
                                           table joining Preqin + LSEG fields
                                           with the constructed variables and
                                           exit_any/exit_subtype.

Ground truth to diff against (read-only):
  csv_exports/Analysis_Ready/descriptive_sector.csv
  csv_exports/Analysis_Ready/descriptive_continuous.csv
  csv_exports/Analysis_Ready/India_Exit_Determinants_Analysis.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, read_merged_left_join, ref  # noqa: E402


def build_india_matched(merged: pd.DataFrame) -> pd.DataFrame:
    firm = merged.dropna(subset=["matched_b_company_id"]).drop_duplicates("matched_b_company_id").copy()
    firm = firm.rename(columns={
        "matched_b_company_id": "preqin_company_id",
        "dataset_a_company_id": "lseg_company_id",
        "Investee Company Name": "lseg_company_name",
        "Investee Company Nation": "country",
        "Investee Company TRBC Economic Sector": "lseg_sector",
    })
    india = firm[firm["country"] == "India"].copy()
    india = india[india["preqin_company_status"].notna()]
    india["exit_any"] = (india["preqin_company_status"] == "Realised").astype(int)
    india["exit_subtype"] = np.where(india["exit_any"] == 1, "realised", "active")
    return india


def descriptive_sector(india: pd.DataFrame) -> pd.DataFrame:
    g = india.groupby(india["preqin_primary_industry"].fillna("Unknown"))
    out = g["exit_any"].agg(exits="sum", n="count").reset_index()
    out = out.rename(columns={"preqin_primary_industry": "sector"})
    out["exit_rate"] = out["exits"] / out["n"]
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def descriptive_continuous(india: pd.DataFrame, first_round_vars: pd.DataFrame) -> pd.DataFrame:
    df = india.merge(first_round_vars, on="lseg_company_name", how="left")
    cols = [
        "first_round_syndicate_size",
        "first_round_foreign_investor_share",
        "first_round_investor_experience_mean",
        "preqin_n_deal_records",
        "preqin_n_unique_investors",
    ]
    grouped = df.groupby("exit_any")[cols].agg(["mean", "median"])
    return grouped


def india_exit_determinants(india: pd.DataFrame, first_round_vars: pd.DataFrame) -> pd.DataFrame:
    df = india.merge(first_round_vars, on="lseg_company_name", how="left")
    df["flag_founding_pre2000"] = (df["preqin_year_established"] < 2000).astype("Int64")
    df["log_total_deal_size"] = np.log1p(df["preqin_total_deal_size_usd_mn"])
    keep = [
        "preqin_company_id", "preqin_company_name", "lseg_company_name", "country",
        "preqin_city", "preqin_primary_industry", "lseg_sector", "preqin_year_established",
        "flag_founding_pre2000", "preqin_company_status",
        "first_round_foreign_investor_share", "first_round_syndicate_size",
        "first_round_corporate_vc_flag", "first_round_late_stage_pe_flag",
        "first_round_investor_experience_max", "first_round_investor_experience_mean",
        "preqin_n_deal_records", "preqin_n_unique_investors", "preqin_total_deal_size_usd_mn",
        "log_total_deal_size", "preqin_max_deal_size_usd_mn", "preqin_most_recent_deal_type",
        "exit_any", "exit_subtype",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def main() -> None:
    merged = read_merged_left_join()
    india = build_india_matched(merged)
    print(f"India matched sample (non-missing status): {len(india)}")

    fr_path = Path(__file__).resolve().parent / "outputs" / "LSEG_First_Round_Variables.csv"
    if not fr_path.exists():
        raise SystemExit("Run 04_variable_construction.py first")
    first_round_vars = pd.read_csv(fr_path)

    out = outputs_dir()

    sector = descriptive_sector(india)
    sector.to_csv(out / "descriptive_sector.csv", index=False)
    print(f"Wrote {out / 'descriptive_sector.csv'} ({len(sector)} sectors)")

    cont = descriptive_continuous(india, first_round_vars)
    cont.to_csv(out / "descriptive_continuous.csv")
    print(f"Wrote {out / 'descriptive_continuous.csv'}")

    wide = india_exit_determinants(india, first_round_vars)
    wide.to_csv(out / "India_Exit_Determinants_Analysis.csv", index=False)
    print(f"Wrote {out / 'India_Exit_Determinants_Analysis.csv'} ({len(wide)} rows)")

    print("\n=== VALIDATION ===")
    gt_sector_path = ref("Analysis_Ready", "descriptive_sector.csv")
    if gt_sector_path.exists():
        gt_sector = pd.read_csv(gt_sector_path)
        common = set(sector["sector"]).intersection(gt_sector["sector"])
        print(f"descriptive_sector.csv: reconstructed n_sectors={len(sector)}  "
              f"ground_truth n_sectors={len(gt_sector)}  shared_labels={len(common)}")
        merged_check = sector.merge(gt_sector, on="sector", suffixes=("_rec", "_gt"))
        if len(merged_check):
            print(f"  n match rate: {(merged_check['n_rec'] == merged_check['n_gt']).mean():.1%}")
            print(f"  exits match rate: {(merged_check['exits_rec'] == merged_check['exits_gt']).mean():.1%}")

    gt_wide_path = ref("Analysis_Ready", "India_Exit_Determinants_Analysis.csv")
    if gt_wide_path.exists():
        gt_wide = pd.read_csv(gt_wide_path)
        print(f"India_Exit_Determinants_Analysis.csv: reconstructed N={len(wide)}  "
              f"ground_truth N={len(gt_wide)}")


if __name__ == "__main__":
    main()
