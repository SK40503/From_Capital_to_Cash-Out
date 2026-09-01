#!/usr/bin/env python3
"""
04_variable_construction.py
=============================
Computes first-round explanatory variables directly from the raw LSEG
GridExport file, for every company that appears in it:

  first_round_syndicate_size            -- distinct Firm Investor Name values
                                            among rows at that company's
                                            earliest Investment Date
  first_round_foreign_investor_share    -- share of DISTINCT first-round
                                            investor names for which at least
                                            one row records a
                                            Firm Investors Nation != 'India'
                                            (denominator excludes investors
                                            with no non-null nation at all)
  first_round_investor_experience_mean  -- mean, over first-round investor
                                            ROWS, of 'Total Number of
                                            Companies Invested in by Fund
                                            Investor'
  first_round_investor_experience_max   -- max of the same field
  first_round_corporate_vc_flag         -- 1 if ANY first-round investor name
                                            matches the CORPORATE_VC_KEYWORDS
                                            list (see _common.py)
  first_round_late_stage_pe_flag        -- 1 if ANY first-round investor's
                                            Fund Investors Stage is in
                                            LATE_STAGE_PE_STAGES

These definitions were reverse-engineered from raw data and validated
row-by-row against Firm_Table_Corrected.csv (see VALIDATION output below):
syndicate size, foreign share, experience mean and experience max reproduce
the historical values EXACTLY for every one of the 2,511 accepted matches
(mean absolute difference = 0.0). The late-stage PE flag also reproduces
exactly (100% agreement). The corporate-VC flag is the one variable that is
a genuine reconstruction rather than a rediscovered formula -- see
CORPORATE_VC_KEYWORDS in _common.py for the accuracy figures (99.5% row
agreement, 0 false positives, 12 false negatives out of 2,511).

INPUT
-----
Raw GridExport file only (via 02_lseg_ingest.py's output, or read fresh).
The company identity crosswalk used purely for VALIDATION (matching an LSEG
company name back to its historical flags) comes from
csv_exports/Corrected_Build/Firm_Table_Corrected.csv, which is read-only
ground truth and is never written to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CORPORATE_VC_KEYWORDS,
    LATE_STAGE_PE_STAGES,
    outputs_dir,
    read_gridexport_raw,
    ref,
)

NAME_COL = "Investee Company Name"
DATE_COL = "Investment Date"
INV_COL = "Firm Investor Name"
NATION_COL = "Firm Investors Nation"
EXPERIENCE_COL = "Total Number of Companies Invested in by Fund Investor"
STAGE_COL = "Fund Investors Stage"


def is_corporate_vc(investor_name: str) -> bool:
    if pd.isna(investor_name):
        return False
    n = str(investor_name).lower()
    return any(k in n for k in CORPORATE_VC_KEYWORDS)


def build_first_round_variables(ge: pd.DataFrame) -> pd.DataFrame:
    ge = ge.copy()
    ge[DATE_COL] = pd.to_datetime(ge[DATE_COL], errors="coerce")
    first_date = ge.groupby(NAME_COL)[DATE_COL].transform("min")
    first_round = ge[ge[DATE_COL] == first_date].copy()

    rows = []
    for company, g in first_round.groupby(NAME_COL, sort=False):
        syndicate_size = g[INV_COL].nunique(dropna=True)

        # Foreign share is computed over DISTINCT first-round investor NAMES
        # (like syndicate size), not over raw rows: an investor counts as
        # "foreign" if ANY of its first-round rows records a non-India
        # nation, and is included in the denominator if it has at least one
        # non-null nation value. This (rather than a simple row-level share)
        # is what reproduces Firm_Table_Corrected's recorded values exactly
        # -- e.g. a repeated 'Undisclosed Firm' slot with mixed India/foreign
        # /missing nation values across its rows is counted once.
        by_investor = g.groupby(INV_COL)[NATION_COL].apply(
            lambda s: s.dropna().ne("India").any() if s.notna().any() else np.nan
        )
        by_investor = by_investor.dropna()
        foreign_share = by_investor.mean() if len(by_investor) else np.nan

        exp = g[EXPERIENCE_COL].dropna()
        exp_mean = exp.mean() if len(exp) else np.nan
        exp_max = exp.max() if len(exp) else np.nan

        corp_vc_flag = int(g[INV_COL].apply(is_corporate_vc).any())
        late_stage_flag = int(g[STAGE_COL].isin(LATE_STAGE_PE_STAGES).any())

        rows.append(
            {
                "lseg_company_name": company,
                "first_round_date": g[DATE_COL].iloc[0],
                "first_round_syndicate_size": syndicate_size,
                "first_round_foreign_investor_share": foreign_share,
                "first_round_investor_experience_mean": exp_mean,
                "first_round_investor_experience_max": exp_max,
                "first_round_corporate_vc_flag": corp_vc_flag,
                "first_round_late_stage_pe_flag": late_stage_flag,
            }
        )
    return pd.DataFrame(rows)


def validate(built: pd.DataFrame) -> None:
    print("\n=== VALIDATION vs csv_exports/Corrected_Build/Firm_Table_Corrected.csv ===")
    ft_path = ref("Corrected_Build", "Firm_Table_Corrected.csv")
    if not ft_path.exists():
        print("ground truth file not found; skipping")
        return
    ft = pd.read_csv(ft_path, low_memory=False)
    matched = ft[ft["match_status"].isin(["matched_exact", "matched_fuzzy"])].copy()

    built_r = built.rename(columns={c: f"{c}__rec" for c in built.columns if c != "lseg_company_name"})
    cmp = matched.merge(built_r, on="lseg_company_name", how="left")
    n = len(cmp)
    print(f"N matched companies compared: {n} (of {len(matched)} in Firm_Table_Corrected)")

    checks = [
        ("first_round_syndicate_size", "first_round_syndicate_size__rec"),
        ("first_round_foreign_investor_share", "first_round_foreign_investor_share__rec"),
        ("first_round_investor_experience_mean", "first_round_investor_experience_mean__rec"),
        ("first_round_lead_investor_experience_proxy_max", "first_round_investor_experience_max__rec"),
        ("first_round_late_stage_pe_flag", "first_round_late_stage_pe_flag__rec"),
        ("first_round_corporate_vc_flag", "first_round_corporate_vc_flag__rec"),
    ]
    for gt_col, rec_col in checks:
        gt = cmp[gt_col]
        rec = cmp[rec_col]
        valid = gt.notna() & rec.notna()
        exact = np.isclose(gt[valid], rec[valid], atol=1e-6).mean() if valid.sum() else float("nan")
        mae = (gt[valid] - rec[valid]).abs().mean() if valid.sum() else float("nan")
        print(f"  {gt_col}: n_valid={valid.sum()}  exact_match_rate={exact:.1%}  MAE={mae:.4f}")


def main() -> None:
    print("Loading raw GridExport and computing first-round variables ...")
    ge = read_gridexport_raw()
    built = build_first_round_variables(ge)
    print(f"  computed variables for {len(built)} LSEG companies")

    out = outputs_dir()
    built.to_csv(out / "LSEG_First_Round_Variables.csv", index=False)
    print(f"\nWrote {out / 'LSEG_First_Round_Variables.csv'}")

    validate(built)


if __name__ == "__main__":
    main()
