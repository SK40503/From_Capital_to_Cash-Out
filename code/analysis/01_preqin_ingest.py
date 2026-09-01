#!/usr/bin/env python3
"""
01_preqin_ingest.py
====================
Ingest the raw Preqin "Deal Private Equity" export(s) and build:

  outputs/Preqin_Deal_Level_Raw.csv          -- the two raw exports concatenated
  outputs/Preqin_Company_Level_Derived.csv   -- one row per TARGET COMPANY ID,
                                                  with entry/exit dates and a
                                                  simple has_exit / duration
                                                  derivation from the deal-level
                                                  EXIT flag.

Ground truth to diff against (read-only, never modified):
  csv_exports/Preqin_Deal_Level/Preqin_Deal_Level_Raw.csv
  csv_exports/Preqin_Deal_Level/Preqin_Company_Level_Derived.csv

IMPORTANT SCOPE CAVEAT -- read before trusting downstream numbers
-------------------------------------------------------------------
The two raw Preqin xlsx files bundled in this repository
("Preqin_DealPrivateEquity-01_08_2026 (1).xlsx" and "(2).xlsx") together
contain only 940 deal-level rows covering 557 unique companies across India,
Pakistan and Bangladesh combined. This is NOT the same population as the
8,186-company Preqin master that Firm_Table_Corrected.csv and
India_Exit_Determinants_Analysis.csv were built from: cross-checking company
names shows only 4 of the 2,506 India firms in India_Exit_Determinants_Analysis.csv
also appear in these two raw deal files. In other words, these are a smaller /
different Preqin pull (likely a buyout/large-deal screen) than whatever
extract originally produced the firm-level Preqin master (with founding year,
sector, city, website and company status) used for entity matching and
modelling -- and that original extract is not present anywhere in this
repository.

This is not a new problem introduced by this reconstruction: Corrected_Build/
Diagnostics.csv itself already flags "Master count: UNRESOLVED SOURCE
SCOPE -- 8,186 observed vs A2 target ~7,691 ... Recover A2 extraction
rule/source list before final sample declaration", i.e. even the original
analyst had lost track of exactly how the 8,186-row master was pulled.

Consequently:
  * This script (01) faithfully reproduces the deal-level/company-level
    aggregation logic on the data that IS available, and validates tightly
    against Preqin_Deal_Level_Raw.csv / Preqin_Company_Level_Derived.csv
    (which were themselves clearly built from exactly these two files -- row
    counts match exactly).
  * It is NOT used later in the pipeline to build the entity-matching master
    or the modelling sample, because it does not cover the right company
    population. Scripts 04/06 onward instead take the outcome/founding
    year/sector fields for the matched sample from merged_dataset_left_join.xlsx,
    which IS a raw, permitted input file that already carries these Preqin
    company-level fields for the 2,506-2,511 matched Indian/Pakistani/
    Bangladeshi firms (see 06_sample_construction.py docstring for the full
    explanation and the exact validation that this substitution is correct).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CENSOR_DATE, outputs_dir, read_preqin_raw, ref  # noqa: E402


def build_deal_level(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.copy()


def build_company_level(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["DEAL DATE"] = pd.to_datetime(raw["DEAL DATE"], errors="coerce")
    raw = raw.sort_values(["TARGET COMPANY ID", "DEAL DATE"])

    rows = []
    for cid, g in raw.groupby("TARGET COMPANY ID", sort=False):
        g = g.sort_values("DEAL DATE")
        first = g.iloc[0]
        entry_date = g["DEAL DATE"].min()
        n_deal_rows = len(g)

        exit_rows = g[g["EXIT"] == "Yes"]
        has_exit = len(exit_rows) > 0
        exit_date = np.nan
        exit_type = np.nan
        duration_years = np.nan
        censor_reason = np.nan
        entry_is_only_row = n_deal_rows == 1

        if entry_is_only_row and has_exit:
            # Entry and exit coincide on the single available row; no
            # meaningful duration can be computed.
            exit_date = exit_rows["DEAL DATE"].iloc[0]
            exit_type = exit_rows["DEAL TYPES"].iloc[0]
            censor_reason = "exit_row_only_no_prior_entry_date"
            duration_years = np.nan
        elif has_exit:
            first_exit = exit_rows.sort_values("DEAL DATE").iloc[0]
            exit_date = first_exit["DEAL DATE"]
            exit_type = first_exit["DEAL TYPES"]
            duration_years = (exit_date - entry_date).days / 365.25
        else:
            censor_reason = "censored_at_study_end_or_last_observed"
            duration_years = (CENSOR_DATE - entry_date).days / 365.25

        overlaps_window = bool(pd.notna(entry_date) and 2014 <= entry_date.year <= 2024)

        rows.append(
            {
                "target_company_id": cid,
                "target_company_name": first["TARGET COMPANY"],
                "country": first["TARGET COMPANY COUNTRY"],
                "strategy": first["STRATEGY"],
                "primary_industry": first["PRIMARY INDUSTRY"],
                "n_deal_rows": n_deal_rows,
                "entry_date": entry_date,
                "has_exit": has_exit,
                "exit_date": exit_date,
                "exit_type": exit_type,
                "duration_years": duration_years,
                "censor_reason": censor_reason,
                "entry_is_only_row": entry_is_only_row,
                "overlaps_2014_2024_window": overlaps_window,
            }
        )
    return pd.DataFrame(rows)


def validate(deal_level: pd.DataFrame, company_level: pd.DataFrame) -> None:
    print("\n=== VALIDATION vs csv_exports/Preqin_Deal_Level/ ===")
    gt_deal_path = ref("Preqin_Deal_Level", "Preqin_Deal_Level_Raw.csv")
    gt_comp_path = ref("Preqin_Deal_Level", "Preqin_Company_Level_Derived.csv")
    if gt_deal_path.exists():
        gt_deal = pd.read_csv(gt_deal_path)
        print(f"deal-level rows: reconstructed={len(deal_level)}  ground_truth={len(gt_deal)}"
              f"  match={len(deal_level) == len(gt_deal)}")
    else:
        print("ground truth deal-level file not found; skipping row-count check")

    if gt_comp_path.exists():
        gt_comp = pd.read_csv(gt_comp_path)
        print(f"company-level rows: reconstructed={len(company_level)}  ground_truth={len(gt_comp)}"
              f"  match={len(company_level) == len(gt_comp)}")
        print(f"has_exit True count: reconstructed={int(company_level['has_exit'].sum())}"
              f"  ground_truth={int(gt_comp['has_exit'].sum())}")
        rec_dur = company_level["duration_years"].dropna()
        gt_dur = gt_comp["duration_years"].dropna()
        print(f"duration_years mean: reconstructed={rec_dur.mean():.4f}  ground_truth={gt_dur.mean():.4f}")
        rec_exit_types = company_level["exit_type"].value_counts(dropna=True).sort_index()
        gt_exit_types = gt_comp["exit_type"].value_counts(dropna=True).sort_index()
        print("exit_type distribution matches:", rec_exit_types.equals(gt_exit_types))
        if not rec_exit_types.equals(gt_exit_types):
            print("  reconstructed:\n", rec_exit_types)
            print("  ground_truth:\n", gt_exit_types)
    else:
        print("ground truth company-level file not found; skipping detailed checks")


def main() -> None:
    print("Loading raw Preqin deal-level export(s) ...")
    raw = read_preqin_raw()
    print(f"  {len(raw)} deal rows loaded from 2 files "
          f"({raw['TARGET COMPANY ID'].nunique()} unique companies, "
          f"countries={sorted(raw['TARGET COMPANY COUNTRY'].dropna().unique())})")

    deal_level = build_deal_level(raw)
    company_level = build_company_level(raw)

    out = outputs_dir()
    deal_level.to_csv(out / "Preqin_Deal_Level_Raw.csv", index=False)
    company_level.to_csv(out / "Preqin_Company_Level_Derived.csv", index=False)
    print(f"\nWrote {out / 'Preqin_Deal_Level_Raw.csv'}")
    print(f"Wrote {out / 'Preqin_Company_Level_Derived.csv'}")

    validate(deal_level, company_level)

    print(
        "\nNOTE: see this script's module docstring -- these two raw Preqin "
        "files cover a different, smaller company population than the "
        "8,186-firm master used to build Firm_Table_Corrected.csv, so this "
        "output is NOT used downstream for entity matching or modelling."
    )


if __name__ == "__main__":
    main()
