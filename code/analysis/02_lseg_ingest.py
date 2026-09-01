#!/usr/bin/env python3
"""
02_lseg_ingest.py
==================
Ingest the raw LSEG (Refinitiv) GridExport file and build:

  outputs/GridExport_Clean.csv     -- the raw export with cleaned column names
                                       (newlines / "('|')" suffixes stripped),
                                       otherwise a full passthrough of all
                                       32,154 rows.
  outputs/LSEG_Company_Summary.csv -- one row per Investee Company Name with
                                       basic coverage stats (n rounds, n
                                       distinct investors, first/last
                                       investment date), used as a sanity
                                       check ahead of 04_variable_construction.py.

Ground truth to diff against (read-only):
  csv_exports/GridExport_Refinitiv/Current_Screen_Template.csv
    -- this is effectively a saved copy of the same raw GridExport pull
       (identical column set, same row count), so this script validates a
       near-exact match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, read_gridexport_raw, ref  # noqa: E402


def build_company_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Investee Company Name", dropna=True)
    summary = g.agg(
        n_rounds=("Investment Date", "size"),
        n_distinct_investors=("Firm Investor Name", "nunique"),
        first_investment_date=("Investment Date", "min"),
        last_investment_date=("Investment Date", "max"),
        nation=("Investee Company Nation", "first"),
        trbc_sector=("Investee Company TRBC Economic Sector", "first"),
    ).reset_index()
    return summary


def validate(df: pd.DataFrame) -> None:
    print("\n=== VALIDATION vs csv_exports/GridExport_Refinitiv/Current_Screen_Template.csv ===")
    gt_path = ref("GridExport_Refinitiv", "Current_Screen_Template.csv")
    if not gt_path.exists():
        print("ground truth file not found; skipping")
        return
    gt = pd.read_csv(gt_path, low_memory=False)
    print(f"rows: reconstructed={len(df)}  ground_truth={len(gt)}  match={len(df) == len(gt)}")
    print(f"unique companies: reconstructed={df['Investee Company Name'].nunique()}"
          f"  ground_truth={gt['Investee Company Name'].nunique()}")
    print(f"unique investors: reconstructed={df['Firm Investor Name'].nunique()}"
          f"  ground_truth={gt['Firm Investor Name'].nunique()}")
    pct_undisclosed = (df["Firm Investor Name"] == "Undisclosed Firm").mean()
    print(f"share of rows with 'Undisclosed Firm' investor: {pct_undisclosed:.1%} "
          f"(dissertation Section 4.5 cites 38.4%)")


def main() -> None:
    print("Loading raw GridExport (LSEG/Refinitiv) file ...")
    df = read_gridexport_raw()
    print(f"  {len(df)} rows, {df['Investee Company Name'].nunique()} unique companies")

    out = outputs_dir()
    df.to_csv(out / "GridExport_Clean.csv", index=False)
    print(f"\nWrote {out / 'GridExport_Clean.csv'}")

    summary = build_company_summary(df)
    summary.to_csv(out / "LSEG_Company_Summary.csv", index=False)
    print(f"Wrote {out / 'LSEG_Company_Summary.csv'} ({len(summary)} companies)")

    validate(df)


if __name__ == "__main__":
    main()
