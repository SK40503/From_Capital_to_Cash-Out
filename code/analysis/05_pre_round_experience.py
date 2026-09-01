#!/usr/bin/env python3
"""
05_pre_round_experience.py
=============================
Reconstructs the "pre-round-only" investor-experience measure described in
Section 4.5 / Table 1 (Table 2 concept row "Investor experience (pre-round
reconstruction)") of the dissertation, computed entirely from the raw
GridExport file.

For every LSEG company:
  1. Find its first observed round (earliest Investment Date, over ALL
     investor rows -- including placeholder rows -- exactly as in
     04_variable_construction.py, so first_round_date and
     first_round_syndicate_size_recomputed are directly comparable to that
     script's outputs).
  2. Take the NAMED first-round investors only, i.e. excluding the literal
     placeholder labels 'Undisclosed Firm' and 'Non-Private Equity Unknown'
     (PLACEHOLDER_INVESTOR_NAMES in _common.py). 'Undisclosed Firm' alone
     accounts for ~38% of all GridExport investor-slots and is a generic
     non-identity, not a real investor -- see 02_lseg_ingest.py's validation
     output.
  3. Companies whose first round has NO named investor at all are dropped
     (this matches the dissertation's "dropped the 447 companies (8.2%) ...
     whose first round involved no named investor").
  4. For each remaining named investor, count the number of DISTINCT
     Investee Company Name values that investor is recorded against in the
     FULL GridExport file with Investment Date STRICTLY BEFORE the focal
     company's first-round date (i.e. pre-round-only, to avoid post-outcome
     contamination of the experience measure).
  5. Average and max that count across the round's named investors, per
     focal company.

Ground truth to diff against (read-only):
  csv_exports/Analysis_Ready/pre_round_experience.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PLACEHOLDER_INVESTOR_NAMES, outputs_dir, read_gridexport_raw, ref  # noqa: E402

NAME_COL = "Investee Company Name"
DATE_COL = "Investment Date"
INV_COL = "Firm Investor Name"


def build_pre_round_experience(ge: pd.DataFrame) -> pd.DataFrame:
    ge = ge.copy()
    ge[DATE_COL] = pd.to_datetime(ge[DATE_COL], errors="coerce")

    first_date = ge.groupby(NAME_COL)[DATE_COL].transform("min")
    first_round = ge[ge[DATE_COL] == first_date]

    # Pre-index: for each investor, the sorted array of investment dates and
    # the company name at each date, so "distinct companies before date X"
    # can be computed without an O(n^2) scan per company.
    inv_groups = {
        inv: g[[DATE_COL, NAME_COL]].sort_values(DATE_COL)
        for inv, g in ge.groupby(INV_COL, sort=False)
    }

    def n_distinct_companies_before(investor: str, cutoff: pd.Timestamp) -> int:
        g = inv_groups.get(investor)
        if g is None:
            return 0
        prior = g[g[DATE_COL] < cutoff]
        return prior[NAME_COL].nunique()

    rows = []
    for company, g in first_round.groupby(NAME_COL, sort=False):
        focal_date = g[DATE_COL].iloc[0]
        syndicate_size_recomputed = g[INV_COL].nunique(dropna=True)

        named = [inv for inv in g[INV_COL].dropna().unique() if inv not in PLACEHOLDER_INVESTOR_NAMES]
        n_named = len(named)
        if n_named == 0:
            continue  # dropped: first round has no named investor

        exp_counts = [n_distinct_companies_before(inv, focal_date) for inv in named]
        rows.append(
            {
                "company": company,
                "first_round_date": focal_date,
                "n_named_first_round_investors": n_named,
                "pre_round_exp_mean": float(np.mean(exp_counts)),
                "pre_round_exp_max": int(np.max(exp_counts)),
                "first_round_syndicate_size_recomputed": syndicate_size_recomputed,
            }
        )
    return pd.DataFrame(rows)


def validate(built: pd.DataFrame) -> None:
    print("\n=== VALIDATION vs csv_exports/Analysis_Ready/pre_round_experience.csv ===")
    gt_path = ref("Analysis_Ready", "pre_round_experience.csv")
    if not gt_path.exists():
        print("ground truth file not found; skipping")
        return
    gt = pd.read_csv(gt_path)
    print(f"rows: reconstructed={len(built)}  ground_truth={len(gt)}  match={len(built) == len(gt)}")

    cmp = gt.merge(built, on="company", how="left", suffixes=("_gt", "_rec"))
    for col in ["n_named_first_round_investors", "pre_round_exp_mean", "pre_round_exp_max",
                "first_round_syndicate_size_recomputed"]:
        gt_c = cmp[f"{col}_gt"] if f"{col}_gt" in cmp.columns else cmp[col]
        rec_c = cmp[f"{col}_rec"] if f"{col}_rec" in cmp.columns else cmp[col]
        valid = gt_c.notna() & rec_c.notna()
        exact = np.isclose(gt_c[valid], rec_c[valid], atol=1e-6).mean() if valid.sum() else float("nan")
        mae = (gt_c[valid] - rec_c[valid]).abs().mean() if valid.sum() else float("nan")
        print(f"  {col}: n_valid={int(valid.sum())}  exact_match_rate={exact:.1%}  MAE={mae:.4f}")


def main() -> None:
    print("Loading raw GridExport and computing pre-round-only investor experience ...")
    ge = read_gridexport_raw()
    built = build_pre_round_experience(ge)
    print(f"  computed for {len(built)} companies "
          f"(dropped companies whose first round had no named investor)")

    out = outputs_dir()
    built.to_csv(out / "pre_round_experience.csv", index=False)
    print(f"\nWrote {out / 'pre_round_experience.csv'}")

    validate(built)


if __name__ == "__main__":
    main()
