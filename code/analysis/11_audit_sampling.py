#!/usr/bin/env python3
"""
11_audit_sampling.py
=======================
IMPORTANT -- READ THIS BEFORE USING THIS SCRIPT
--------------------------------------------------
csv_exports/Corrected_Build/Audit_Accepted_100.csv and Audit_Rejected_100.csv
record a BLINDED MANUAL AUDIT described in Section 3.1.1 of the
dissertation: 100 accepted matches and 100 rejected/manual-review candidates
were sampled, and a human reviewer then read each pair's company names,
country and other fields and judged whether it was a correct match. That
judgment ("100 of 100 audited accepted matches are judged correct"; "23 of
100 audited rejected candidates are judged to be genuine matches") is a
GENUINE, MANUAL RESEARCH STEP. IT IS NOT AN ALGORITHM, AND IT CANNOT BE
LEGITIMATELY RECONSTRUCTED AS CODE.

Consequently, this script implements ONLY the sampling step:

  1. Draw a reproducible random sample of N=100 accepted matches from the
     matched-pairs universe (matched_exact / matched_fuzzy in
     Firm_Table_Corrected.csv terms; reconstructed here from
     merged_dataset_left_join.xlsx, the raw permitted input).
  2. Draw a reproducible random sample of N=100 rejected / manual-review
     candidates (from Manual_Review.csv's candidate-generation schema).
  3. Write both samples out with an EMPTY judgment column
     (auditor_correct_match / auditor_should_match), exactly as the
     original audit worksheets look before a human fills them in.

This script contains NO code that classifies a pair as a correct or
incorrect match. Any such classification is a human research task performed
OUTSIDE this codebase. Do not add automated judgment logic to this script --
doing so would misrepresent a manual audit as an algorithmic one, which is
exactly what the task briefing for this reconstruction explicitly warned
against.

The random seed here is fixed for reproducibility of the SAMPLING draw only;
it has no bearing on, and cannot substitute for, the actual audit judgments
reported in Section 3.1.1 of the dissertation (100/100 accepted correct;
23/100 rejected candidates judged to be missed genuine matches).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, read_merged_left_join, ref  # noqa: E402

RANDOM_SEED = 42
N_ACCEPTED = 100
N_REJECTED = 100


def sample_accepted(merged: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    firm = merged.dropna(subset=["matched_b_company_id"]).drop_duplicates("matched_b_company_id")
    firm = firm[["matched_b_company_id", "dataset_a_company_id", "Investee Company Name",
                 "preqin_company_name", "match_method", "match_score",
                 "Investee Company Nation"]].copy()
    firm = firm.rename(columns={
        "matched_b_company_id": "preqin_company_id",
        "dataset_a_company_id": "grid_company_id",
        "Investee Company Name": "grid_name",
        "preqin_company_name": "preqin_name",
        "Investee Company Nation": "country_grid",
    })
    firm["country_preqin"] = firm["country_grid"]  # not independently available from this raw file
    n = min(N_ACCEPTED, len(firm))
    sample = firm.sample(n=n, random_state=rng)
    sample = sample.assign(
        audit_class="accepted",
        score=sample["match_score"],
        auditor_correct_match=np.nan,
        auditor_note=np.nan,
    )
    cols = ["audit_class", "match_method", "grid_company_id", "grid_name", "preqin_company_id",
            "preqin_name", "country_grid", "country_preqin", "score", "auditor_correct_match", "auditor_note"]
    return sample[cols].reset_index(drop=True)


def sample_rejected(rng: np.random.RandomState) -> pd.DataFrame:
    mr_path = ref("Corrected_Build", "Manual_Review.csv")
    if not mr_path.exists():
        print("  Manual_Review.csv not found (read-only ground truth); "
              "cannot demonstrate rejected-candidate sampling without a "
              "candidate-generation step upstream of this script.")
        return pd.DataFrame()
    mr = pd.read_csv(mr_path)
    n = min(N_REJECTED, len(mr))
    sample = mr.sample(n=n, random_state=rng).copy()
    sample = sample.rename(columns={
        "dataset_a_company_id": "grid_company_id",
        "dataset_a_name": "grid_name",
        "dataset_b_candidate_id": "preqin_candidate_id",
        "dataset_b_candidate_name": "preqin_candidate_name",
        "country_a": "country_grid",
        "country_b": "country_preqin",
        "composite_score": "score",
        "score_margin_best_vs_second": "margin",
    })
    sample["audit_class"] = "manual_review_candidate"
    sample["auditor_should_match"] = np.nan
    sample["auditor_note"] = np.nan
    cols = ["audit_class", "candidate_rule", "grid_company_id", "grid_name", "preqin_candidate_id",
            "preqin_candidate_name", "country_grid", "country_preqin", "score", "margin",
            "auditor_should_match", "auditor_note"]
    cols = [c for c in cols if c in sample.columns]
    return sample[cols].reset_index(drop=True)


def main() -> None:
    print(f"Drawing reproducible audit samples with random_state={RANDOM_SEED} "
          f"(sampling step ONLY -- see module docstring).")
    rng = np.random.RandomState(RANDOM_SEED)

    merged = read_merged_left_join()
    accepted = sample_accepted(merged, rng)
    rejected = sample_rejected(rng)

    out = outputs_dir()
    accepted.to_csv(out / "Audit_Accepted_100.csv", index=False)
    print(f"Wrote {out / 'Audit_Accepted_100.csv'} ({len(accepted)} rows, judgment column blank)")
    if not rejected.empty:
        rejected.to_csv(out / "Audit_Rejected_100.csv", index=False)
        print(f"Wrote {out / 'Audit_Rejected_100.csv'} ({len(rejected)} rows, judgment column blank)")

    print("\n=== STRUCTURAL comparison vs csv_exports/Corrected_Build/Audit_*_100.csv ===")
    print("(comparing schema only -- the actual sampled rows and all judgments "
          "are a separate, manual research artifact and are not reproduced here)")
    for name, df in [("Audit_Accepted_100.csv", accepted), ("Audit_Rejected_100.csv", rejected)]:
        gt_path = ref("Corrected_Build", name)
        if gt_path.exists() and not df.empty:
            gt = pd.read_csv(gt_path)
            print(f"  {name}: reconstructed columns={list(df.columns)}")
            print(f"  {name}: ground_truth columns ={list(gt.columns)}")
            print(f"  {name}: row counts reconstructed={len(df)} ground_truth={len(gt)}")


if __name__ == "__main__":
    main()
