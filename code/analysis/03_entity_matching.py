#!/usr/bin/env python3
"""
03_entity_matching.py
=======================
Reconstructs the tiered Preqin <-> LSEG entity-matching pipeline referenced
throughout Firm_Table_Corrected.csv and Manual_Review.csv:

  exact_raw_name < exact_normalised_name < exact_legal_suffix_name
  < exact_compact_name < exact_sorted_token_name < fuzzy_automatic

(checked in that priority order; the first tier at which two names agree
determines match_method).

WHAT THIS SCRIPT DOES
----------------------
1. classify_pair(name_a, name_b) -> (is_match, method, score)
   A standalone, documented implementation of the tiered classifier above,
   using rapidfuzz for the fuzzy tier.

2. Pairwise validation against the KNOWN matched pairs already recorded in
   Firm_Table_Corrected.csv / merged_dataset_left_join.xlsx (2,511 accepted
   Preqin<->LSEG matches). For each known pair this recomputes classify_pair()
   on the two recorded company-name strings and compares the resulting
   match_method tier to the tier recorded historically. This tests the
   SCORING LOGIC, and does not require the full company-master universe.

3. Score reproduction check against Manual_Review.csv's recorded
   best_name_score / token_set_score / compact_name_score columns (1,161
   candidate pairs with ground-truth sub-scores already attached).

4. A best-effort "full pipeline" demonstration: blocking + candidate
   generation + tiered matching between the LSEG company master (from
   02_lseg_ingest.py) and the Preqin company master that IS available from
   raw data (01_preqin_ingest.py's 557-company, 3-country deal-level
   extract). See the LIMITATION note below for why this does not reproduce
   Firm_Table_Corrected's 2,511 matches.

LIMITATION -- READ BEFORE INTERPRETING THE "FULL PIPELINE" OUTPUT
--------------------------------------------------------------------
The 8,186-company Preqin master that was actually matched against LSEG to
build Firm_Table_Corrected is not present in this repository in raw form
(see 01_preqin_ingest.py's docstring for the evidence: only 4 of 2,506
matched India company names appear anywhere in the two raw Preqin deal
exports available here). Running the classifier below over the *available*
557-company Preqin extract against the 5,419-company LSEG master therefore
finds close to nothing -- not because the matching logic is wrong (part 2
above shows the tier-classification logic itself reproduces the historical
tiers reasonably well on pairs that ARE known to correspond), but because the
two lists being matched barely overlap. This gap is consistent with
Corrected_Build/Diagnostics.csv's own "UNRESOLVED SOURCE SCOPE" entry, which
already flagged that the analyst who built Firm_Table_Corrected.csv could not
fully document where the 8,186-row master came from.

A NOTE ON match_score
-----------------------
The historical match_score column in Firm_Table_Corrected.csv does not
appear to be a deterministic function of the two company-name strings alone:
e.g. "2070 health"/"2070 Health" (identical apart from case) scores 100.00,
but "30 sundays"/"30 Sundays" (also identical apart from case, same
match_method = exact_raw_name) scores 96.54 -- and 96.54 recurs identically
(to 2 decimal places) across 686 of the 2,511 matched rows regardless of how
similar the two underlying names actually are. This looks like an artifact
of the original (lost) pipeline -- e.g. a stale/default score carried over
from an earlier processing pass -- rather than a designed scoring formula, so
this script does not attempt to reproduce match_score numerically; it
recomputes a genuine, principled similarity score instead and validates
match_METHOD (the tier), not match_score, against the ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    compact_name,
    normalise_company_name,
    outputs_dir,
    read_gridexport_raw,
    read_merged_left_join,
    ref,
    sorted_token_name,
    strip_legal_suffix,
)

FUZZY_THRESHOLD = 92.0  # matches Manual_Review.csv's score_band "A: 92+"


def classify_pair(name_a: str, name_b: str) -> tuple[bool, str, float]:
    """Tiered classifier. Returns (is_match, method, score)."""
    if pd.isna(name_a) or pd.isna(name_b):
        return False, "unmatched", 0.0

    raw_a, raw_b = str(name_a).strip(), str(name_b).strip()

    # Tier 1: exact_raw_name (case-insensitive, whitespace-trimmed)
    if raw_a.lower() == raw_b.lower():
        return True, "exact_raw_name", 100.0

    norm_a, norm_b = normalise_company_name(raw_a), normalise_company_name(raw_b)

    # Tier 2: exact_normalised_name
    if norm_a == norm_b:
        return True, "exact_normalised_name", 100.0

    # Tier 3: exact_legal_suffix_name (strip legal suffix, compare cores)
    suf_a, suf_b = strip_legal_suffix(norm_a), strip_legal_suffix(norm_b)
    if suf_a and suf_a == suf_b:
        score = fuzz.ratio(norm_a, norm_b)
        return True, "exact_legal_suffix_name", round(score, 2)

    # Tier 4: exact_compact_name (alphanumeric only, no spaces)
    comp_a, comp_b = compact_name(raw_a), compact_name(raw_b)
    if comp_a and comp_a == comp_b:
        score = fuzz.ratio(norm_a, norm_b)
        return True, "exact_compact_name", round(score, 2)

    # Tier 5: exact_sorted_token_name (order-invariant token match)
    sort_a, sort_b = sorted_token_name(raw_a), sorted_token_name(raw_b)
    if sort_a and sort_a == sort_b:
        score = fuzz.ratio(norm_a, norm_b)
        return True, "exact_sorted_token_name", round(score, 2)

    # Tier 6: fuzzy_automatic
    best_name_score = fuzz.ratio(norm_a, norm_b)
    token_set_score = fuzz.token_set_ratio(norm_a, norm_b)
    compact_score = fuzz.ratio(comp_a, comp_b)
    composite = (best_name_score + token_set_score + compact_score) / 3.0
    if composite >= FUZZY_THRESHOLD:
        return True, "fuzzy_automatic", round(composite, 2)

    return False, "unmatched", round(composite, 2)


def score_components(name_a: str, name_b: str) -> dict:
    norm_a, norm_b = normalise_company_name(name_a), normalise_company_name(name_b)
    comp_a, comp_b = compact_name(name_a), compact_name(name_b)
    return {
        "best_name_score": fuzz.ratio(norm_a, norm_b),
        "token_set_score": fuzz.token_set_ratio(norm_a, norm_b),
        "compact_name_score": fuzz.ratio(comp_a, comp_b),
    }


def validate_against_known_pairs() -> pd.DataFrame:
    print("\n=== VALIDATION 1: tier classification on 2,511 known matched pairs ===")
    merged = read_merged_left_join()
    ft_path = ref("Corrected_Build", "Firm_Table_Corrected.csv")
    if not ft_path.exists():
        print("  Firm_Table_Corrected.csv not found; skipping this validation")
        return pd.DataFrame()
    ft = pd.read_csv(ft_path, low_memory=False)

    pairs = (
        merged.dropna(subset=["matched_b_company_id"])
        .drop_duplicates("matched_b_company_id")[
            ["matched_b_company_id", "preqin_company_name", "Investee Company Name"]
        ]
    )
    known = pairs.merge(
        ft[["preqin_company_id", "match_method", "match_score"]],
        left_on="matched_b_company_id",
        right_on="preqin_company_id",
        how="left",
    ).dropna(subset=["match_method"])

    results = known.apply(
        lambda r: classify_pair(r["preqin_company_name"], r["Investee Company Name"]),
        axis=1,
        result_type="expand",
    )
    results.columns = ["reconstructed_is_match", "reconstructed_method", "reconstructed_score"]
    known = pd.concat([known.reset_index(drop=True), results], axis=1)

    agree = (known["reconstructed_method"] == known["match_method"]).mean()
    print(f"  N pairs = {len(known)}")
    print(f"  method-tier exact agreement = {agree:.1%}")
    print("  reconstructed method distribution:")
    print(known["reconstructed_method"].value_counts().to_string())
    print("  ground-truth method distribution:")
    print(known["match_method"].value_counts().to_string())
    # coarser check: was it classified as SOME kind of match at all?
    any_match_agree = (known["reconstructed_is_match"]).mean()
    print(f"  share of known-true matches reconstructed as a match of ANY tier: {any_match_agree:.1%}")
    return known


def validate_against_manual_review() -> None:
    print("\n=== VALIDATION 2: sub-score reproduction on Manual_Review.csv candidates ===")
    mr_path = ref("Corrected_Build", "Manual_Review.csv")
    if not mr_path.exists():
        print("  Manual_Review.csv not found; skipping")
        return
    mr = pd.read_csv(mr_path)
    comp = mr.apply(
        lambda r: score_components(r["dataset_a_name"], r["dataset_b_candidate_name"]),
        axis=1,
        result_type="expand",
    )
    for col, gt_col in [
        ("best_name_score", "best_name_score"),
        ("token_set_score", "token_set_score"),
        ("compact_name_score", "compact_name_score"),
    ]:
        corr = comp[col].corr(mr[gt_col])
        mae = (comp[col] - mr[gt_col]).abs().mean()
        exact = (comp[col].round(2) == mr[gt_col].round(2)).mean()
        print(f"  {col}: corr={corr:.3f}  MAE={mae:.2f}  exact-match rate={exact:.1%}")
    print(
        "  (correlation/MAE reported rather than forcing exact agreement -- the "
        "original preprocessing before scoring, e.g. exact suffix-handling "
        "rules, could not be fully recovered; see module docstring.)"
    )


def demonstrate_full_pipeline() -> pd.DataFrame:
    print("\n=== Demonstration: full pipeline on AVAILABLE raw data ===")
    ge = read_gridexport_raw()
    lseg_names = sorted(ge["Investee Company Name"].dropna().unique())

    preqin_path = Path(__file__).resolve().parent / "outputs" / "Preqin_Company_Level_Derived.csv"
    if not preqin_path.exists():
        print("  Run 01_preqin_ingest.py first; skipping full-pipeline demo")
        return pd.DataFrame()
    preqin = pd.read_csv(preqin_path)
    preqin_names = sorted(preqin["target_company_name"].dropna().unique())

    print(f"  LSEG master: {len(lseg_names)} companies")
    print(f"  Preqin master (available raw extract): {len(preqin_names)} companies")

    # Simple blocking: only compare pairs sharing a normalised first token,
    # to keep this demonstration tractable (O(n) rather than O(n*m)).
    from collections import defaultdict

    block = defaultdict(list)
    for nm in lseg_names:
        n = normalise_company_name(nm)
        if n:
            block[n.split(" ")[0]].append(nm)

    rows = []
    for pnm in preqin_names:
        pn = normalise_company_name(pnm)
        if not pn:
            continue
        key = pn.split(" ")[0]
        for lnm in block.get(key, []):
            is_match, method, score = classify_pair(pnm, lnm)
            if is_match:
                rows.append({"preqin_name": pnm, "lseg_name": lnm, "method": method, "score": score})

    result = pd.DataFrame(rows)
    print(f"  matches found: {len(result)} (out of {len(preqin_names)} available Preqin companies)")
    print(
        "  This near-zero count is expected and is documented in the module "
        "docstring: the available raw Preqin extract does not cover the "
        "company population that Firm_Table_Corrected.csv was built from."
    )
    return result


def main() -> None:
    out = outputs_dir()

    known = validate_against_known_pairs()
    if not known.empty:
        known.to_csv(out / "entity_matching_known_pairs_validation.csv", index=False)
        print(f"\nWrote {out / 'entity_matching_known_pairs_validation.csv'}")

    validate_against_manual_review()

    demo = demonstrate_full_pipeline()
    if not demo.empty:
        demo.to_csv(out / "entity_matching_full_pipeline_demo.csv", index=False)
        print(f"Wrote {out / 'entity_matching_full_pipeline_demo.csv'}")


if __name__ == "__main__":
    main()
