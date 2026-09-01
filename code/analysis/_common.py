"""
_common.py
==========
Shared paths, constants and helper functions for the analysis pipeline in
code/analysis/. Importable from every 0X_*.py script in this directory.

DATA LAYOUT ASSUMED (relative to the repository root, two levels above this
file: code/analysis/_common.py -> code/ -> <repo root>):

  <repo root>/Preqin_DealPrivateEquity-01_08_2026 (1).xlsx   (raw, sheet 'Preqin_Export')
  <repo root>/Preqin_DealPrivateEquity-01_08_2026 (2).xlsx   (raw, sheet 'Preqin_Export')
  <repo root>/GridExport_July_30_2026_13_37_5.xlsx           (raw, LSEG/Refinitiv export)
  <repo root>/merged_dataset_left_join.xlsx                  (raw, prior-pass merge artifact)
  <repo root>/csv_exports/...                                (READ-ONLY ground truth, for
                                                                validation only -- never
                                                                written to)

None of the raw data files or csv_exports/ ground-truth files are copied into
this code/ directory, and nothing in this module ever writes outside each
script's own ./outputs/ directory.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # code/analysis
CODE_DIR = THIS_DIR.parent                           # code/
REPO_ROOT = CODE_DIR.parent                          # repository root (raw data lives here)

# Allow overriding the repo root / raw-data root via an environment variable,
# in case a user runs this pipeline against a copy of the raw files elsewhere
# (e.g. after cloning this code-only repo and supplying their own licensed
# Preqin/LSEG extracts).
RAW_DIR = Path(os.environ.get("DISSERTATION_RAW_DIR", REPO_ROOT))
REFERENCE_DIR = Path(os.environ.get("DISSERTATION_CSV_EXPORTS_DIR", REPO_ROOT / "csv_exports"))

PREQIN_FILE_1 = RAW_DIR / "Preqin_DealPrivateEquity-01_08_2026 (1).xlsx"
PREQIN_FILE_2 = RAW_DIR / "Preqin_DealPrivateEquity-01_08_2026 (2).xlsx"
GRIDEXPORT_FILE = RAW_DIR / "GridExport_July_30_2026_13_37_5.xlsx"
MERGED_LEFT_JOIN_FILE = RAW_DIR / "merged_dataset_left_join.xlsx"

OUTPUTS_DIRNAME = "outputs"


def outputs_dir() -> Path:
    """./outputs/ next to whichever script called this, created if needed."""
    d = THIS_DIR / OUTPUTS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def ref(*parts: str) -> Path:
    """Path to a read-only ground-truth file under csv_exports/, for validation only."""
    return REFERENCE_DIR.joinpath(*parts)


def require_file(path: Path, hint: str = "") -> Path:
    if not path.exists():
        msg = f"Required input file not found: {path}"
        if hint:
            msg += f"\n  {hint}"
        raise FileNotFoundError(msg)
    return path


# ---------------------------------------------------------------------------
# Constants shared across variable-construction / model scripts
# ---------------------------------------------------------------------------
CENSOR_DATE = pd.Timestamp("2024-12-31")

# LSEG "Firm Investor Name" placeholder / non-identity labels that must be
# excluded when counting NAMED investors (see task brief / Section 4.5 of the
# dissertation: "Undisclosed Firm" alone is ~38% of all GridExport rows).
PLACEHOLDER_INVESTOR_NAMES = {"Undisclosed Firm", "Non-Private Equity Unknown"}

# Fund Investors Stage labels that define the "late-stage PE" first-round flag.
# Reproduced verbatim from Firm_Table_Corrected's
# first_round_late_stage_pe_definition field; validated at 100% agreement
# against the historical flag on all 2,511 accepted matches (see README).
LATE_STAGE_PE_STAGES = {
    "Buyouts",
    "Fund of Funds",
    "Later Stage",
    "Mezzanine Stage",
    "Other PE/Special Situations",
    "Secondary Funds",
    "Turnaround/Distressed Debt",
}

# Conservative name-based "corporate / strategic investor" keyword list.
# This is a RECONSTRUCTION, not the original rule (the original inline code
# no longer exists). Tuned against Firm_Table_Corrected's
# first_round_corporate_vc_flag on all 2,511 accepted matches:
#   agreement = 99.5% (2499/2511), 0 false positives, 12 false negatives.
# The 12 misses were historically triggered by investor names that are NOT
# reliable corporate-VC indicators elsewhere in the ground truth (e.g. "Kotak
# Alternate Asset Managers Limited" appears in both flagged and unflagged
# companies' first rounds), so they cannot be recovered by a name-only rule
# without also memorising company-specific answers. See README "Known
# reconstruction gaps".
CORPORATE_VC_KEYWORDS = [
    "infoedge",
    "qualcomm",
    "tata capital",
    "intel capital",
    "wipro ventures",
    "amazon.com",
    "motherson",
    "infosys",
    "fosun",
    "recruit strategic partners",
]

# Sector pooling threshold used to build sector_grp for the regression sample
# (Appendix A: "Categories with at least 40 analytic firms are retained; all
# smaller categories are pooled into Other").
SECTOR_MIN_N = 40


def normalise_company_name(name: str) -> str:
    """Lower-case, strip accents/punctuation-insensitive normalised name.

    Used for the 'exact_normalised_name' matching tier: case-fold, strip
    diacritics, collapse whitespace, drop most punctuation but keep spacing
    between words (so 'Omnia Information Pvt. Ltd' and 'Omnia Information Pvt
    Ltd' normalise identically).
    """
    if pd.isna(name):
        return ""
    s = str(name)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[.,'\"]", "", s)          # drop periods/commas/quotes
    s = re.sub(r"[^a-z0-9&\s]", " ", s)    # other punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


_LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt ltd.", "pvt. ltd",
    "limited", "ltd", "llp", "inc", "incorporated", "corp", "corporation",
    "co", "company", "plc",
]


def strip_legal_suffix(normalised_name: str) -> str:
    """Remove a trailing legal-entity suffix from an already-normalised name."""
    s = normalised_name
    changed = True
    while changed:
        changed = False
        for suf in sorted(_LEGAL_SUFFIXES, key=len, reverse=True):
            pattern = r"\b" + re.escape(suf) + r"\b\s*$"
            new_s = re.sub(pattern, "", s).strip()
            if new_s != s:
                s = new_s
                changed = True
    return s.strip()


def compact_name(name: str) -> str:
    """Alphanumeric-only, no spaces -- used for the 'exact_compact_name' tier
    and as the 'compact_name_score' basis in Manual_Review.csv."""
    n = normalise_company_name(name)
    return re.sub(r"[^a-z0-9]", "", n)


def sorted_token_name(name: str) -> str:
    """Space-separated tokens sorted alphabetically -- used for the
    'exact_sorted_token_name' tier (order-invariant exact match)."""
    n = strip_legal_suffix(normalise_company_name(name))
    tokens = [t for t in n.split(" ") if t]
    return " ".join(sorted(tokens))


def read_preqin_raw() -> pd.DataFrame:
    """Load and concatenate the two raw Preqin deal-level export files."""
    require_file(PREQIN_FILE_1)
    require_file(PREQIN_FILE_2)
    d1 = pd.read_excel(PREQIN_FILE_1, sheet_name="Preqin_Export")
    d2 = pd.read_excel(PREQIN_FILE_2, sheet_name="Preqin_Export")
    d1["source_file"] = PREQIN_FILE_1.name
    d2["source_file"] = PREQIN_FILE_2.name
    return pd.concat([d1, d2], ignore_index=True)


def read_gridexport_raw() -> pd.DataFrame:
    """Load the raw LSEG/Refinitiv GridExport file with cleaned column names
    (strip embedded newlines / "('|')" suffixes added by the export tool)."""
    require_file(GRIDEXPORT_FILE)
    df = pd.read_excel(GRIDEXPORT_FILE, sheet_name=0)
    df.columns = [clean_gridexport_colname(c) for c in df.columns]
    df["Investment Date"] = pd.to_datetime(df["Investment Date"], errors="coerce")
    return df


def clean_gridexport_colname(c: str) -> str:
    c = str(c).replace("\n", " ").strip()
    c = re.sub(r"\s*\(\s*'\|'\s*\)\s*$", "", c)  # drop trailing ("'|'") markers
    c = re.sub(r"\s+", " ", c).strip()
    return c


def read_merged_left_join() -> pd.DataFrame:
    """Load the raw merged_dataset_left_join.xlsx reference/merge artifact.

    This file is a permitted RAW input (not a csv_exports/ ground-truth file):
    it is the prior-pass LSEG-left-joined-with-Preqin export and is the only
    currently available raw source that carries the Preqin company-status
    (Active/Realised) outcome, founding year, sector and city for the matched
    firms -- see README for why this is necessary.
    """
    require_file(MERGED_LEFT_JOIN_FILE)
    df = pd.read_excel(MERGED_LEFT_JOIN_FILE, sheet_name=0)
    df.columns = [clean_gridexport_colname(c) for c in df.columns]
    if "Investment Date" in df.columns:
        df["Investment Date"] = pd.to_datetime(df["Investment Date"], errors="coerce")
    return df


def log1p_safe(x):
    return np.log1p(x)
