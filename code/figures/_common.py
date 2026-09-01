"""Shared paths/helpers for the figure scripts in code/figures/."""
from __future__ import annotations

import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent               # code/figures
CODE_DIR = THIS_DIR.parent                                 # code/
REPO_ROOT = CODE_DIR.parent                                 # repository root

ANALYSIS_OUTPUTS = CODE_DIR / "analysis" / "outputs"
REFERENCE_DIR = Path(os.environ.get("DISSERTATION_CSV_EXPORTS_DIR", REPO_ROOT / "csv_exports"))


def outputs_dir() -> Path:
    d = THIS_DIR / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ref(*parts: str) -> Path:
    return REFERENCE_DIR.joinpath(*parts)


def require_analysis_output(name: str) -> Path:
    p = ANALYSIS_OUTPUTS / name
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. Run the code/analysis/ pipeline first "
            f"(cd code/analysis && python3 run_all.py)."
        )
    return p
