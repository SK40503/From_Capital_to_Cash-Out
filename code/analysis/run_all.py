#!/usr/bin/env python3
"""
run_all.py
============
Runs every numbered script in code/analysis/ in order, in-process, so a
single `python3 run_all.py` regenerates everything into each script's own
./outputs/ directory. Stops and reports clearly if a script fails, rather
than silently continuing with stale downstream outputs.

Usage:
    cd code/analysis
    python3 run_all.py
"""
from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

SCRIPTS = [
    "01_preqin_ingest.py",
    "02_lseg_ingest.py",
    "03_entity_matching.py",
    "04_variable_construction.py",
    "05_pre_round_experience.py",
    "06_sample_construction.py",
    "07_main_model.py",
    "08_sensitivity_models.py",
    "09_vif_diagnostics.py",
    "10_prerounds_model.py",
    "11_audit_sampling.py",
    "12_descriptives.py",
    "13_full_diagnostics_and_variants.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        path = here / script
        print("\n" + "=" * 80)
        print(f"RUNNING {script}")
        print("=" * 80)
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0):
                print(f"\n{script} exited with code {e.code}; stopping run_all.py.")
                sys.exit(e.code)
        except Exception:
            print(f"\n{script} raised an exception; stopping run_all.py.")
            traceback.print_exc()
            sys.exit(1)
    print("\n" + "=" * 80)
    print("All analysis scripts completed. See each script's ./outputs/ directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
