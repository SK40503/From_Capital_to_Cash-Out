#!/usr/bin/env python3
"""
04_fig4_experience_reversal.py
=================================
Figure 4 (dissertation, Section 4.5): comparison of the investor-experience
coefficient/effect under the current-snapshot construction vs the
pre-round-only reconstruction (the "reversal" finding).

A prior version of this figure had title-overflow issues; fixed here with a
wider figsize and a two-line wrapped title.

Input: code/analysis/outputs/{main_model_results.csv,
       prerounds_experience_comparison.csv}
       (run 07_main_model.py and 10_prerounds_model.py first)
Output: outputs/fig4_experience_reversal.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ANALYSIS_OUTPUTS, outputs_dir, require_analysis_output  # noqa: E402


def main() -> None:
    comp_path = require_analysis_output("prerounds_experience_comparison.csv")
    comp = pd.read_csv(comp_path)

    main_res_path = ANALYSIS_OUTPUTS / "main_model_results.csv"
    labels, coefs, ps = [], [], []

    if main_res_path.exists():
        main_res = pd.read_csv(main_res_path, index_col=0)
        if "log_investor_experience_mean" in main_res.index:
            labels.append("Full-sample\ncurrent snapshot\n(07_main_model.py)")
            coefs.append(main_res.loc["log_investor_experience_mean", "coef"])
            ps.append(main_res.loc["log_investor_experience_mean", "p"])

    row_old = comp[comp["specification"] == "current_snapshot"].iloc[0]
    row_new = comp[comp["specification"] == "pre_round_only"].iloc[0]
    labels.append("Reduced-sample\ncurrent snapshot")
    coefs.append(row_old["experience_coef"])
    ps.append(row_old["experience_p"])
    labels.append("Reduced-sample\npre-round-only\nreconstruction")
    coefs.append(row_new["experience_coef"])
    ps.append(row_new["experience_p"])

    colors = ["#4C72B0" if c >= 0 else "#C44E52" for c in coefs]

    # Fix: a wider figure and a two-line wrapped title avoid the title
    # overflowing/clipping that occurred in an earlier version of this figure.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(labels))
    bars = ax.bar(x, coefs, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    for xi, (c, p) in zip(x, zip(coefs, ps)):
        star = "*" if p < 0.05 else ""
        ax.text(xi, c + (0.01 if c >= 0 else -0.03), f"{c:+.3f}{star}", ha="center",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Logistic regression coefficient\non log(mean investor experience)")
    ax.set_title(
        "Figure 4. Investor-experience coefficient:\n"
        "current-snapshot vs pre-round-only construction",
        fontsize=12,
    )
    fig.tight_layout()

    out = outputs_dir()
    fig.savefig(out / "fig4_experience_reversal.png", dpi=150)
    pd.DataFrame({"specification": labels, "coef": coefs, "p": ps}).to_csv(
        out / "fig4_experience_reversal_data.csv", index=False
    )
    print(f"Wrote {out / 'fig4_experience_reversal.png'}")
    for l, c, p in zip(labels, coefs, ps):
        print(f"  {l.replace(chr(10), ' ')}: coef={c:+.4f}  p={p:.4g}")


if __name__ == "__main__":
    main()
