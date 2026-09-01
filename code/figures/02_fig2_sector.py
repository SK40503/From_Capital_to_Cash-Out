#!/usr/bin/env python3
"""
02_fig2_sector.py
====================
Figure 2 (dissertation): exit rate by sector with 95% Wilson confidence
intervals. Uses the pooled sector_grp (categories with < 40 analytic firms
folded into 'Other', matching Appendix A / Table 5).

Input: code/analysis/outputs/model_sample_main_with_ids.csv
Output: outputs/fig2_sector_exit_rate.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import outputs_dir, require_analysis_output  # noqa: E402


def main() -> None:
    path = require_analysis_output("model_sample_main_with_ids.csv")
    df = pd.read_csv(path)

    g = df.groupby("sector_grp")["exit_any"].agg(exits="sum", n="count")
    g["rate"] = g["exits"] / g["n"]
    ci_lo, ci_hi = proportion_confint(g["exits"], g["n"], method="wilson")
    g["ci_lo"], g["ci_hi"] = ci_lo, ci_hi
    g = g.sort_values("rate")

    fig, ax = plt.subplots(figsize=(9, 8))
    y = range(len(g))
    ax.barh(y, g["rate"], color="#55A868", zorder=2)
    xerr = [g["rate"] - g["ci_lo"], g["ci_hi"] - g["rate"]]
    ax.errorbar(g["rate"], y, xerr=xerr, fmt="none", ecolor="black", capsize=4, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{s} (n={int(n)})" for s, n in zip(g.index, g["n"])], fontsize=9)
    ax.set_xlabel("Exit (Realised) rate")
    ax.set_title("Figure 2. Exit rate by sector\n(95% Wilson confidence intervals)")
    ax.axvline(df["exit_any"].mean(), color="grey", linestyle="--", linewidth=1,
               label=f"Overall rate = {df['exit_any'].mean():.1%}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = outputs_dir()
    fig.savefig(out / "fig2_sector_exit_rate.png", dpi=150)
    g.to_csv(out / "fig2_sector_exit_rate_data.csv")
    print(f"Wrote {out / 'fig2_sector_exit_rate.png'}")
    print(g)


if __name__ == "__main__":
    main()
