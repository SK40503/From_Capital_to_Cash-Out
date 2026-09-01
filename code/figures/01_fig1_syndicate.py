#!/usr/bin/env python3
"""
01_fig1_syndicate.py
=======================
Figure 1 (dissertation): distribution of Realised vs Active status by
first-round syndicate-size category (1, 2, 3, 4+), with Wilson score
confidence intervals for the exit rate in each category.

Uses statsmodels.stats.proportion.proportion_confint (Wilson method) as
specified in the task brief.

Input: code/analysis/outputs/model_sample_main_with_ids.csv
       (run code/analysis/run_all.py first)
Output: outputs/fig1_syndicate_exit_rate.png
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


def bucket(n: float) -> str:
    if n >= 4:
        return "4+"
    return str(int(n))


def main() -> None:
    path = require_analysis_output("model_sample_main_with_ids.csv")
    df = pd.read_csv(path)
    df["syndicate_cat"] = df["first_round_syndicate_size"].apply(bucket)

    order = ["1", "2", "3", "4+"]
    g = df.groupby("syndicate_cat")["exit_any"].agg(exits="sum", n="count").reindex(order)
    g["rate"] = g["exits"] / g["n"]
    ci_lo, ci_hi = proportion_confint(g["exits"], g["n"], method="wilson")
    g["ci_lo"], g["ci_hi"] = ci_lo, ci_hi

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(order))
    ax.bar(x, g["rate"], color="#4C72B0", width=0.6, zorder=2)
    yerr = [g["rate"] - g["ci_lo"], g["ci_hi"] - g["rate"]]
    ax.errorbar(x, g["rate"], yerr=yerr, fmt="none", ecolor="black", capsize=5, zorder=3)
    for xi, (n, exits) in enumerate(zip(g["n"], g["exits"])):
        ax.text(xi, g["rate"].iloc[xi] + (yerr[1].iloc[xi] if hasattr(yerr[1], "iloc") else yerr[1][xi]) + 0.01,
                 f"{int(exits)}/{int(n)}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(order)
    ax.set_xlabel("First-round syndicate size")
    ax.set_ylabel("Realised (exit) rate")
    ax.set_title("Figure 1. Realised status by first-round syndicate size\n"
                  "(95% Wilson confidence intervals)")
    ax.set_ylim(0, max(g["ci_hi"]) * 1.25)
    fig.tight_layout()

    out = outputs_dir()
    fig.savefig(out / "fig1_syndicate_exit_rate.png", dpi=150)
    g.to_csv(out / "fig1_syndicate_exit_rate_data.csv")
    print(f"Wrote {out / 'fig1_syndicate_exit_rate.png'}")
    print(g)


if __name__ == "__main__":
    main()
