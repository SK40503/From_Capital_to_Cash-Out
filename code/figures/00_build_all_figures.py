"""
00_build_all_figures.py — canonical figure build for the submission-ready dissertation.

Regenerates Figures 1-4 from code/analysis/outputs/ at print quality:
  * Arial 9-10pt at final printed width (5.7 in, fits the 5.79 in text block)
  * Okabe-Ito colour-blind-safe palette; every figure also readable in greyscale
    (categories are distinguished by position/shading, never by hue alone)
  * 400 dpi, tight bounding box, no clipped or overlapping text
  * every panel annotated with its own n, so each figure is self-contained

Run after code/analysis/run_all.py and 14_independent_liquidity_and_multiplicity.py.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
AN = HERE.parent / "analysis" / "outputs"
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 400,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
BLUE, ORANGE, GREY, DARK = "#0072B2", "#E69F00", "#999999", "#333333"
W = 5.7

EX = {"IPO", "Sale to Management", "Unspecified Exit"}
PAT = ["Trade Sale", "Secondary Buyout"]
def is_liq(dt):
    if pd.isna(dt): return False
    return dt in EX or any(p in dt for p in PAT)

def load():
    df = pd.read_csv(AN / "model_sample_main_with_ids.csv", parse_dates=["preqin_earliest_deal_date"])
    df["syndicate_cat"] = pd.Categorical(
        df.first_round_syndicate_size.apply(lambda n: "4+" if n >= 4 else str(int(n))),
        categories=["1", "2", "3", "4+"], ordered=True)
    df["first_investment_year"] = df.preqin_earliest_deal_date.dt.year.clip(2015, 2024)
    df["liquidity_dealtype"] = df.preqin_most_recent_deal_type.apply(is_liq).astype(int)
    return df

def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return lo, hi

# ---------------------------------------------------------------- Figure 1
def fig1(df):
    g = df.groupby("syndicate_cat", observed=True).exit_any.agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    g["lo"], g["hi"] = zip(*[wilson(k, n) for k, n in zip(g["sum"], g["count"])])
    fig, ax = plt.subplots(figsize=(W, 2.9))
    x = np.arange(len(g))
    ax.bar(x, g.rate * 100, width=0.6, color=BLUE, edgecolor=DARK, linewidth=0.6, zorder=2)
    ax.errorbar(x, g.rate * 100, yerr=[(g.rate - g.lo) * 100, (g.hi - g.rate) * 100],
                fmt="none", ecolor=DARK, elinewidth=1.0, capsize=3.5, zorder=3)
    tb = ax.get_xaxis_transform()          # x in data coords, y in axes fraction
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(i, r.hi * 100 + 0.85, f"{r.rate*100:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=4)
        ax.text(i, -0.135, f"n = {int(r['count']):,}", ha="center", va="top",
                fontsize=8, color=DARK, transform=tb, clip_on=False)
        ax.text(i, -0.235, f"{int(r['sum'])} Realised", ha="center", va="top",
                fontsize=8, color=DARK, transform=tb, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(g.index.astype(str))
    ax.set_xlabel("First-round syndicate size (number of distinct investors)", labelpad=40)
    ax.set_ylabel("Firms coded Realised (%)")
    ax.set_ylim(0, max(g.hi) * 100 + 3.2); ax.set_xlim(-0.6, len(g) - 0.4)
    ax.yaxis.grid(True, linewidth=0.5, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig1_syndicate_exit_rate.png"); plt.close(fig)
    g.to_csv(OUT / "fig1_syndicate_exit_rate_data.csv")
    print(f"  Fig 1: {len(g)} categories, N={int(g['count'].sum())}, events={int(g['sum'].sum())}")

# ---------------------------------------------------------------- Figure 2
def fig2(df):
    g = df.groupby("sector_grp").exit_any.agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    g["lo"], g["hi"] = zip(*[wilson(k, n) for k, n in zip(g["sum"], g["count"])])
    g = g.sort_values("rate")
    base = df.exit_any.mean()
    fig, ax = plt.subplots(figsize=(W, 4.5))
    y = np.arange(len(g))
    ax.axvline(base * 100, color=ORANGE, linestyle="--", linewidth=1.2, zorder=1)
    ax.errorbar(g.rate * 100, y, xerr=[(g.rate - g.lo) * 100, (g.hi - g.rate) * 100],
                fmt="o", color=BLUE, ecolor=DARK, elinewidth=1.0, capsize=2.8,
                markersize=4.5, markeredgecolor=DARK, markeredgewidth=0.5, zorder=3)
    labels = [f"{s}  (n = {int(g.loc[s,'count'])})" for s in g.index]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Firms coded Realised (%), with 95% Wilson confidence intervals")
    ax.set_xlim(0, max(g.hi) * 100 + 2); ax.set_ylim(-0.7, len(g) - 0.3)
    # place the baseline label in clear space at the top, never over a marker
    ax.annotate(f"Sample rate {base*100:.1f}%", xy=(base * 100, len(g) - 0.45),
                xytext=(base * 100 + 1.6, len(g) - 0.45), fontsize=8.5, color=ORANGE,
                fontweight="bold", va="center", ha="left")
    ax.xaxis.grid(True, linewidth=0.5, color="#DDDDDD", zorder=0); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig2_sector_exit_rate.png"); plt.close(fig)
    g.to_csv(OUT / "fig2_sector_exit_rate_data.csv")
    print(f"  Fig 2: {len(g)} sectors, N={int(g['count'].sum())}, events={int(g['sum'].sum())}")

# ---------------------------------------------------------------- Figure 3
def fig3(df):
    BASE = ("C(syndicate_cat, Treatment(reference='1')) + C(sector_grp, Treatment(reference='Software')) "
            "+ flag_founding_pre2000 + first_round_foreign_investor_share + log_investor_experience_mean "
            "+ C(first_investment_year, Treatment(reference=2015))")
    NOYR = BASE.replace(" + C(first_investment_year, Treatment(reference=2015))", "")
    key = "C(syndicate_cat, Treatment(reference='1'))[T.2]"
    def est(formula, data, k=key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.logit(formula, data=data).fit(disp=0, maxiter=200)
        c, se = f.params[k], f.bse[k]
        return np.exp(c), np.exp(c - 1.96 * se), np.exp(c + 1.96 * se), f.pvalues[k]
    # Rows 3 and 4 are read from the saved outputs of analysis script 13 so that the
    # figure and the Results text are driven by exactly one source per specification.
    def from_csv(name, ev):
        t = pd.read_csv(AN / name, index_col=0)
        r = [i for i in t.index if "syndicate_cat" in i and "[T.2]" in i][0]
        return t.loc[r, "OR"], t.loc[r, "OR_lo"], t.loc[r, "OR_hi"], t.loc[r, "p"], ev
    res = []
    for lab, f, ev in [("Primary model\nRealised status", f"exit_any ~ {BASE}", 227),
                       ("Year effects dropped\n(simplified)", f"exit_any ~ {NOYR}", 227)]:
        o, lo, hi, p = est(f, df); res.append((lab, o, lo, hi, p, ev))
    res.append(("Founded 2013 or later",) + from_csv("main_model_post2013_yearfe_results.csv", 149))
    res.append(("Realised AND liquidity\ndeal type (nested)",) + from_csv("confirmedexit_categorical_yearfe_results.csv", 61))
    res.append(("Liquidity deal type only\n(independent of status)",) + from_csv("liquidity_dealtype_results.csv", 332))
    fig, ax = plt.subplots(figsize=(W, 3.0))
    fig.subplots_adjust(left=0.30, right=0.62)          # reserve a clear text column
    y = np.arange(len(res))[::-1]
    ax.axvline(1.0, color=GREY, linestyle="-", linewidth=1.0, zorder=1)
    # annotation columns live OUTSIDE the axes, in axes-fraction x / data y
    tc = ax.get_yaxis_transform()
    for i, (lab, o, lo, hi, p, ev) in enumerate(res):
        yy = y[i]
        sig = hi < 1.0 or lo > 1.0
        col, mk = (BLUE, "o") if sig else (ORANGE, "s")
        ax.plot([lo, hi], [yy, yy], color=col, linewidth=1.6, zorder=2, solid_capstyle="round")
        ax.plot([o], [yy], mk, color=col, markersize=6, markeredgecolor=DARK,
                markeredgewidth=0.6, zorder=3)
        ax.text(1.06, yy, f"{o:.2f}", fontsize=8.4, va="center", ha="left",
                transform=tc, clip_on=False)
        ax.text(1.24, yy, f"[{lo:.2f}, {hi:.2f}]", fontsize=8.4, va="center", ha="left",
                transform=tc, clip_on=False)
        ax.text(1.72, yy, f"{ev:,}", fontsize=8.4, va="center", ha="right",
                transform=tc, clip_on=False)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in res], fontsize=8.6)
    ax.set_xscale("log"); ax.set_xlim(0.12, 2.6)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.125", "0.25", "0.50", "1.00", "2.00"])
    ax.set_xlabel("Odds ratio, two investors versus one (log scale), with 95% CI")
    ax.set_ylim(-0.7, len(res) - 0.15)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    hy = len(res) - 0.42
    ax.text(1.06, hy, "OR", fontsize=8.4, fontweight="bold", ha="left", transform=tc, clip_on=False)
    ax.text(1.24, hy, "95% CI", fontsize=8.4, fontweight="bold", ha="left", transform=tc, clip_on=False)
    ax.text(1.72, hy, "Events", fontsize=8.4, fontweight="bold", ha="right", transform=tc, clip_on=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", markersize=5.5, lw=1.6,
                              markeredgecolor=DARK, label="95% CI excludes 1"),
                       Line2D([], [], color=ORANGE, marker="s", markersize=5.5, lw=1.6,
                              markeredgecolor=DARK, label="95% CI includes 1")],
              loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2, frameon=False,
              handletextpad=0.5, columnspacing=1.6)
    fig.savefig(OUT / "fig3_syndicate_sensitivity.png"); plt.close(fig)
    pd.DataFrame(res, columns=["specification", "OR", "CI_lo", "CI_hi", "p", "events"]).to_csv(
        OUT / "fig3_syndicate_sensitivity_data.csv", index=False)
    for r in res: print(f"  Fig 3: {r[0][:34]:36} OR={r[1]:.3f} [{r[2]:.3f},{r[3]:.3f}] ev={r[5]}")

# ---------------------------------------------------------------- Figure 4
def fig4():
    pr = pd.read_csv(AN / "model_sample_with_prerounds_exp.csv")
    # canonical specification: identical to code/analysis/10_prerounds_model.py
    # (spec A continuous form), so the figure matches the odds ratios reported in
    # Section 4.5 exactly. Only the experience term differs between the two fits.
    B = ("C(sector_grp, Treatment(reference='Software')) + years_since_first_investment "
         "+ flag_founding_pre2000 + first_round_syndicate_size "
         "+ first_round_foreign_investor_share")
    res = []
    for lab, v in [("Current snapshot\n(count at extraction)", "log_old_experience"),
                   ("Pre-round only\n(deals before first round)", "log_new_experience")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.logit(f"exit_any ~ {B} + {v}", data=pr).fit(disp=0, maxiter=200)
        c, se = f.params[v], f.bse[v]
        res.append((lab, np.exp(c), np.exp(c - 1.96 * se), np.exp(c + 1.96 * se), f.pvalues[v]))
    fig, ax = plt.subplots(figsize=(W, 2.1))
    fig.subplots_adjust(left=0.33, right=0.55)
    tc = ax.get_yaxis_transform()
    y = [1, 0]
    ax.axvline(1.0, color=GREY, linewidth=1.0, zorder=1)
    for i, (lab, o, lo, hi, p) in enumerate(res):
        col, mk = (ORANGE, "s") if hi > 1.0 else (BLUE, "o")
        ax.plot([lo, hi], [y[i], y[i]], color=col, linewidth=1.8, zorder=2, solid_capstyle="round")
        ax.plot([o], [y[i]], mk, color=col, markersize=6.5, markeredgecolor=DARK,
                markeredgewidth=0.6, zorder=3)
        ax.text(1.07, y[i], f"OR {o:.2f} [{lo:.2f}, {hi:.2f}]", fontsize=8.6,
                va="center", ha="left", transform=tc, clip_on=False)
        ax.text(1.07, y[i] - 0.17, f"p = {p:.3f}", fontsize=8.6, color=DARK,
                va="center", ha="left", transform=tc, clip_on=False)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in res], fontsize=8.8)
    ax.set_xlim(0.66, 1.14); ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1])
    ax.set_xlabel("Odds ratio per unit of log mean first-round\ninvestor experience, with 95% CI")
    ax.set_ylim(-0.55, 1.55)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "fig4_experience_reversal.png"); plt.close(fig)
    pd.DataFrame(res, columns=["specification", "OR", "CI_lo", "CI_hi", "p"]).to_csv(
        OUT / "fig4_experience_reversal_data.csv", index=False)
    for r in res: print(f"  Fig 4: {r[0][:30]:32} OR={r[1]:.3f} [{r[2]:.3f},{r[3]:.3f}] p={r[4]:.4f}  (N={len(pr)})")

if __name__ == "__main__":
    df = load()
    print(f"Base: N={len(df)}, Realised={int(df.exit_any.sum())}, liquidity_dealtype={int(df.liquidity_dealtype.sum())}")
    fig1(df); fig2(df); fig3(df); fig4()
    print(f"\nFigures written to {OUT}")
