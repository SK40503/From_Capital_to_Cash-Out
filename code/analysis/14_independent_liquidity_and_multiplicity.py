"""
14_independent_liquidity_and_multiplicity.py
============================================
Two additions required for the final dissertation, both flagged during the
submission-readiness review:

(1) AN OUTCOME THAT DOES NOT DEPEND ON PREQIN'S COMPANY-STATUS FIELD.

    The sensitivity outcome used previously was defined as
        exit_any == 1  AND  qualifying most-recent deal type
    i.e. it was NESTED inside the disputed Preqin `Realised` classification
    and therefore could not test whether the syndicate-size association is an
    artefact of that classification.

    This script builds `liquidity_dealtype`, which uses ONLY the deal-type
    field and never references company status:

        liquidity_dealtype = 1 if preqin_most_recent_deal_type is
            'IPO', 'Sale to Management' or 'Unspecified Exit' (exact), or
            contains 'Trade Sale' or 'Secondary Buyout' (compound labels such
            as 'Add-on, Trade Sale' and 'PIPE, Secondary Buyout'); else 0.

    Exact matching for IPO deliberately excludes 'Pre-IPO', which is a
    financing round, not a liquidity event. The rule is otherwise identical to
    the one used for the nested outcome, so the two differ ONLY in whether
    company status is required.

    Because this outcome is independent of the aggregation rule, Equation (1)
    can be fitted in full (sector and year blocks retained).

(2) A FORMAL MULTIPLICITY CORRECTION.

    Earlier text referred to "multiplicity control" without any correction
    having been computed. This script applies Holm-Bonferroni to the 15
    sector contrasts of Equation (1) and writes the adjusted p-values.

Inputs : outputs/model_sample_main_with_ids.csv  (N = 2,430; 227 Realised)
Outputs: liquidity_dealtype_*.csv, nested_realised_liquidity_*.csv,
         sector_contrasts_holm.csv, outcome_crosstab.csv
"""
from __future__ import annotations
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from _common import outputs_dir

LIQUIDITY_EXACT = {"IPO", "Sale to Management", "Unspecified Exit"}
LIQUIDITY_PATTERNS = ["Trade Sale", "Secondary Buyout"]

def is_liquidity_dealtype(dt) -> bool:
    if pd.isna(dt): return False
    if dt in LIQUIDITY_EXACT: return True
    return any(p in dt for p in LIQUIDITY_PATTERNS)

def or_table(fit) -> pd.DataFrame:
    p = fit.params; se = fit.bse
    return pd.DataFrame({"coef": p, "se": se, "OR": np.exp(p),
                         "OR_lo": np.exp(p - 1.96*se), "OR_hi": np.exp(p + 1.96*se),
                         "p": fit.pvalues})

def main() -> None:
    out = outputs_dir()
    df = pd.read_csv(out / "model_sample_main_with_ids.csv", parse_dates=["preqin_earliest_deal_date"])
    df["syndicate_cat"] = df.first_round_syndicate_size.apply(lambda n: "4+" if n >= 4 else str(int(n)))
    df["first_investment_year"] = df.preqin_earliest_deal_date.dt.year.clip(2015, 2024)
    df["liquidity_dealtype"] = df.preqin_most_recent_deal_type.apply(is_liquidity_dealtype).astype(int)
    df["realised_and_liquidity"] = ((df.exit_any == 1) & (df.liquidity_dealtype == 1)).astype(int)

    print(f"Base sample: N={len(df)}, Realised (exit_any)={int(df.exit_any.sum())}")
    print("\n=== INDEPENDENCE CHECK: company status x liquidity-type most-recent deal ===")
    ct = pd.crosstab(df.exit_any, df.liquidity_dealtype, margins=True)
    ct.to_csv(out / "outcome_crosstab.csv"); print(ct)
    n_liq = int(df.liquidity_dealtype.sum()); n_act = int(((df.exit_any==0)&(df.liquidity_dealtype==1)).sum())
    print(f"\nliquidity_dealtype events: {n_liq} ({n_liq/len(df)*100:.1f}% of {len(df)})")
    print(f"  of which coded Active by Preqin : {n_act} ({n_act/n_liq*100:.1f}%)")
    print(f"  of which coded Realised         : {n_liq-n_act} ({(n_liq-n_act)/n_liq*100:.1f}%)")
    print(f"nested realised_and_liquidity events: {int(df.realised_and_liquidity.sum())}")

    BASE = ("C(syndicate_cat, Treatment(reference='1')) "
            "+ C(sector_grp, Treatment(reference='Software')) + flag_founding_pre2000 "
            "+ first_round_foreign_investor_share + log_investor_experience_mean "
            "+ C(first_investment_year, Treatment(reference=2015))")

    # --- (1a) independent outcome, full Equation (1) form ---
    print("\n=== MODEL A: liquidity_dealtype (independent of company status), Equation (1) form ===")
    fa = smf.logit(f"liquidity_dealtype ~ {BASE}", data=df).fit(disp=0, maxiter=200)
    ta = or_table(fa); ta.to_csv(out / "liquidity_dealtype_results.csv")
    k = len(fa.params)
    for lab in ["[T.2]", "[T.3]", "[T.4+]"]:
        r = [i for i in ta.index if "syndicate_cat" in i and lab in i][0]
        print(f"  syndicate {lab:6} OR={ta.loc[r,'OR']:.3f} [{ta.loc[r,'OR_lo']:.3f}, {ta.loc[r,'OR_hi']:.3f}] p={ta.loc[r,'p']:.4g}")
    r = "log_investor_experience_mean"
    print(f"  log mean experience  OR={ta.loc[r,'OR']:.3f} [{ta.loc[r,'OR_lo']:.3f}, {ta.loc[r,'OR_hi']:.3f}] p={ta.loc[r,'p']:.4g}")
    r = "first_round_foreign_investor_share"
    print(f"  foreign share        OR={ta.loc[r,'OR']:.3f} [{ta.loc[r,'OR_lo']:.3f}, {ta.loc[r,'OR_hi']:.3f}] p={ta.loc[r,'p']:.4g}")
    ev = int(df.liquidity_dealtype.sum())
    print(f"  N={int(fa.nobs)} events={ev} params={k} EPP={ev/k:.2f} pseudoR2={1-fa.llf/fa.llnull:.4f}")
    pd.DataFrame({"metric": ["N","events","n_parameters","events_per_parameter","pseudo_R2_McFadden"],
                  "value": [int(fa.nobs), ev, k, round(ev/k,2), round(1-fa.llf/fa.llnull,4)]}
                 ).to_csv(out / "liquidity_dealtype_diag.csv", index=False)

    # syndicate block joint LR test on the independent outcome
    fa_nos = smf.logit(f"liquidity_dealtype ~ {BASE.replace(chr(34),chr(34))}".replace(
        "C(syndicate_cat, Treatment(reference='1')) + ", ""), data=df).fit(disp=0, maxiter=200)
    lr = 2*(fa.llf - fa_nos.llf); dfree = len(fa.params)-len(fa_nos.params)
    from scipy import stats
    p_lr = stats.chi2.sf(lr, dfree)
    print(f"  syndicate block joint LR: chi2({dfree})={lr:.2f}, p={p_lr:.4g}")
    pd.DataFrame({"test":["syndicate_block_liquidity_dealtype"],"LR_chi2":[lr],"df":[dfree],"p":[p_lr]}
                 ).to_csv(out / "liquidity_dealtype_syndicate_LR.csv", index=False)

    # --- (1b) nested outcome, same form: REPORTED ONLY IF IT CONVERGES ---
    print("\n=== MODEL B: realised_and_liquidity (NESTED inside Realised), same form ===")
    fb = smf.logit(f"realised_and_liquidity ~ {BASE}", data=df).fit(disp=0, maxiter=200)
    nb = int(df.realised_and_liquidity.sum())
    if fb.mle_retvals.get("converged") and np.abs(fb.params).max() < 10:
        or_table(fb).to_csv(out / "nested_realised_liquidity_results.csv")
        print(f"  converged; N={int(fb.nobs)} events={nb}")
    else:
        print(f"  NOT REPORTED: {nb} events on {len(fb.params)} parameters "
              f"(EPP={nb/len(fb.params):.2f}) does not converge (max|coef|="
              f"{np.abs(fb.params).max():.1f}); this is quasi-separation, not an estimate.")
        print("  The nested outcome is reported only in the reduced Equation (2) form (script 13).")

    # --- (2) Holm-Bonferroni on the 15 sector contrasts of the PRIMARY model ---
    print("\n=== MULTIPLICITY: Holm-Bonferroni over the 15 sector contrasts, primary model ===")
    fp = smf.logit(f"exit_any ~ {BASE}", data=df).fit(disp=0, maxiter=200)
    tp = or_table(fp)
    sec = tp[[("sector_grp" in i) for i in tp.index]].copy()
    rej, padj, _, _ = multipletests(sec["p"].values, alpha=0.05, method="holm")
    sec["p_holm"] = padj; sec["significant_after_holm"] = rej
    sec.index = [i.split("[T.")[1].rstrip("]") for i in sec.index]
    sec[["OR","OR_lo","OR_hi","p","p_holm","significant_after_holm"]].to_csv(out / "sector_contrasts_holm.csv")
    print(sec[["OR","p","p_holm","significant_after_holm"]].sort_values("p").to_string(
        float_format=lambda v: f"{v:.4f}"))
    print(f"\n  contrasts nominally p<0.05 : {int((sec['p']<0.05).sum())} of {len(sec)}")
    print(f"  surviving Holm correction  : {int(sec['significant_after_holm'].sum())} of {len(sec)}")
    print(f"\nAll outputs written to {out}")

if __name__ == "__main__":
    main()
