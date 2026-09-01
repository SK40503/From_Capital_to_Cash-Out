# From Capital to Cash-Out: data and code

Companion repository for the MSc dissertation **"From Capital to Cash-Out: Determinants of Venture and Private-Equity Exit Incidence in India, with a Descriptive Comparison of Pakistan and Bangladesh"** .

The study asks which first-round deal and investor characteristics are associated with a realised exit among 2,430 Indian venture- and private-equity-backed firms observed over 2014 to 2024, using binary logistic regression on Preqin and LSEG (Refinitiv) data. This repository holds everything a reader needs to check the numbers in the dissertation: the anonymised estimation sample, every model output the text cites, the four figures with the exact values plotted in each, and the full analysis pipeline.

## Repository map

| Folder | Contents |
|---|---|
| [`data/`](data/) | The anonymised estimation sample used to fit the primary model |
| [`results/`](results/) | The model outputs behind every estimate, table and test reported in the dissertation |
| [`figures/`](figures/) | The four dissertation figures at 400 dpi, each with a CSV of its plotted values |
| [`code/`](code/) | The analysis pipeline (`code/analysis/`) and figure scripts (`code/figures/`) |
| `requirements.txt` | Python dependencies (Python 3.9+) |

## data/

`model_sample_main.csv` is the primary estimation sample: 2,430 firms, of which 227 are coded Realised. One row per firm, containing only the modelling variables (outcome, sector group, years since first investment, pre-2000 founding flag, first-round syndicate size, foreign investor share, corporate-VC and late-stage-PE flags, and investor experience). It contains no company names or identifiers, so the primary model can be re-fitted from this file alone with any logistic-regression package.

## results/

Every figure, table and in-text estimate in the dissertation traces to one of these files. The two "main model" families are distinct on purpose: the dissertation's confirmatory specification, Equation (1), enters syndicate size categorically with first-investment-year fixed effects, while a continuous-form replication supports the sensitivity checks (the dissertation explains this in Sections 3.5 and 4.5).

| File(s) | What it holds | Where it appears |
|---|---|---|
| `main_model_categorical_yearfe_results.csv` | Equation (1), all 31 coefficients | Section 4.1, Appendix E |
| `main_model_categorical_yearfe_full_diagnostics.csv` | Log-likelihood, pseudo R-squared, cross-validated AUC, Brier score, calibration | Section 4.1, Appendix B |
| `main_model_categorical_yearfe_vif.csv`, `vif_table.csv` | Variance inflation factors | Appendix B |
| `main_model_results.csv`, `main_model_diagnostics.csv` | The continuous-form replication and its diagnostics | Sections 4.2 and 4.5 |
| `main_model_log_syndicate_yearfe_results.csv` | Log-syndicate functional-form check | Section 4.2.1 |
| `sensitivity_post2000_results.csv`, `_diag.csv` | Post-2000 founding-cohort restriction | Section 4.2.1 |
| `main_model_post2013_yearfe_results.csv`, `_diag.csv` | Post-2013 vintage restriction | Section 4.2.2 |
| `year_block_LR_test.csv`, `parsimony_drop_year_block_results.csv` | Year-block joint test and the leaner re-fit without it | Sections 3.5 and 4.2.2 |
| `sector_block_LR_test.csv`, `sector_contrasts_holm.csv` | Sector joint test and Holm-corrected contrasts | Section 4.2.3, Appendix A |
| `descriptive_sector.csv` | Realised rate by sector group | Appendix A |
| `h1_*.csv` | Alternative investor-experience measures (log max, mean without syndicate, top quartile) | Section 4.2.4 |
| `liquidity_dealtype_results.csv`, `_diag.csv`, `liquidity_dealtype_syndicate_LR.csv` | Equation (2): the independent deal-type liquidity outcome, 332 events | Section 4.2.5 |
| `sensitivity_confirmedexit_results.csv`, `_diag.csv` | Equation (3): the stricter Realised-and-liquidity outcome, 61 events | Section 4.2.5 |
| `outcome_crosstab.csv` | The classification disagreement between the two outcome definitions | Section 4.2.5 |
| `prerounds_experience_comparison.csv` | Current-snapshot vs pre-round investor experience | Section 4.2.4, Figure 4 |
| `descriptive_continuous.csv` | Descriptive statistics for the continuous variables | Table 3 |
| `confirmedexit_categorical_yearfe_results.csv`, `_diag.csv` | Categorical-syndicate variant of the confirmed-liquidity model | Section 4.2.5 |
| `lseg_investment_rounds_by_country.csv` | LSEG investment-round counts for India, Pakistan and Bangladesh | Section 4.4 |

## figures/

The four figures exactly as published, with the plotted values beside each so every figure can be checked without running anything: `fig1_syndicate_exit_rate` (exit rate by first-round syndicate size, Wilson intervals), `fig2_sector_exit_rate` (exit rate by sector), `fig3_syndicate_sensitivity` (the syndicate gradient across specifications and outcome definitions), `fig4_experience_reversal` (current-snapshot vs pre-round experience).

## code/

`code/analysis/` contains the numbered pipeline (`01` ingestion through `14` the independent outcome and multiplicity correction), plus `run_all.py`. Each script's docstring states what it builds and how closely it reproduces the historical outputs. `code/figures/` contains one script per figure and `00_build_all_figures.py`.

**What runs from this repository alone.** The raw Preqin and LSEG exports are licensed commercial products and are not redistributed (see below), so the ingestion and matching scripts (01 to 06) cannot run here. What a reader can do without them:

- **Re-fit the primary model** from `data/model_sample_main.csv` directly; it is the exact estimation sample.
- **Regenerate Figures 3 and 4**, which read only saved model outputs: copy the contents of `results/` into `code/analysis/outputs/` and run the two figure scripts.
- **Audit any reported number** against the corresponding file in `results/`.

Anyone holding licensed copies of the four raw exports can run the full pipeline end to end:

```bash
pip install -r requirements.txt
export DISSERTATION_RAW_DIR=/path/to/raw/exports
cd code/analysis
python3 run_all.py
python3 14_independent_liquidity_and_multiplicity.py
cd ../figures
python3 00_build_all_figures.py
```

## What is deliberately not here

Preqin and LSEG (Refinitiv) data are licensed commercial products, so this repository excludes the raw deal-level exports and every derived file that carries company names or identifiers (including the ID-bearing variant of the estimation sample that Figures 1 and 2 are drawn from; their plotted values are in `figures/` instead). The anonymised sample and the aggregated model outputs are the my own analytical results and contain no vendor records.

## Provenance

Every number in the dissertation traces to a file in `results/`, written by the scripts in `code/analysis/`, which read the licensed vendor exports. Nothing was transcribed by hand. Three sample sizes recur in the text and are all correct: 2,430 firms with 227 events (primary estimation sample), 2,303 firms with 179 events (pre-round experience replication) and 332 firms (the independent liquidity outcome).
