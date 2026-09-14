# 5′UTR burden analysis in the UK Biobank — downstream analysis code

Analysis code accompanying:

> **Mechanistic 5′UTR Variant Scoring Expands Rare Variant Discovery in the UK Biobank**
> Matthieu Chaldebas, Khoren Ponsin, Haralambos A. Mourelatos, Yoann Seeleuthner,
> Clément Conil, Jonathan Bohlen, Jean-Laurent Casanova, Peng Zhang, Aurélie Cobat
> *medRxiv* (2026). doi: [10.64898/2026.09.09.26362607](https://doi.org/10.64898/2026.09.09.26362607)

The 5ULTRA classifier itself is a separate repository:
https://github.com/casanova-lab/5ULTRA

This repository contains the **downstream analysis** — every step **after** the
REGENIE burden association testing (Step 2): results aggregation, comparison
against the UK Biobank WGS Consortium (Nature 2025) catalog, supplementary-table
generation, genomic-inflation (λGC) calibration, threshold and carrier-floor
sensitivity analyses, the translational direction-of-effect (rheostat) analyses,
the exome-PTV and cis-pQTL validations, and all main and supplementary figures.

---

## UK Biobank data governance

**No individual-level UK Biobank data are distributed in this repository.**

All analyses were performed under UK Biobank application **98772**. The upstream
pipeline — variant extraction, 5ULTRA/CADD annotation, phenotype curation, LD
pruning, and REGENIE Step 1 / Step 2 — was performed within the approved UK
Biobank Research Analysis Platform and is **not** included here.

This repository ships **code only**: no genotypes, no phenotypes, no participant
identifiers. It operates on the gene-level summary statistics produced by REGENIE
Step 2 and on publicly available external reference data (the Nature 2025
Consortium results and Hawkes *et al.* cis-pQTL summary data).

The UK Biobank Material Transfer Agreement forbids redistribution of
individual-level data or participant identifiers. `.gitignore` blocks the
relevant file classes at the repository level; please keep it in force when
contributing. Access to UK Biobank data is by application
(https://www.ukbiobank.ac.uk/), and individual-level inputs must be regenerated
by approved researchers inside their own UK Biobank environment.

Generated outputs — figures, supplementary tables — are deliberately **not**
tracked. Regenerate them with the scripts below, or use the published
supplementary material.

---

## Repository layout

```
analysis_config.py   Single source of truth for significance thresholds and model sets
05_results/          Aggregate REGENIE Step-2 output into a single results table
06_analysis/         Comparison, supplementary tables, sensitivity analyses, rheostat, pQTL
07_figures/          Main Figures 1–5, Supplementary Figure 1, shared label/classification modules
environment.yml      Conda environment (Python dependencies only)
```

### `analysis_config.py`

Defines `DATA_DIR`, the master results path, the carrier floor (`TRUE_K_MIN = 5`)
and the genome-wide significance thresholds.

Thresholds are Bonferroni corrections for the **exact number of association tests
performed in each suite**, counted after the `k ≥ 5` filter:

| Suite | Tests | α | Threshold |
|-------|------:|---|-----------|
| 5ULTRA | 518,380 | 9.645 × 10⁻⁸ | −log₁₀P > 7.016 |
| CADD | 107,033 | 4.671 × 10⁻⁷ | −log₁₀P > 6.331 |
| Ablation | 68,510 | 7.298 × 10⁻⁷ | −log₁₀P > 6.137 |

`verify_test_counts(df)` asserts that these counts still hold after the master
table is loaded; if the pipeline is rerun and the counts move, the thresholds
must move with them. Set `THRESH_SCHEME=current` to reproduce the pre-revision
thresholds, or `THRESH_SCHEME=pooled` for the one-bar-for-both-suites sensitivity
analysis.

### `05_results/`

| Script | Purpose |
|--------|---------|
| `01_aggregate_results.py` | Parse all `.regenie.gz` files into `Master_Results_Clean.csv.gz`, recording the burden magnitude `k` and deriving `true_k`, the carrier floor used throughout |

> **On `k` and `true_k`.** REGENIE reports `k = A1FREQ × 2N` for every mask. For
> binary-weight masks this equals a physical carrier count; for weighted masks
> (`Weighted`, `Mirror`, `Flux`) it is a weighted burden sum. `01_aggregate_results.py`
> propagates the binary-mask carrier count to `true_k` where one exists, so that
> the `true_k ≥ 5` filter means the same thing across models. All analyses and
> reported tables use this single `k` definition.

### `06_analysis/`

| Script | Purpose |
|--------|---------|
| `01_nature2025_overlap.py` | Compare against the Nature 2025 Consortium 5′UTR catalog; classify replicated, suggestive, missed, novel and high-gain hits |
| `02_generate_supp_tables_nature2025.py` | **Tables S6 and S7** — Consortium replication and novel/high-gain genes |
| `03_generate_supp_tables_S3_S4.py` | **Tables S3 and S4** — genome-wide significant 5ULTRA and CADD associations |
| `04_build_pqtl_conditioning.py` | Build the cis-pQTL conditioning set from Hawkes *et al.* summary data |
| `04_generate_supp_table_lambdaGC.py` | **Table S5** — genomic inflation (λGC) per model × spectrum |
| `05_extract_rheostat_burden.py` | Per-individual UP/DN burden for the highlighted pairs — **runs on the UKB platform only** |
| `06_rheostat_beta_equality.py` | Portable Wald test of β_UP − β_DN = 0 (`p_equality`) and β_UP + β_DN = 0 (`p_symmetry`) |
| `07_cross_ancestry_replication.py` | Replication and generalization across AFR / ASJ / EAS / SAS / OTH cohorts |
| `08_combined_model_comparison.py` | Combined 5ULTRA + CADD mask comparison |
| `09_threshold_sensitivity.py` | Discovery counts under exact Bonferroni vs. the earlier 1e-6 / N_models construction; produces the test counts used in `analysis_config.py` |
| `10_check_k_filter_symmetry.py` | Diagnostic: is the k ≥ 5 filter applied to the same quantity in both suites? |
| `11_exclusive_hits_coverage.py` | Is the 5ULTRA-vs-CADD gap a detection difference or a coverage difference? |
| `12_directionality_census.py` | Significance-based directionality census over all informative UP/DN pairs |
| `13_ptv_k_floor_sensitivity.py` | Dependence of the 5′UTR-DN vs. exome-PTV correlation on the carrier floor |
| `14_generate_supp_table_S2.py` | **Table S2** — carrier-floor sensitivity of the PTV correlation |
| `15_generate_supp_table_S1.py` | **Table S1** — the 59 traits, UK Biobank field IDs, abbreviations and full names |
| `16_ablated_pairs_in_5ultra.py` | Are the CADD associations lost to ablation detected by 5ULTRA? |

`rheostat_pairs_nfe.txt` lists the gene / UK Biobank field-ID pairs used by the
rheostat β-equality analysis. It contains no participant-level information.

### `07_figures/`

| Script | Purpose |
|--------|---------|
| `figure1.py` | 5ULTRA score × allele frequency; mechanism abundance; ancestry ridgelines |
| `figure2.py` | Discovery landscape: 5ULTRA vs. CADD butterfly chart + ablation cascade |
| `figure3.py` | Translational rheostat: DN vs. UP Z-scores, bidirectional examples (+ β-equality overlay if `RHEO_EQ_FILE` is set) |
| `figure4.py` | 5′UTR-DN vs. exome-PTV concordance |
| `figure5.py` | cis-pQTL validation |
| `supp_figure1.py` | Supplementary Figure 1 — Miami and QQ plots |
| `pheno_labels.py` | Canonical phenotype abbreviations shared across scripts |
| `rheostat_classify.py` | Significance-based UP/DN pair classification shared by Figure 3 and the census |

---

## Required inputs (not included — produced upstream on the approved platform)

Scripts read from the directory given by the `DATA_DIR` environment variable
(default: `data/` relative to the repository root).

| File | Produced by | Level |
|------|-------------|-------|
| `Master_Results_Clean.csv.gz` | REGENIE Step 2 → `05_results/01_aggregate_results.py` | gene × phenotype × model summary statistics |
| `Final_Triad_Comparison.csv.gz` | REGENIE Step 2 (5ULTRA / CADD / ablation suites) | summary statistics |
| `Nature_2025_Reproduction_Data.csv` | UK Biobank WGS Consortium (Nature 2025) | external summary data |
| `EUR.5UTRs.MANE.GENE.cis.tsv` | Hawkes *et al.* cis-pQTL | external summary data |
| `pheno_covar_final_clean*.txt`, `final_complete_set/*_5utr_complete` (PLINK2 pgen) | UK Biobank | **individual-level — run only within the approved environment** |

Most scripts run entirely on the gene-level summary statistics. Only the rheostat
β-equality reconstruction (`06_analysis/05_extract_rheostat_burden.py`, then
`06_analysis/06_rheostat_beta_equality.py`) additionally requires individual-level
genotype/phenotype data and must therefore be run inside the approved UK Biobank
environment.

---

## Installation

```bash
conda env create -f environment.yml
conda activate 5utr_burden
```

Python 3.10 with numpy, pandas, scipy, matplotlib, seaborn, openpyxl and
adjustText. No compilation step.

### External tools

The upstream pipeline (not in this repository) additionally requires the
following binaries on `PATH`. They are not conda-installable as Python packages
and must be installed separately:

| Tool | Used for |
|------|----------|
| REGENIE (≥ 3.2) | Step 1 whole-genome regression and Step 2 burden testing |
| PLINK2 | Genotype extraction, LD pruning, pgen handling |
| bcftools | VCF filtering and normalization |
| bedtools | 5′UTR interval operations |
| tabix / bgzip (HTSlib) | Indexing and compression |
| 5ULTRA CLI | 5′UTR variant scoring — https://github.com/casanova-lab/5ULTRA |

None of these are needed to run the downstream scripts in this repository, which
operate on the aggregated summary statistics.

---

## Usage

```bash
export DATA_DIR=/path/to/your/summary-statistics

python 05_results/01_aggregate_results.py          # build Master_Results_Clean.csv.gz

python 06_analysis/09_threshold_sensitivity.py     # exact-Bonferroni test counts (run first)
python 06_analysis/01_nature2025_overlap.py        # Consortium comparison
python 06_analysis/03_generate_supp_tables_S3_S4.py
python 06_analysis/04_generate_supp_table_lambdaGC.py
python 06_analysis/02_generate_supp_tables_nature2025.py
python 06_analysis/13_ptv_k_floor_sensitivity.py && python 06_analysis/14_generate_supp_table_S2.py
python 06_analysis/15_generate_supp_table_S1.py

cd 07_figures && python figure1.py                 # etc.
```

Run `09_threshold_sensitivity.py` before the table generators: it produces the
realised test counts that `analysis_config.py` checks its hard-coded thresholds
against.

The β-equality overlay in Figure 3 is drawn only when `RHEO_EQ_FILE` points at
the output of `06_analysis/06_rheostat_beta_equality.py`; without it the figure
renders the population-level sign/binomial story only.

---

## Citation

If you use this code, please cite:

> Chaldebas M, Ponsin K, Mourelatos HA, Seeleuthner Y, Conil C, Bohlen J,
> Casanova J-L, Zhang P, Cobat A. Mechanistic 5′UTR Variant Scoring Expands Rare
> Variant Discovery in the UK Biobank. *medRxiv* (2026).
> doi: [10.64898/2026.09.09.26362607](https://doi.org/10.64898/2026.09.09.26362607)

and, for the classifier itself:

> Chaldebas M, *et al.* Genome-wide detection of human 5′UTR variants that impact
> protein translation. *Am J Hum Genet.* 2026;113(4):809–827.
> doi: [10.1016/j.ajhg.2026.02.020](https://doi.org/10.1016/j.ajhg.2026.02.020)

---

## License

Released under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License ([CC BY-NC-ND 4.0](LICENSE)), matching the license of the
5ULTRA classifier repository.

This license covers the code in this repository only. It does not extend to UK
Biobank data, which remain governed by the UK Biobank Material Transfer
Agreement, nor to third-party summary data reproduced from the cited sources.

---

## Contact

> **Developer:** [Matthieu Chaldebas, Ph.D. candidate](https://mchaldebas.github.io)
> **Email:** <mchaldebas@rockefeller.edu>
> **Laboratory:** St. Giles Laboratory of Human Genetics of Infectious Diseases
> **Institutions:** The Rockefeller University, New York, NY, USA — and
> Laboratory of Human Genetics of Infectious Diseases, INSERM U1163,
> Institut Imagine, Université Paris Cité, Paris, France

Questions about the code: open an issue on this repository.
