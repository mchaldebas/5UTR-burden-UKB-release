"""
Diagnostic: is the k >= 5 filter applied on the same quantity in both suites?

Why this matters
----------------
In 01_aggregate_results.py, `true_k` is a PHYSICAL CARRIER COUNT for masks with
is_physical == True (5U_Binary, CD_Binary, Binary_DN) and a WEIGHTED BURDEN SUM
for everything else, and it is propagated to non-physical rows by a merge on
(gene, mask, spectrum). So whether a given model is filtered on carriers or on a
weighted sum depends on whether it shares a mask name with a physical model.

A carrier-count bar of 5 and a weighted-sum bar of 5 are not the same bar. If the
CADD-family models are effectively gated on carrier counts while the 5ULTRA-family
models are gated on weighted sums, the two suites enter the discovery comparison
having been filtered at different stringencies -- which would matter for the
headline 5ULTRA-vs-CADD count.

Motivation: 09_threshold_sensitivity.py reports 3,145 genes tested in the 5ULTRA
suite vs 615 in the CADD suite. Both suites are built from the same 5'UTR variant
set, so a ~5x difference in gene coverage needs an explanation.

Usage:
    DATA_DIR="/path/to/UKB-data" python 06_analysis/10_check_k_filter_symmetry.py
"""

import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", 'data'))
FILE     = DATA_DIR / "Master_Results_Clean.csv.gz"
TRUE_K   = 5.0

print(f"Loading {FILE} …")
df = pd.read_csv(FILE)

print("\n--- is_physical by model (does this model's own k mean carriers?) ---")
print(pd.crosstab(df["model"], df["is_physical"]).to_string())

print("\n--- mask names per model (the merge key that propagates true_k) ---")
print(df.groupby("model")["mask"].unique().to_string())

print("\n--- did true_k come from a physical mask, or from the row's own k? ---")
# true_k was filled from physical_map where available; where it equals k and the
# row is not physical, it kept its own weighted sum.
df["true_k_is_own_k"] = (df["true_k"] == df["k"])
print(pd.crosstab(df["model"], [df["is_physical"], df["true_k_is_own_k"]]).to_string())

print("\n--- genes and pairs per model, before vs after the k >= 5 filter ---")
rows = []
for model, g in df.groupby("model"):
    kept = g[g["true_k"] >= TRUE_K]
    rows.append({
        "model": model,
        "is_physical": bool(g["is_physical"].iloc[0]),
        "genes_before": g["gene"].nunique(),
        "genes_after": kept["gene"].nunique(),
        "pairs_before": g.groupby(["gene", "pheno"]).ngroups,
        "pairs_after": kept.groupby(["gene", "pheno"]).ngroups,
        "pct_pairs_kept": round(100 * len(kept) / len(g), 1),
        "median_true_k": round(g["true_k"].median(), 2),
    })
print(pd.DataFrame(rows).sort_values("model").to_string(index=False))

print("\n--- distribution of true_k by model ---")
print(df.groupby("model")["true_k"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2).to_string())

print("""
How to read this
----------------
If the CADD_* rows show is_physical True (or inherit a physical true_k) while the
5ULTRA weighted / Mirror / Flux rows keep their own weighted-sum k, the two suites
are being filtered on different quantities and the k >= 5 cut is not comparable
across suites. The fix would be to gate both suites on the same quantity -- either
physical carrier count everywhere (preferred, it is interpretable), or the model's
own weighted burden everywhere.

If instead the gene-count gap is explained by something else (e.g. CADD scores
missing for many 5'UTR variants), that is a coverage statement that belongs in
Methods rather than a bug.
""")
