"""
Threshold sensitivity: how do discovery counts change under an exact
Bonferroni correction instead of the 1e-6 / N_models construction?

Reviewer point (A. Cobat): the 1e-6 base is arbitrary; correct by the exact
number of tests actually performed.

Three schemes are compared:

  CURRENT   -log10P > 6.0 + log10(K)        K = 8 (5ULTRA), 4 (CADD)
            i.e. p < 1.25e-7 (5ULTRA), 2.5e-7 (CADD)

  EXACT     p < 0.05 / N_tests_in_suite      computed from the data
            each suite judged against its own test count

  POOLED    p < 0.05 / (N_5ULTRA + N_CADD)   one bar for both suites
            the only scheme under which the head-to-head discovery
            comparison uses an identical threshold

Note on CURRENT: dividing by the number of *models* under-corrects the suite
that has more models and over-corrects the one that has fewer, because the
number of tests scales with the number of models. 5ULTRA is therefore held to
a more permissive bar than CADD under CURRENT. EXACT and POOLED both remove
that asymmetry.

Usage:
    DATA_DIR=/path/to/summary-statistics python 09_threshold_sensitivity.py

Outputs (written to CWD):
    threshold_sensitivity_summary.tsv   counts under each scheme
    threshold_sensitivity_delta.tsv     pairs gained/lost vs CURRENT
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", 'data'))
FILE     = DATA_DIR / "Master_Results_Clean.csv.gz"
TRUE_K   = 5.0
ALPHA    = 0.05

MODELS_5ULTRA   = {"5U_Logic", "5ULTRA_Binary", "5ULTRA_Weighted", "Flux_Joint"}
MODELS_CADD     = {"CADD_Binary", "CADD_Weighted"}
MODELS_ABLATION = {"Ablation", "Ablation_Weighted"}

# ---------------------------------------------------------------- load
print(f"Loading {FILE} …")
df = pd.read_csv(FILE)
n_raw = len(df)
df = df[df["true_k"] >= TRUE_K].copy()
df["pheno"] = pd.to_numeric(df["pheno"], errors="coerce")

df["suite"] = "Other"
df.loc[df["model"].isin(MODELS_5ULTRA),   "suite"] = "5ULTRA"
df.loc[df["model"].isin(MODELS_CADD),     "suite"] = "CADD"
df.loc[df["model"].isin(MODELS_ABLATION), "suite"] = "Ablation"

print(f"  rows total {n_raw:,} -> {len(df):,} at true_k >= {TRUE_K:g}")

# ------------------------------------------------- test counts per suite
n_tests = df.groupby("suite").size()
N_U = int(n_tests.get("5ULTRA", 0))
N_C = int(n_tests.get("CADD", 0))
N_A = int(n_tests.get("Ablation", 0))

n_pairs = df.groupby("suite").apply(
    lambda s: s.groupby(["gene", "pheno"]).ngroups, include_groups=False
)
n_genes = df.groupby("suite")["gene"].nunique()
n_phenos = df["pheno"].nunique()

print("\nTests actually performed (true_k >= 5):")
for s in ["5ULTRA", "CADD", "Ablation"]:
    if s in n_tests.index:
        print(f"  {s:<9} {n_tests[s]:>9,} tests   "
              f"{n_pairs[s]:>8,} gene-phenotype pairs   "
              f"{n_genes[s]:>6,} genes")
print(f"  phenotypes: {n_phenos}")

# ------------------------------------------------------------ thresholds
def mlog10(p):
    return -np.log10(p)

schemes = {
    "CURRENT": {
        "5ULTRA": 6.0 + np.log10(8),
        "CADD":   6.0 + np.log10(4),
        "Ablation": 6.0 + np.log10(4),
    },
    "EXACT": {
        "5ULTRA": mlog10(ALPHA / N_U),
        "CADD":   mlog10(ALPHA / N_C),
        "Ablation": mlog10(ALPHA / N_A) if N_A else np.nan,
    },
    "POOLED": {
        "5ULTRA": mlog10(ALPHA / (N_U + N_C)),
        "CADD":   mlog10(ALPHA / (N_U + N_C)),
        "Ablation": mlog10(ALPHA / (N_U + N_C)),
    },
}

print("\nThresholds (-log10 P):")
hdr = f"  {'scheme':<9}" + "".join(f"{s:>22}" for s in ["5ULTRA", "CADD"])
print(hdr)
for name, th in schemes.items():
    row = f"  {name:<9}"
    for s in ["5ULTRA", "CADD"]:
        row += f"{th[s]:>13.3f} (p<{10**-th[s]:.2e})".rjust(22)
    print(row)

# ------------------------------------------------------- best model per pair
best = {
    s: df[df["suite"] == s]
         .sort_values("logp", ascending=False)
         .drop_duplicates(["gene", "pheno"])
    for s in ["5ULTRA", "CADD", "Ablation"]
}

def hits(suite, thr):
    b = best[suite]
    return set(
        b.loc[b["logp"] > thr, ["gene", "pheno"]]
         .itertuples(index=False, name=None)
    )

# ---------------------------------------------------------------- compare
rows = []
hitsets = {}
for name, th in schemes.items():
    u = hits("5ULTRA", th["5ULTRA"])
    c = hits("CADD", th["CADD"])
    a = hits("Ablation", th["Ablation"]) if N_A else set()
    hitsets[name] = {"5ULTRA": u, "CADD": c, "Ablation": a}
    rows.append({
        "scheme": name,
        "thr_5ULTRA_mlog10": round(th["5ULTRA"], 3),
        "thr_CADD_mlog10": round(th["CADD"], 3),
        "n_5ULTRA": len(u),
        "n_CADD": len(c),
        "n_shared": len(u & c),
        "n_5ULTRA_exclusive": len(u - c),
        "n_CADD_exclusive": len(c - u),
        "n_ablation": len(a),
        "ratio_5ULTRA_over_CADD": round(len(u) / len(c), 3) if c else np.nan,
        "n_genes_5ULTRA": len({g for g, _ in u}),
    })

summary = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(summary.to_string(index=False))
print("=" * 78)

summary.to_csv("threshold_sensitivity_summary.tsv", sep="\t", index=False)

# --------------------------------------------------- gained / lost vs CURRENT
delta_rows = []
for name in ["EXACT", "POOLED"]:
    for suite in ["5ULTRA", "CADD"]:
        cur = hitsets["CURRENT"][suite]
        new = hitsets[name][suite]
        for gp in sorted(cur - new):
            delta_rows.append({"scheme": name, "suite": suite, "change": "lost",
                               "gene": gp[0], "pheno": gp[1]})
        for gp in sorted(new - cur):
            delta_rows.append({"scheme": name, "suite": suite, "change": "gained",
                               "gene": gp[0], "pheno": gp[1]})

delta = pd.DataFrame(delta_rows)
if len(delta):
    # attach the logp of the best model so it is obvious how marginal each is
    lookup = pd.concat(
        [best[s].assign(suite=s)[["suite", "gene", "pheno", "model", "logp", "true_k"]]
         for s in ["5ULTRA", "CADD"]]
    )
    delta = delta.merge(lookup, on=["suite", "gene", "pheno"], how="left")
    delta = delta.sort_values(["scheme", "suite", "change", "logp"],
                              ascending=[True, True, True, False])
    print("\nPairs changing status vs CURRENT:")
    print(delta.to_string(index=False))
else:
    print("\nNo pairs change status.")

delta.to_csv("threshold_sensitivity_delta.tsv", sep="\t", index=False)

print("\nWrote threshold_sensitivity_summary.tsv and threshold_sensitivity_delta.tsv")
print("\nFor the manuscript, the arithmetic to quote is:")
print(f"  theoretical 5ULTRA tests : n_genes x 8 models x {n_phenos} phenotypes")
print(f"  retained at true_k >= 5  : {N_U:,}  -> alpha = 0.05/{N_U:,} = {ALPHA/N_U:.3e}")
print(f"  retained CADD            : {N_C:,}  -> alpha = 0.05/{N_C:,} = {ALPHA/N_C:.3e}")
print(f"  pooled                   : {N_U+N_C:,}  -> alpha = {ALPHA/(N_U+N_C):.3e}")
