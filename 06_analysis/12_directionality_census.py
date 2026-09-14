"""
Directionality census over ALL informative gene-phenotype pairs.

Aurelie comment #4: a pair must not be called a "rheostat" on the sign of the two
coefficients alone. The categories must be defined on the SIGNIFICANCE of each
arm (see 07_figures/rheostat_classify.py, which implements exactly that):

    rheostat    opposite sign AND both arms significant
    DN-driven   only the DN arm significant   (sign irrelevant)
    UP-driven   only the UP arm significant   (sign irrelevant)
    fragility   same sign AND both arms significant
    neither     neither arm significant

Two scopes are reported, and the manuscript must keep them apart:

  MARGINAL  every informative pair (k_UP >= 5 and k_DN >= 5, af5), classified on
            the two separate Directional-model burden tests. This is the census.
  JOINT     the highlighted pairs refit in one model by
            06_rheostat_beta_equality.py, which additionally supplies the
            symmetry / equality Wald contrasts. That file drives Figure 3.

Because "significant" is a choice, the census is reported at three alphas:
nominal 0.05, Bonferroni 0.05/n_pairs, and Benjamini-Hochberg FDR 5% (applied
across all 2*n_pairs arm tests). Pick one for the manuscript and state it.

The sign-only statistics (proportion with beta_DN*beta_UP < 0) are printed too,
since they remain reportable as a population-level description -- but they do
NOT license any per-pair claim of opposing effects.

Usage:
    DATA_DIR="/path/to/UKB-data" python 06_analysis/12_directionality_census.py
"""

import os
import sys
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "07_figures"))
from analysis_config import DATA_DIR, MASTER_CSV, TRUE_K_MIN
from rheostat_classify import classify, count_table, CATEGORY_ORDER

SPECTRUM = "af5"

# ------------------------------------------------------------------ load
print(f"Loading {MASTER_CSV} …")
df = pd.read_csv(MASTER_CSV)
df = df[(df["true_k"] >= TRUE_K_MIN) & (df["spectrum"] == SPECTRUM)].copy()

logic = df[df["model"] == "5U_Logic"].copy()          # "Directional" (Mirror UP/DN)
dn = logic[logic["mask"].str.contains("DN", case=False, na=False)]
up = logic[logic["mask"].str.contains("UP", case=False, na=False)]

pairs = pd.merge(dn, up, on=["gene", "pheno"], suffixes=("_DN", "_UP"))
pairs["p_DN"] = np.power(10.0, -pairs["logp_DN"])
pairs["p_UP"] = np.power(10.0, -pairs["logp_UP"])
pairs = pairs.rename(columns={"beta_DN": "beta_DN", "beta_UP": "beta_UP"})
n = len(pairs)
print(f"Informative pairs (k_UP >= {TRUE_K_MIN:g} and k_DN >= {TRUE_K_MIN:g}, {SPECTRUM}): {n:,}")

# --------------------------------------------------- sign-only description
opposite = (pairs["beta_DN"] * pairs["beta_UP"]) < 0
n_opp = int(opposite.sum())
p_binom = stats.binomtest(n_opp, n, 0.5).pvalue
print(f"\nSign-only (population-level description, NOT a classification):")
print(f"  opposite sign: {n_opp:,}/{n:,} = {100*n_opp/n:.1f}%  "
      f"(exact binomial vs 50%: P = {p_binom:.3g})")

for t in (2.0, 3.0, 4.0):
    sub = pairs[pairs[["logp_DN", "logp_UP"]].max(axis=1) >= t]
    if len(sub) == 0:
        continue
    o = int(((sub["beta_DN"] * sub["beta_UP"]) < 0).sum())
    pv = stats.binomtest(o, len(sub), 0.5).pvalue
    print(f"  max(-log10P) >= {t:.0f}: {o}/{len(sub)} = {100*o/len(sub):.0f}%  (P = {pv:.2g})")

# --------------------------------------------------------------- alphas
def bh_threshold(pvals, q=0.05):
    """Largest p passing Benjamini-Hochberg at level q; 0 if none pass."""
    p = np.sort(np.asarray(pvals))
    m = len(p)
    crit = q * np.arange(1, m + 1) / m
    passing = p <= crit
    return float(p[passing].max()) if passing.any() else 0.0

all_arm_p = np.concatenate([pairs["p_DN"].values, pairs["p_UP"].values])
alphas = {
    "nominal 0.05": 0.05,
    f"Bonferroni 0.05/{n:,}": 0.05 / n,
    "BH-FDR 5% (both arms)": bh_threshold(all_arm_p, 0.05),
}

print("\nPer-arm significance thresholds compared:")
for lab, a in alphas.items():
    print(f"  {lab:<28} alpha = {a:.3e}")

# ------------------------------------------------------------- classify
out_frames = []
print("\n" + "=" * 72)
for lab, a in alphas.items():
    cl = classify(pairs, alpha=a)
    counts = (cl["category"].value_counts()
              .reindex(CATEGORY_ORDER).fillna(0).astype(int))
    print(f"\n--- {lab}  (alpha = {a:.3e})")
    for cat in CATEGORY_ORDER:
        c = int(counts.get(cat, 0))
        if cat == "untested" and c == 0:
            continue
        print(f"    {cat:<11} {c:>6,}  ({100*c/n:5.2f}%)")
    ratio = (counts.get("DN-driven", 0) /
             counts.get("UP-driven", 1) if counts.get("UP-driven", 0) else np.nan)
    print(f"    DN:UP ratio = {ratio:.2f}" if np.isfinite(ratio) else "    DN:UP ratio = n/a")
    tmp = counts.rename("n").to_frame()
    tmp["alpha_label"] = lab
    tmp["alpha"] = a
    out_frames.append(tmp.reset_index().rename(columns={"index": "category"}))
print("\n" + "=" * 72)

pd.concat(out_frames).to_csv("directionality_census.tsv", sep="\t", index=False)

# Per-pair output at the strictest alpha that still yields signal, for the supp table
primary_label = f"Bonferroni 0.05/{n:,}"
cl = classify(pairs, alpha=alphas[primary_label])
keep = ["gene", "pheno", "beta_DN", "p_DN", "logp_DN", "beta_UP", "p_UP", "logp_UP",
        "true_k_DN", "true_k_UP", "category", "category_detail"]
keep = [c for c in keep if c in cl.columns]
cl[keep].sort_values("p_DN").to_csv("directionality_perpair.tsv", sep="\t", index=False)

print("\nWrote directionality_census.tsv and directionality_perpair.tsv")
print("\nReminder for the text: rejecting beta_UP = beta_DN means the coefficients")
print("DIFFER, not that they OPPOSE. Only the 'rheostat' category licenses the")
print("word 'opposing', and only when both arms are individually significant.")
