"""
Is the 5ULTRA-vs-CADD discovery gap a detection difference or a coverage difference?

Background (from 10_check_k_filter_symmetry.py): the k >= 5 filter leaves the
5ULTRA suite testing ~3,145 genes and the CADD suite ~615. So a 5ULTRA-exclusive
association can arise two very different ways:

  MISSED   CADD was tested on that gene-phenotype pair and did not reach
           significance  -> a real detection win for 5ULTRA.

  UNTESTED CADD had no surviving test for that pair (all its models fell below
           k >= 5) -> a coverage difference, not a detection difference.

The wording of the Results depends on which dominates. This script splits the
5ULTRA-exclusive hits into those two buckets and, for the MISSED ones, reports
how close CADD actually came.

Usage:
    DATA_DIR="/path/to/UKB-data" python 06_analysis/11_exclusive_hits_coverage.py
    THRESH_SCHEME=exact DATA_DIR=... python 06_analysis/11_exclusive_hits_coverage.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis_config import (DATA_DIR, MASTER_CSV, TRUE_K_MIN,
                             MODELS_5ULTRA, MODELS_CADD,
                             THRESH_U, THRESH_C, SCHEME, describe)

FILE   = MASTER_CSV
TRUE_K = TRUE_K_MIN
ALPHA  = 0.05
print(describe())

print(f"Loading {FILE} …")
raw = pd.read_csv(FILE)
raw["pheno"] = pd.to_numeric(raw["pheno"], errors="coerce")
raw["suite"] = np.where(raw["model"].isin(MODELS_5ULTRA), "5ULTRA",
                np.where(raw["model"].isin(MODELS_CADD), "CADD", "Other"))

df = raw[raw["true_k"] >= TRUE_K].copy()
N_U = int((df["suite"] == "5ULTRA").sum())
N_C = int((df["suite"] == "CADD").sum())

print(f"tests at k >= {TRUE_K:g}:  5ULTRA {N_U:,}   CADD {N_C:,}")

best = lambda s: (df[df["suite"] == s]
                  .sort_values("logp", ascending=False)
                  .drop_duplicates(["gene", "pheno"])
                  .set_index(["gene", "pheno"]))
u_best, c_best = best("5ULTRA"), best("CADD")

u_hits = set(u_best.index[u_best["logp"] > THRESH_U])
c_hits = set(c_best.index[c_best["logp"] > THRESH_C])
u_excl = sorted(u_hits - c_hits)
print(f"\n5ULTRA hits {len(u_hits)}   CADD hits {len(c_hits)}   "
      f"5ULTRA-exclusive {len(u_excl)}")

# CADD tests that existed BEFORE the k filter, for the same pairs
c_all = (raw[raw["suite"] == "CADD"]
         .sort_values("logp", ascending=False)
         .drop_duplicates(["gene", "pheno"])
         .set_index(["gene", "pheno"]))

rows = []
for gp in u_excl:
    tested = gp in c_best.index
    rec = {
        "gene": gp[0], "pheno": gp[1],
        "u_model": u_best.loc[gp, "model"],
        "u_logp": round(u_best.loc[gp, "logp"], 3),
        "u_true_k": round(u_best.loc[gp, "true_k"], 2),
        "cadd_status": "MISSED" if tested else "UNTESTED",
        "cadd_logp": round(c_best.loc[gp, "logp"], 3) if tested else np.nan,
        "cadd_true_k": round(c_best.loc[gp, "true_k"], 2) if tested else np.nan,
    }
    if not tested and gp in c_all.index:
        # CADD did run here, but every model fell below k >= 5
        rec["cadd_logp_prefilter"] = round(c_all.loc[gp, "logp"], 3)
        rec["cadd_true_k_prefilter"] = round(c_all.loc[gp, "true_k"], 3)
    rows.append(rec)

out = pd.DataFrame(rows).sort_values(["cadd_status", "u_logp"],
                                     ascending=[True, False])
print("\n" + out.to_string(index=False))

n_missed = (out["cadd_status"] == "MISSED").sum()
n_untest = (out["cadd_status"] == "UNTESTED").sum()
print(f"\n  MISSED   (CADD tested, not significant) : {n_missed}")
print(f"  UNTESTED (no CADD test survived k>=5)   : {n_untest}")

if n_missed:
    m = out[out["cadd_status"] == "MISSED"]
    print(f"\n  Of the {n_missed} MISSED, CADD -log10P: "
          f"median {m['cadd_logp'].median():.2f}, max {m['cadd_logp'].max():.2f}"
          f"  ({(m['cadd_logp'] > 3).sum()} above nominal -log10P 3)")

out.to_csv(f"exclusive_hits_coverage_{SCHEME}.tsv", sep="\t", index=False)
print(f"\nWrote exclusive_hits_coverage_{SCHEME}.tsv")

# --- mirror check: are CADD-exclusive hits untested by 5ULTRA? -----------
c_excl = sorted(c_hits - u_hits)
n_c_untested = sum(1 for gp in c_excl if gp not in u_best.index)
print(f"\nMirror check: {len(c_excl)} CADD-exclusive hits, "
      f"{n_c_untested} of them untested by 5ULTRA")
