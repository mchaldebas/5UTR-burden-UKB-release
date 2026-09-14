"""
Aurelie comment #13: the CADD associations lost to ablation — are they
significant in 5ULTRA?

The ablation removes every 5ULTRA-annotated variant from the CADD masks and
CADD's genome-wide significant count falls from 38 to 30. The paragraph reads
that fall as evidence that the lost signal IS uORF/Kozak biology. That inference
is only sound if 5ULTRA itself detects those same gene-phenotype pairs; if it
does not, the ablation shows only that the variants mattered to CADD.

This script identifies the pairs CADD loses (and any it gains) and reports, for
each, the best 5ULTRA result: genome-wide significant, nominally significant, or
nothing.

Usage:
    python 06_analysis/16_ablated_pairs_in_5ultra.py
"""

import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis_config import (MASTER_CSV, TRUE_K_MIN, MODELS_5ULTRA, MODELS_CADD,
                             MODELS_ABLATION, THRESH_U, THRESH_C, THRESH_A,
                             describe)

NOMINAL = 3.0   # -log10P, the nominal tier used elsewhere in the paper

print(describe())
print(f"Loading {MASTER_CSV} …")
raw = pd.read_csv(MASTER_CSV)
raw["pheno"] = pd.to_numeric(raw["pheno"], errors="coerce")
df = raw[raw["true_k"] >= TRUE_K_MIN].copy()


def best(models):
    return (df[df["model"].isin(models)]
            .sort_values("logp", ascending=False)
            .drop_duplicates(["gene", "pheno"])
            .set_index(["gene", "pheno"]))


u, c, a = best(MODELS_5ULTRA), best(MODELS_CADD), best(MODELS_ABLATION)

c_hits = set(c.index[c["logp"] > THRESH_C])
a_hits = set(a.index[a["logp"] > THRESH_A])
lost, gained, kept = c_hits - a_hits, a_hits - c_hits, c_hits & a_hits

print(f"\nCADD baseline {len(c_hits)}   after ablation {len(a_hits)}")
print(f"  retained {len(kept)}   lost {len(lost)}   newly significant {len(gained)}")

rows = []
for label, pairs in (("lost", sorted(lost)), ("gained", sorted(gained))):
    for gp in pairs:
        in_u = gp in u.index
        u_lp = float(u.loc[gp, "logp"]) if in_u else np.nan
        if not in_u:
            status = "not tested by 5ULTRA"
        elif u_lp > THRESH_U:
            status = "5ULTRA genome-wide significant"
        elif u_lp > NOMINAL:
            status = "5ULTRA nominal only"
        else:
            status = "no 5ULTRA signal"
        rows.append({
            "change": label, "gene": gp[0], "pheno": int(gp[1]),
            "cadd_logp": round(float(c.loc[gp, "logp"]), 2) if gp in c.index else np.nan,
            "ablation_logp": round(float(a.loc[gp, "logp"]), 2) if gp in a.index else np.nan,
            "u_model": u.loc[gp, "model"] if in_u else "",
            "u_logp": round(u_lp, 2) if in_u else np.nan,
            "u_status": status,
        })

out = pd.DataFrame(rows)
if len(out):
    print("\n" + out.to_string(index=False))
    out.to_csv("ablated_pairs_in_5ultra.tsv", sep="\t", index=False)
    print("\nWrote ablated_pairs_in_5ultra.tsv")

    l = out[out["change"] == "lost"]
    n_gws = int((l["u_status"] == "5ULTRA genome-wide significant").sum())
    n_nom = int((l["u_status"] == "5ULTRA nominal only").sum())
    print(f"\nOf the {len(l)} associations CADD loses to ablation:")
    print(f"  {n_gws} are genome-wide significant in 5ULTRA")
    print(f"  {n_nom} reach only nominal significance (-log10P > {NOMINAL:g})")
    print(f"  {len(l) - n_gws - n_nom} have no 5ULTRA signal")
    print("""
Reading this: a high proportion recovered by 5ULTRA supports the claim that the
ablated signal is uORF/Kozak biology. A low proportion means the ablation shows
the variants mattered to CADD without showing 5ULTRA captures them — in which
case the Results sentence should say the narrower thing.""")
else:
    print("\nNo pairs changed status under ablation.")
