"""
How does the 5'UTR-DN vs exome-PTV correlation depend on the carrier floor?

Figure 4 uses k >= 15, a threshold that appears nowhere else in the pipeline
(everything else uses k >= 5) and carries no justification in the code. Aurelie
asks where the 23 pairs come from; the honest answer is:

    pairs with an unweighted 5'UTR-DN burden (af5) AND a published exome PTV
    statistic in the consortium file AND k >= 15

so both the intersection and the floor need stating. This script reports the
correlation across a range of floors so the choice can either be justified or
replaced with the k >= 5 used elsewhere.

It reproduces figure4.py's data path exactly: same two input files, same
Binary_DN / Binary_DN_AF5 selection, same merge keys.

Usage:
    DATA_DIR="/path/to/UKB-data" python 06_analysis/13_ptv_k_floor_sensitivity.py
"""

import os
import sys
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis_config import DATA_DIR

FILE_TRIAD = DATA_DIR / "Final_Triad_Comparison.csv.gz"
FILE_NAT   = DATA_DIR / "Nature_2025_Reproduction_Data.csv"

# Consortium damaging-missense / nonsynonymous CDS masks (as in figure4.py).
MISSENSE_MASKS = {"flexdmg", "flexnonsynmtr", "raredmg", "raredmgmtr"}

print(f"Loading {FILE_TRIAD.name} and {FILE_NAT.name} …")
df  = pd.read_csv(FILE_TRIAD)
nat = pd.read_csv(FILE_NAT)

nat["pheno"] = nat["phenotype"].astype(int)
nat["z_ptv"] = nat["beta.CDS_PTV"] / nat["se.CDS_PTV"]
nat["beta_missense"] = np.where(nat["model.CDS"].isin(MISSENSE_MASKS),
                                nat["beta.CDS"], np.nan)
nat_ref = nat[["gene", "pheno", "beta.CDS_PTV", "se.CDS_PTV", "z_ptv",
               "model.CDS", "beta_missense"]].copy()
nat_ref.columns = ["gene", "pheno", "beta_ptv", "se_ptv", "z_ptv",
                   "cds_model", "beta_missense"]

dn = df[(df["model"] == "Binary_DN") & (df["mask"] == "Binary_DN_AF5")].copy()
dn["z"] = dn["beta"] / dn["se"]

print("\n--- how the 23 pairs are reached ---")
print(f"  unweighted 5'UTR-DN (af5) gene-phenotype pairs tested : {len(dn):,}")
merged = pd.merge(dn, nat_ref, on=["gene", "pheno"])
print(f"  ... also carrying a consortium exome PTV statistic     : {len(merged):,}")
for floor in (0, 5, 10, 15, 20, 25):
    print(f"  ... and k >= {floor:<3d}                                    : "
          f"{(merged['k'] >= floor).sum():,}")

# UP mask, for the negative control at each floor
up = df[df["model"].astype(str).str.contains("Mirror_UP|UP", case=False, na=False)].copy()
if "beta" in up.columns and "se" in up.columns and len(up):
    up["z_up"] = up["beta"] / up["se"]
    up_ref = up[["gene", "pheno", "z_up"]].drop_duplicates(["gene", "pheno"])
else:
    up_ref = pd.DataFrame(columns=["gene", "pheno", "z_up"])

print("\n--- correlation vs carrier floor ---")
rows = []
for floor in (0, 5, 10, 15, 20, 25, 30):
    s = merged[merged["k"] >= floor]
    if len(s) < 3:
        continue
    r, p = stats.pearsonr(s["z_ptv"], s["z"])
    rho, prho = stats.spearmanr(s["z_ptv"], s["z"])
    conc = int((s["beta"] * s["beta_ptv"] > 0).sum())
    nmis = int(s["beta_missense"].notna().sum())
    u = s[["gene", "pheno", "z_ptv"]].merge(up_ref, on=["gene", "pheno"])
    if len(u) >= 3:
        r_up, p_up = stats.pearsonr(u["z_ptv"], u["z_up"])
    else:
        r_up, p_up = np.nan, np.nan
    rows.append({
        "k_floor": floor, "n_pairs": len(s),
        "pearson_r": round(r, 3), "pearson_p": f"{p:.2e}",
        "spearman_rho": round(rho, 3), "spearman_p": f"{prho:.2e}",
        "n_concordant": conc,
        "pct_concordant": round(100 * conc / len(s), 1),
        "n_with_missense": nmis,
        "n_UP_control": len(u),
        "r_UP_control": round(r_up, 3) if np.isfinite(r_up) else np.nan,
        "p_UP_control": round(p_up, 3) if np.isfinite(p_up) else np.nan,
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("ptv_k_floor_sensitivity.tsv", sep="\t", index=False)
print("\nWrote ptv_k_floor_sensitivity.tsv")

print("""
Reading this: if r stays in the same range and stays significant across floors,
the k >= 15 cut is a presentational choice, not a result-generating one, and the
manuscript can say so in a clause. If the correlation only appears at k >= 15,
that needs stating plainly instead.
""")
