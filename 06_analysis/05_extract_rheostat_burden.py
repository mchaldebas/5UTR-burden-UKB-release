#!/usr/bin/env python
"""
05_extract_rheostat_burden.py   (run on the cluster — needs the WGS pgen)
-------------------------------------------------------------------------------
Build per-individual Mirror UP / DN burden values for the rheostat genes, so the
DN-vs-UP beta-equality test (Aurelie comment) can be run portably afterwards by
06_rheostat_beta_equality.py.

For each gene it reproduces REGENIE's discovery burden (--build-mask sum
--weights-col 4) separately for the UP and DN masks:

    burden_UP[i]  = sum over UP variants v of  W_PHRED_v * dosage_v[i]
    burden_DN[i]  = sum over DN variants v of  W_PHRED_v * dosage_v[i]

(The absolute scale need not match REGENIE to the digit; the equality test only
needs UP and DN built the same way, which they are.)

Variants, tags (UP/DN) and weights are read from the cohort's Mirror anno file
(anno_mirror_<spec>.anno : ID  gene  UP|DN  W_PHRED). Genotypes are pulled from
the per-chromosome 5'UTR pgen with plink2 --export A.

Inputs:
  --genes     file whose FIRST column is the rheostat gene symbols (a gene list
              or a gene<TAB>pheno pairs file both work)
  --cohort    nfe / nonnfe / afr / sas / eas / oth   (selects keep list + annodir)
  --spec      rare | af5   (which Mirror anno spectrum; default af5)

Output: rheostat_burden_<cohort>_<spec>.tsv.gz  (FID IID gene burden_UP burden_DN)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--spec", default="af5", choices=["rare", "af5"])
    ap.add_argument("--annodir", default=None)
    ap.add_argument("--pgen-dir", default="./final_complete_set")
    ap.add_argument("--keep", default=None)
    ap.add_argument("--plink2", default="plink2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    annodir = args.annodir or ("." if args.cohort == "nfe" else f"anno_{args.cohort}")
    keep = args.keep or f"{args.cohort}_ids_numeric.txt"
    out = args.out or f"rheostat_burden_{args.cohort}_{args.spec}.tsv.gz"
    anno_path = os.path.join(annodir, f"anno_mirror_{args.spec}.anno")
    for p in (args.genes, keep, anno_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing input {p}")

    genes = pd.read_csv(args.genes, sep=r"\s+", header=None)[0].astype(str).unique().tolist()
    print(f"Rheostat genes: {len(genes)}")

    anno = pd.read_csv(anno_path, sep=r"\s+", header=None,
                       names=["ID", "gene", "tag", "W"])
    anno = anno[anno["gene"].isin(genes)].copy()
    anno["chrom"] = anno["ID"].str.split(":").str[0]
    missing = sorted(set(genes) - set(anno["gene"]))
    if missing:
        print(f"  [WARN] {len(missing)} genes absent from {anno_path}: {missing[:10]}")
    print(f"  Mirror variants for these genes: {len(anno):,} "
          f"(UP {sum(anno['tag']=='UP'):,} / DN {sum(anno['tag']=='DN'):,})")

    # --- extract dosages per chromosome via plink2 --export A ----------------
    dosage_cols = {}   # variant ID -> per-IID dosage Series
    iid_index = None
    for chrom, grp in anno.groupby("chrom"):
        pgen = os.path.join(args.pgen_dir, f"chr{chrom}_5utr_complete")
        if not os.path.exists(pgen + ".pgen"):
            print(f"  [WARN] no pgen for chr{chrom} ({pgen}); skipping its variants")
            continue
        with tempfile.TemporaryDirectory() as td:
            idfile = os.path.join(td, "ids.txt")
            grp["ID"].drop_duplicates().to_csv(idfile, index=False, header=False)
            pref = os.path.join(td, "ext")
            cmd = [args.plink2, "--pfile", pgen, "--keep", keep,
                   "--extract", idfile, "--export", "A", "--out", pref]
            r = subprocess.run(cmd, capture_output=True, text=True)
            raw = pref + ".raw"
            if not os.path.exists(raw):
                print(f"  [WARN] chr{chrom}: plink2 produced no .raw\n{r.stderr[-400:]}")
                continue
            df = pd.read_csv(raw, sep=r"\s+")
            meta = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
            var_cols = [c for c in df.columns if c not in meta]
            if iid_index is None:
                iid_index = df[["FID", "IID"]].copy()
            df = df.set_index("IID")
            for c in var_cols:
                vid = c.rsplit("_", 1)[0]   # strip "_<countedallele>"; IDs use ':' not '_'
                dosage_cols[vid] = df[c].reindex(iid_index["IID"].values).fillna(0).values
        print(f"  chr{chrom}: extracted {len(var_cols)} variants")

    if iid_index is None:
        sys.exit("ERROR: no genotypes extracted (check pgen dir / keep list).")

    # --- weighted UP / DN burden per gene ------------------------------------
    rows = []
    n = len(iid_index)
    for gene, grp in anno.groupby("gene"):
        bUP = np.zeros(n); bDN = np.zeros(n)
        for _, v in grp.iterrows():
            d = dosage_cols.get(v["ID"])
            if d is None:      # variant not in genotypes (filtered/absent)
                continue
            (bUP if v["tag"] == "UP" else bDN)[:] += float(v["W"]) * d
        gdf = iid_index.copy()
        gdf["gene"] = gene
        gdf["burden_UP"] = bUP
        gdf["burden_DN"] = bDN
        rows.append(gdf)

    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out, sep="\t", index=False, compression="gzip")
    print(f"\nWrote {out}  ({res['gene'].nunique()} genes x {n:,} samples)")

    # gene -> chromosome sidecar (for the optional LOCO offset in script 06)
    gc = anno[["gene", "chrom"]].drop_duplicates()
    gc_path = "rheostat_gene_chrom.tsv"
    gc.to_csv(gc_path, sep="\t", index=False)
    print(f"Wrote {gc_path}  ({len(gc)} gene-chrom rows)")


if __name__ == "__main__":
    main()
