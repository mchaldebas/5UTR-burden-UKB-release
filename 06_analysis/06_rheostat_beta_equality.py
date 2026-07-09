#!/usr/bin/env python
"""
06_rheostat_beta_equality.py   (portable — numpy / scipy / pandas only)
-------------------------------------------------------------------------------
DN-vs-UP beta-equality test for the rheostat genes (Aurelie comment): does a
single combined effect fit as well as separate UP and DN effects?

For each (gene, phenotype) it fits ONE Gaussian model on the RINT-transformed
phenotype, adjusting for the standard covariates:

    y ~ covars + burden_UP + burden_DN

and tests two linear contrasts on (beta_UP, beta_DN) with their exact
covariance (Wald):

    symmetry / rheostat : beta_UP + beta_DN = 0   (equal magnitude, OPPOSITE sign)
    plain equality      : beta_UP - beta_DN = 0   (same value)

A rheostat gene is one where each direction is individually significant with
OPPOSITE signs AND the symmetry test is NOT rejected (p_symmetry large => a
single magnitude with mirror signs fits). A small p_symmetry means the up- and
down-regulating variants do NOT act as a clean mirror. p_equality answers
Aurelie's literal "are the betas equal" question. (Wald and the LRT of a
one-beta vs two-beta model are asymptotically equivalent; the contrast form
gives both tests from a single fit and uses the UP-DN covariance correctly.)

Optional --loco-pred (Step-1 *_pred.list): for each gene-phenotype, subtract the
REGENIE Step-1 LOCO prediction for that gene's chromosome before fitting, matching
how REGENIE removes the polygenic background (full fidelity). Needs the gene->chrom
sidecar from 05_extract_rheostat_burden.py. If omitted, covariates only — fine for
the relative UP-vs-DN contrast, which the common polygenic offset barely moves.

Inputs:
  --burden  rheostat_burden_<cohort>_<spec>.tsv.gz  (from 05_extract_rheostat_burden.py)
  --pheno   pheno_covar_final_clean_<cohort>.txt
  --pairs   gene<TAB>pheno file (which gene-phenotype pairs to test)
Output: rheostat_beta_equality_<cohort>.tsv
"""

import argparse
import gzip
import numpy as np
import pandas as pd
from scipy import stats


def load_loco(path):
    """Read a REGENIE Step-1 .loco(.gz) prediction file for one phenotype.

    Format: whitespace-delimited with a header row 'FID IID <chr1> <chr2> ...'
    and one row PER INDIVIDUAL (columns = chromosomes). Returns a DataFrame
    indexed by integer IID, columns = chromosome as int (X -> 23). Robust to
    FID being present or absent. Chromosomes not analysed in Step 1 (e.g. X if
    Step 1 used autosomes only) simply won't appear as columns.
    """
    df = pd.read_csv(path, sep=r"\s+")          # pandas handles .gz
    upper = {str(c).upper(): c for c in df.columns}
    if "IID" not in upper:
        raise ValueError(f"{path}: no IID column; header={list(df.columns)[:6]}")
    iid_col = upper["IID"]
    drop = {iid_col} | ({upper["FID"]} if "FID" in upper else set())
    chrom_cols = [c for c in df.columns if c not in drop]
    out = df.set_index(df[iid_col].astype("int64"))[chrom_cols]

    def norm(c):
        s = str(c).replace("chr", "")
        return 23 if s.upper() == "X" else int(s)

    out.columns = [norm(c) for c in chrom_cols]
    return out                                  # index=IID(int), cols=chrom(int)

# Superset of covariate names across cohorts; design() silently drops any not
# present. The harmonised non-NFE cohort files use wgs_batch + PC1-20; the NFE
# main file (pheno_covar_final_clean.txt) uses provider + PC1-10.
COVARS = (["age", "age_sq", "sex", "wgs_batch", "provider", "center"]
          + [f"PC{i}" for i in range(1, 21)])
CAT_COVARS = ["wgs_batch", "provider", "center"]


def rint(x):
    x = pd.Series(x).astype(float)
    r = x.rank(method="average")
    return stats.norm.ppf((r - 0.5) / r.notna().sum())


def design(df, extra_cols):
    """Build a model matrix: intercept + covariates (cats dummied) + extra_cols."""
    parts = [np.ones((len(df), 1))]
    for c in COVARS:
        if c not in df.columns:
            continue
        if c in CAT_COVARS:
            d = pd.get_dummies(df[c].astype("category"), drop_first=True, prefix=c)
            parts.append(d.to_numpy(dtype=float))
        else:
            parts.append(df[[c]].to_numpy(dtype=float))
    parts.append(df[extra_cols].to_numpy(dtype=float))
    X = np.hstack(parts)
    # drop zero-variance columns (keep intercept col 0)
    keep = [0] + [j for j in range(1, X.shape[1]) if np.nanstd(X[:, j]) > 0]
    return X[:, keep]


def ols_fit(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    n, p = X.shape
    sigma2 = rss / (n - p)
    cov = np.linalg.pinv(X.T @ X) * sigma2
    return beta, cov, rss


def contrast(beta, cov, a):
    """Wald test of a . beta = 0 for the last len(a) coefficients."""
    k = len(a)
    a = np.asarray(a, dtype=float)
    b = beta[-k:]
    V = cov[-k:, -k:]
    c = float(a @ b)
    var = float(a @ V @ a)
    z = c / np.sqrt(var) if var > 0 else np.nan
    return c, z, 2 * stats.norm.sf(abs(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burden", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--cohort", default="cohort")
    ap.add_argument("--loco-pred", default=None,
                    help="Step-1 *_pred.list; if given, subtract the LOCO "
                         "prediction for each gene's chromosome (full REGENIE fidelity)")
    ap.add_argument("--gene-chrom", default="rheostat_gene_chrom.tsv",
                    help="gene<TAB>chrom sidecar from 05_extract_rheostat_burden.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"rheostat_beta_equality_{args.cohort}.tsv"

    pairs = pd.read_csv(args.pairs, sep=r"\s+", header=None, names=["gene", "pheno"])
    need_genes = set(pairs["gene"].astype(str))

    # The burden table is LONG (gene x sample) and can be tens of GB. Stream it
    # in chunks and keep only the genes we will actually test, so we never hold
    # the whole file in memory.
    print(f"Streaming burden, keeping {len(need_genes)} test genes...")
    parts = []
    for ch in pd.read_csv(args.burden, sep="\t", chunksize=5_000_000):
        ch = ch[ch["gene"].astype(str).isin(need_genes)]
        if len(ch):
            parts.append(ch)
    burden = (pd.concat(parts, ignore_index=True) if parts else
              pd.DataFrame(columns=["FID", "IID", "gene", "burden_UP", "burden_DN"]))
    print(f"  loaded {len(burden):,} rows for "
          f"{burden['gene'].nunique()} of {len(need_genes)} genes")

    pheno = pd.read_csv(args.pheno, sep="\t")
    pheno["IID"] = pheno["IID"].astype("int64")
    burden["IID"] = burden["IID"].astype("int64")

    # Optional LOCO offset machinery.
    pred_map, gene_chrom, loco_cache = {}, {}, {}
    if args.loco_pred:
        pm = pd.read_csv(args.loco_pred, sep=r"\s+", header=None, names=["pheno", "path"])
        pred_map = dict(zip(pm["pheno"].astype(str), pm["path"]))
        gc = pd.read_csv(args.gene_chrom, sep=r"\s+")
        gene_chrom = {str(g): (23 if str(c).upper() == "X" else int(c))
                      for g, c in zip(gc["gene"], gc["chrom"])}
        print(f"LOCO offset ON ({len(pred_map)} pheno preds, {len(gene_chrom)} genes)")

    results = []
    for _, pr in pairs.iterrows():
        gene, ph = str(pr["gene"]), str(pr["pheno"])
        # Pairs file lists bare field IDs (30710); different pheno files name the
        # column p30710 (cohort files) or p30710_INT (NFE main, already RINT'd).
        # Resolve to whichever exists.
        cand = [ph, f"p{ph}", f"p{ph}_INT", f"{ph}_INT"]
        ph = next((c for c in cand if c in pheno.columns), None)
        if ph is None:
            print(f"  [WARN] pheno {pr['pheno']} not in file (tried {cand}); skip {gene}")
            continue
        b = burden[burden["gene"] == gene]
        if b.empty:
            print(f"  [WARN] no burden for gene {gene}; skip")
            continue

        keep_cols = ["IID", ph] + [c for c in COVARS if c in pheno.columns]
        d = b.merge(pheno[keep_cols], on="IID", how="inner")
        d = d.dropna(subset=[ph])
        if len(d) < 100:
            print(f"  [WARN] {gene}/{ph}: only {len(d)} samples; skip")
            continue

        # Optional: subtract the Step-1 LOCO prediction for this gene's chromosome.
        offset = None
        if args.loco_pred:
            if ph not in pred_map or gene not in gene_chrom:
                print(f"  [WARN] {gene}/{ph}: no LOCO pred/chrom; covariate-only.")
            else:
                if ph not in loco_cache:
                    loco_cache[ph] = load_loco(pred_map[ph])
                chrom = gene_chrom[gene]
                lc = loco_cache[ph]
                if chrom not in lc.columns:
                    print(f"  [WARN] {gene}/{ph}: chrom {chrom} not in LOCO "
                          f"(Step 1 autosomes only?); covariate-only.")
                else:
                    d = d[d["IID"].isin(lc.index)]
                    offset = lc.loc[d["IID"].values, chrom].to_numpy(dtype=float)

        y = np.asarray(rint(d[ph]), dtype=float)
        if offset is not None:
            y = y - offset

        X = design(d, ["burden_UP", "burden_DN"])   # UP, DN are the last 2 cols
        beta, cov, _ = ols_fit(X, y)
        n = len(y)
        bUP, bDN = beta[-2], beta[-1]
        seUP, seDN = np.sqrt(cov[-2, -2]), np.sqrt(cov[-1, -1])
        pUP = 2 * stats.norm.sf(abs(bUP / seUP))
        pDN = 2 * stats.norm.sf(abs(bDN / seDN))
        # symmetry / rheostat: beta_UP + beta_DN = 0 ; equality: beta_UP - beta_DN = 0
        _, _, p_sym = contrast(beta, cov, [1.0, 1.0])
        _, _, p_eq = contrast(beta, cov, [1.0, -1.0])

        results.append(dict(
            cohort=args.cohort, gene=gene, pheno=ph, n=n,
            beta_UP=bUP, se_UP=seUP, p_UP=pUP,
            beta_DN=bDN, se_DN=seDN, p_DN=pDN,
            opposite_sign=bool(bUP * bDN < 0),
            p_symmetry=p_sym, p_equality=p_eq))
        print(f"  {gene:12s} {ph:8s} n={n:>7}  bUP={bUP:+.3f}({pUP:.1e}) "
              f"bDN={bDN:+.3f}({pDN:.1e})  opp={bUP*bDN<0}  "
              f"p_sym={p_sym:.2g} p_eq={p_eq:.2g}")

    res = pd.DataFrame(results)
    res.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}  ({len(res)} gene-phenotype tests)")


if __name__ == "__main__":
    main()
