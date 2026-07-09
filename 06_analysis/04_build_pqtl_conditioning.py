#!/usr/bin/env python
"""
04_build_pqtl_conditioning.py
-------------------------------------------------------------------------------
Pick each GWS gene's LEAD cis-pQTL from study pQTL summary statistics and emit
the inputs needed to condition the burden test on it (Aurelie #80):

  condition_list.txt   one variant ID (chr:pos:ref:alt) per line  -> --condition-list
  pqtl_positions.bed   chr  start(0-based)  end  gene             -> region extraction
  pqtl_lead_summary.tsv gene, variant, chr, pos, ref, alt, p, beta (audit)

Column names are configurable so it works with whatever your sumstats look like.
The variant ID is written as chr:pos:ref:alt (chr WITHOUT 'chr' by default, to
match your 5'UTR pgen IDs from --set-all-var-ids @:#:$r:$a); flip with --chr-prefix.

IMPORTANT — genome build: your WGS / burden pgen are GRCh38. If the pQTL
sumstats are GRCh37 (older studies often are), liftover the positions to 38
BEFORE extracting genotypes, or the IDs/positions won't match.

Usage:
    python 04_build_pqtl_conditioning.py --sumstats EUR.5UTRs.MANE.GENE.cis.tsv \
        --genes GP1BA,NELFCD,... --gene-col gene --chr-col chr --pos-col pos \
        --ref-col ref --alt-col alt --p-col p [--beta-col beta]
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumstats", required=True)
    ap.add_argument("--genes", required=True,
                    help="comma-separated GWS gene symbols (the 4 GWS genes)")
    ap.add_argument("--gene-col", default="gene")
    ap.add_argument("--chr-col", default="chr")
    ap.add_argument("--pos-col", default="pos")
    ap.add_argument("--ref-col", default="ref")
    ap.add_argument("--alt-col", default="alt")
    ap.add_argument("--p-col", default="p")
    ap.add_argument("--beta-col", default=None)
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--chr-prefix", action="store_true",
                    help="write IDs as chrN:... (default N:... to match the 5'UTR pgen)")
    args = ap.parse_args()

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    df = pd.read_csv(args.sumstats, sep=args.sep)
    for c in (args.gene_col, args.chr_col, args.pos_col, args.ref_col,
              args.alt_col, args.p_col):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not in sumstats; have: {list(df.columns)}")

    df = df[df[args.gene_col].astype(str).isin(genes)].copy()
    df[args.p_col] = pd.to_numeric(df[args.p_col], errors="coerce")
    df = df.dropna(subset=[args.p_col])

    leads = (df.sort_values(args.p_col)
               .groupby(args.gene_col, as_index=False)
               .first())

    def vid(r):
        chrom = str(r[args.chr_col]).replace("chr", "")
        if args.chr_prefix:
            chrom = "chr" + chrom
        return f"{chrom}:{int(r[args.pos_col])}:{r[args.ref_col]}:{r[args.alt_col]}"

    leads["variant_id"] = leads.apply(vid, axis=1)

    leads["variant_id"].to_csv("condition_list.txt", index=False, header=False)
    bed = leads[[args.chr_col, args.pos_col, args.pos_col, args.gene_col]].copy()
    bed[args.chr_col] = bed[args.chr_col].astype(str).str.replace("chr", "")
    bed.iloc[:, 1] = bed.iloc[:, 1].astype(int) - 1     # 0-based start
    bed.to_csv("pqtl_positions.bed", sep="\t", index=False, header=False)

    cols = [args.gene_col, "variant_id", args.chr_col, args.pos_col,
            args.ref_col, args.alt_col, args.p_col]
    if args.beta_col and args.beta_col in leads.columns:
        cols.append(args.beta_col)
    leads[cols].to_csv("pqtl_lead_summary.tsv", sep="\t", index=False)

    missing = sorted(set(genes) - set(leads[args.gene_col].astype(str)))
    print(leads[cols].to_string(index=False))
    if missing:
        print(f"\n[WARN] no cis-pQTL found for: {missing}")
    print("\nWrote: condition_list.txt, pqtl_positions.bed, pqtl_lead_summary.tsv")


if __name__ == "__main__":
    main()
