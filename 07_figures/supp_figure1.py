"""
Supplementary Figure 1 — Miami plot + QQ plots (publication quality).

Panel a  Miami plot: 5ULTRA (top, up) vs CADD (bottom, down).
         Separate GWS thresholds per arm, Bonferroni over the number of
         association tests performed in each suite (see analysis_config.py):
           THRESH_U = 0.05 / 518,380 -> -log10P > 7.016
           THRESH_C = 0.05 / 107,033 -> -log10P > 6.331

Panel b  QQ plot — 5ULTRA, single model (High-confidence, score >= 0.5, af5)
Panel c  QQ plot — CADD,   single model (High-confidence, CADD >= 5, af5)

         λGC is computed on ONE fixed model rather than the best-of-N per
         gene-pheno pair. Taking the max over 4 (or 2) models × 2 spectra is a
         selection on the most significant result, which inflates λGC by
         construction and makes it uninterpretable as a calibration metric.
         The High-confidence masks (score/CADD-threshold-passing variants) are the
         best-calibrated single models — λGC ≈ 1.01 (5ULTRA) / 1.05 (CADD) on
         >120k / >60k tests, versus the inflated Weighted masks — and the af5
         spectrum gives one independent association test per gene-pheno pair.

Output: SuppFig1_Miami_QQ.png / .pdf
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import scipy.stats as stats
from adjustText import adjust_text

DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))

# ── 0.  GLOBAL STYLE ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'Arial',
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.titlesize':     9,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'axes.linewidth':     0.7,
    'axes.edgecolor':     '#2C2C2C',
    'xtick.color':        '#2C2C2C',
    'ytick.color':        '#2C2C2C',
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'xtick.major.size':   3.0,
    'ytick.major.size':   3.0,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
})

# ── 1.  CONSTANTS ─────────────────────────────────────────────────────────────
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis_config import (MASTER_CSV, TRUE_K_MIN, THRESH_U, THRESH_C,
                             MODELS_5ULTRA, MODELS_CADD, describe)

BED_FILE      = DATA_DIR / 'mane_5utrs_with_names.bed'

# QQ panels are drawn against an uncorrected reference line, not the
# suite thresholds (see the draw_qq calls below).
THRESHOLD     = 6.0
print(describe())

# Colour palette consistent with main figures
C_5U_DARK  = '#2E86C1'   # significant 5ULTRA
C_5U_A     = '#2E86C1'   # chr alternating odd
C_5U_B     = '#AED6F1'   # chr alternating even
C_CA_DARK  = '#7B7D7D'   # significant CADD
C_CA_A     = '#7B7D7D'   # chr alternating odd
C_CA_B     = '#CCD1D1'   # chr alternating even
C_THRESH   = '#E74C3C'

from pheno_labels import PHENO_LABELS

# ── 2.  LOAD GENE COORDINATES ─────────────────────────────────────────────────
print("Loading gene coordinates …")
bed = pd.read_csv(BED_FILE, sep='\t', header=None,
                  names=['chrom', 'start', 'end', 'gene'])
bed['chrom'] = bed['chrom'].str.replace('chr', '', regex=False)
bed = bed[bed['chrom'].isin([str(i) for i in range(1, 23)] + ['X'])].copy()
bed['chrom_num'] = bed['chrom'].replace({'X': 23}).astype(int)
bed['pos'] = bed['start']

# One representative position per gene (first 5'UTR start)
gene_coords = (bed.sort_values(['chrom_num', 'pos'])
                  .groupby('gene')[['chrom_num', 'pos']]
                  .first()
                  .reset_index())

# Cumulative genomic X-axis
chrom_max = gene_coords.groupby('chrom_num')['pos'].max()
chrom_offsets = chrom_max.cumsum() - chrom_max
gene_coords['pos_cum'] = gene_coords['pos'] + gene_coords['chrom_num'].map(chrom_offsets)

x_ticks, x_labels = [], []
for chrom, grp in gene_coords.groupby('chrom_num'):
    x_ticks.append(grp['pos_cum'].median())
    x_labels.append(str(chrom) if chrom != 23 else 'X')

print(f"  Gene coords: {len(gene_coords):,} unique genes")

# ── 3.  LOAD RESULTS & TAKE BEST PER GENE-PHENO ───────────────────────────────
print("Loading results …")
df = pd.read_csv(MASTER_CSV)
df = df[df['true_k'] >= TRUE_K_MIN].copy()

df_5u   = df[df['model'].isin(MODELS_5ULTRA)]
df_cadd = df[df['model'].isin(MODELS_CADD)]

best_5u   = df_5u.loc[df_5u.groupby(['gene','pheno'])['logp'].idxmax()].copy()
best_cadd = df_cadd.loc[df_cadd.groupby(['gene','pheno'])['logp'].idxmax()].copy()

plot_5u   = pd.merge(best_5u,   gene_coords, on='gene', how='inner').dropna(subset=['pos_cum','logp'])
plot_cadd = pd.merge(best_cadd, gene_coords, on='gene', how='inner').dropna(subset=['pos_cum','logp'])

plot_5u['pheno_label']   = plot_5u['pheno'].map(PHENO_LABELS).fillna('')
plot_cadd['pheno_label'] = plot_cadd['pheno'].map(PHENO_LABELS).fillna('')

print(f"  5ULTRA gene-pheno pairs: {len(plot_5u):,}")
print(f"  CADD   gene-pheno pairs: {len(plot_cadd):,}")

# ── 3b.  SINGLE-MODEL SELECTION FOR QQ / λGC ──────────────────────────────────
# λGC must be computed on ONE fixed model, NOT best-of-N. Selecting the max logp
# over multiple models/spectra inflates λGC by construction (selection on the
# most significant result) and makes it uninterpretable. We use the single
# best-calibrated model (High-confidence mask = score/CADD-threshold-passing variants;
# λGC ≈ 1.01 / 1.05) in the af5 spectrum, giving one independent association
# test per gene-pheno pair.
QQ_MODEL_5ULTRA = '5ULTRA_Binary'
QQ_MODEL_CADD   = 'CADD_Binary'
QQ_SPECTRUM     = 'af5'

qq_5u   = df[(df['model'] == QQ_MODEL_5ULTRA) & (df['spectrum'] == QQ_SPECTRUM)].dropna(subset=['logp'])
qq_cadd = df[(df['model'] == QQ_MODEL_CADD)   & (df['spectrum'] == QQ_SPECTRUM)].dropna(subset=['logp'])
print(f"  QQ 5ULTRA tests ({QQ_MODEL_5ULTRA}, {QQ_SPECTRUM}): {len(qq_5u):,}")
print(f"  QQ CADD   tests ({QQ_MODEL_CADD}, {QQ_SPECTRUM}): {len(qq_cadd):,}")

# ── 4.  QQ HELPER ─────────────────────────────────────────────────────────────
def compute_qq(logp_arr):
    logp_arr = np.sort(np.asarray(logp_arr, dtype=float))[::-1]
    n = len(logp_arr)
    exp_p  = np.arange(1, n + 1) / (n + 1)
    exp_lp = -np.log10(exp_p)
    chi2   = stats.chi2.isf(np.clip(10**-logp_arr, 1e-300, 1.0), df=1)
    lam_gc = float(np.median(chi2) / 0.4549364)
    return exp_lp, logp_arr, lam_gc

exp_5u, obs_5u, lam_5u = compute_qq(qq_5u['logp'].values)
exp_ca, obs_ca, lam_ca = compute_qq(qq_cadd['logp'].values)
print(f"  λGC 5ULTRA = {lam_5u:.3f}   λGC CADD = {lam_ca:.3f}")

# ── 5.  FIGURE SKELETON ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.5), dpi=300)

gs_main = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.55, 1.0],
    hspace=0.40,
    left=0.10, right=0.97, top=0.97, bottom=0.06,
)

# Panel a: Miami (two sub-rows sharing x-axis)
gs_miami = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_main[0],
    height_ratios=[1, 1], hspace=0.0,
)
ax_top = fig.add_subplot(gs_miami[0])
ax_bot = fig.add_subplot(gs_miami[1], sharex=ax_top)

# Panels b & c: QQ plots side by side
gs_qq = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_main[1],
    wspace=0.42,
)
ax_b = fig.add_subplot(gs_qq[0])
ax_c = fig.add_subplot(gs_qq[1])

# ── 6.  MIAMI PLOT ────────────────────────────────────────────────────────────
max_y_5u = plot_5u['logp'].max()
max_y_ca = plot_cadd['logp'].max()
ylim_top = max_y_5u * 1.15
ylim_bot = max_y_ca * 1.15

x_min = gene_coords['pos_cum'].min() - 1e7
x_max = gene_coords['pos_cum'].max() + 1e7

for ax, df_plot, thresh, ylim, c_odd, c_even, c_sig, is_bottom in [
    (ax_top, plot_5u,   THRESH_U, ylim_top, C_5U_A,  C_5U_B,  C_5U_DARK, False),
    (ax_bot, plot_cadd, THRESH_C, ylim_bot, C_CA_A,  C_CA_B,  C_CA_DARK, True),
]:
    for chrom, grp in df_plot.groupby('chrom_num'):
        c_bg = c_odd if chrom % 2 == 1 else c_even
        sub = grp[grp['logp'] < thresh]
        if not sub.empty:
            ax.scatter(sub['pos_cum'], sub['logp'],
                       color=c_bg, s=4, alpha=0.55, edgecolors='none', rasterized=True)
        sig = grp[grp['logp'] >= thresh]
        if not sig.empty:
            ax.scatter(sig['pos_cum'], sig['logp'],
                       color=c_sig, s=16, alpha=0.92, edgecolors='none', zorder=5,
                       rasterized=True)

    ax.axhline(thresh, color=C_THRESH, ls='--', lw=0.9, alpha=0.75, zorder=3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, ylim)
    ax.spines[['top', 'right']].set_visible(False)

    # Gene labels for top hits
    top_hits = (df_plot[df_plot['logp'] >= thresh]
                .sort_values('logp', ascending=False)
                .drop_duplicates('gene')
                .head(14))
    texts = []
    for _, row in top_hits.iterrows():
        lbl = f"{row['gene']}\n({row['logp']:.1f})"
        t = ax.text(row['pos_cum'], row['logp'], lbl,
                    fontsize=6.0, fontweight='bold', color='#1A252F',
                    ha='center', va='bottom' if not is_bottom else 'top',
                    zorder=8)
        texts.append(t)
    if texts:
        adjust_text(
            texts, ax=ax,
            expand=(1.2, 1.4),
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.5),
        )

    if is_bottom:
        ax.invert_yaxis()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=6.5)
        ax.set_xlabel('Chromosome', fontsize=9, fontweight='bold')
        ax.set_ylabel(r'CADD ($-\log_{10}P$)', fontsize=8.5, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', which='both', top=False)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.set_ylabel(r'5ULTRA ($-\log_{10}P$)', fontsize=8.5, fontweight='bold')
        ax.spines['bottom'].set_visible(False)

ax_top.text(-0.09, 1.04, 'a', transform=ax_top.transAxes,
            fontsize=13, fontweight='bold', va='top')

# ── 7.  QQ PLOTS ─────────────────────────────────────────────────────────────
def draw_qq(ax, exp_lp, obs_lp, lam, thresh, c_sig, label, panel_letter, ymax):
    mask_sig = obs_lp >= thresh
    # Sub-threshold: downsample to keep file size manageable
    rng = np.random.default_rng(42)
    idx_sub = np.where(~mask_sig)[0]
    keep = rng.choice(idx_sub, size=min(len(idx_sub), 80_000), replace=False)
    idx_keep = np.sort(np.concatenate([keep, np.where(mask_sig)[0]]))

    ax.scatter(exp_lp[idx_keep][~mask_sig[idx_keep]],
               obs_lp[idx_keep][~mask_sig[idx_keep]],
               color='#95A5A6', s=3, alpha=0.5, edgecolors='none',
               rasterized=True, zorder=2)
    ax.scatter(exp_lp[mask_sig], obs_lp[mask_sig],
               color=c_sig, s=12, alpha=0.90, edgecolors='none', zorder=4)

    lim = max(exp_lp.max(), obs_lp.max()) * 1.08
    ax.plot([0, lim], [0, lim], color='#2C2C2C', ls='--', lw=0.9, alpha=0.6)

    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, ymax)

    ax.text(0.05, 0.97,
            f'$\\lambda_{{\\mathrm{{GC}}}}$ = {lam:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', linewidth=0.6))

    ax.set_xlabel(r'Expected $(-\log_{10}P)$', fontsize=9, fontweight='bold')
    ax.set_ylabel(r'Observed $(-\log_{10}P)$', fontsize=9, fontweight='bold')
    ax.set_title(label, fontsize=8.5, loc='left', fontweight='bold', pad=4)
    ax.spines[['top', 'right']].set_visible(False)

    ax.text(-0.18, 1.10, panel_letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

# Single-model QQ: no model-selection correction, so highlight at the base GWS
# threshold (THRESHOLD) rather than the best-of-N-corrected THRESH_U/THRESH_C.
# Shared y-axis across both QQ panels for direct visual comparison
qq_ymax = max(obs_5u.max(), obs_ca.max()) * 1.10
draw_qq(ax_b, exp_5u, obs_5u, lam_5u, THRESHOLD, C_5U_DARK,
        '5ULTRA High-confidence (af5)', 'b', qq_ymax)
draw_qq(ax_c, exp_ca, obs_ca, lam_ca, THRESHOLD, C_CA_DARK,
        'CADD High-confidence (af5)', 'c', qq_ymax)

# ── 8.  SAVE ─────────────────────────────────────────────────────────────────
for fmt in ('png', 'pdf'):
    out = f'SuppFig1_Miami_QQ.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"Saved: {out}")
