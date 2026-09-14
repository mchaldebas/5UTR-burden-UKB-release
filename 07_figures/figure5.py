"""
Figure 5 — proteomics validation of 5ULTRA burden associations (4 panels).

a  Rank enrichment histogram  — genome-wide, 5ULTRA variants sit in the top 10%
   of each gene's pQTL distribution at ~3× the expected rate.

b  Strip chart by burden tier  — best 5ULTRA cis-pQTL per gene on a log scale.
   The gradient from no-burden → suggestive → GWS mirrors the burden signal
   strength, confirmed by a one-sided Mann-Whitney test.

c  Back-to-back evidence for the 4 GWS genes  — left arm = burden -log10P
   (UKB 500K, blood traits); right arm = best 5ULTRA cis-pQTL -log10P
   (Hawkes et al. proteomics, independent cohort).  Both on the same scale.

d  Forest plot  — individual 5ULTRA variant pQTL effect sizes for GWS genes.
   BETA = SD change in protein level per ALT allele (Hawkes et al.).
   Colour = 5ULTRA predicted translation direction.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker
import glob
from scipy import stats
from scipy.stats import norm

DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))

# ── Style — Nature/Science journal standards ──────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        'Arial',
    'font.size':           11,
    'axes.labelsize':      12,
    'axes.titlesize':      12,
    'xtick.labelsize':     10.5,
    'ytick.labelsize':     10.5,
    'axes.linewidth':      0.6,
    'axes.edgecolor':      '#333333',
    'xtick.color':         '#333333',
    'ytick.color':         '#333333',
    'xtick.major.width':   0.6,
    'ytick.major.width':   0.6,
    'xtick.major.size':    3.0,
    'ytick.major.size':    3.0,
    'legend.frameon':      False,
    'legend.fontsize':     11,
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
    'savefig.facecolor':   'white',
    'pdf.fonttype':        42,   # embed fonts as TrueType (required by journals)
    'ps.fonttype':         42,
})

# ── Colours (aligned with Figure 2 palette) ────────────────────────────────
C_OBS    = '#2E86C1'   # observed / hematology blue
C_GWS    = '#C0392B'   # GWS / 5ULTRA-exclusive red
C_SUG    = '#E67E22'   # suggestive / endocrine orange
C_NONE   = '#85929E'   # no-burden / shared grey
C_BURDEN = '#2C3E50'   # dark navy for burden bars
C_PQTL   = '#2E86C1'   # blue for pQTL bars
C_DEC    = '#2E86C1'   # decreased translation
C_INC    = '#27AE60'   # increased translation
C_NTERM  = '#8E44AD'   # N-terminal extension
STRIPE   = '#F4F6F7'   # alternating row stripe (matches Fig 2)

CSQ_COLORS = {
    'uStart_gain':                       '#2E86C1',
    'uStop_loss longer Non-Overlapping': '#E67E22',
}
TRANS_COLOR = {'decreased': C_DEC, 'increased': C_INC,
               'N-terminal extension': C_NTERM}

# ── Phenotype labels (shared canonical mapping) ───────────────────────────────
from pheno_labels import PHENO_LABELS

# Model-selection-corrected threshold for 5ULTRA suite (8 model×spectrum combinations)
THRESH_U = 6.0 + np.log10(8)   # 6.903

GRP_ORDER  = ['No burden\nsignal', 'Suggestive\n(logP 3-6.9)', 'GWS\nburdenhit']
GRP_LABELS = ['No burden signal', 'Suggestive\n(logP 3-6.9)', 'GWS burden hit']
GRP_COLOR  = {'No burden\nsignal': C_NONE, 'Suggestive\n(logP 3-6.9)': C_SUG,
              'GWS\nburdenhit': C_GWS}
GWS_GENES      = ['GP1BA', 'IGDCC4', 'BCL7A', 'WDR46']
GWS_LABEL_GENES = set(GWS_GENES)

# ── Load shared data ──────────────────────────────────────────────────────────
print("Loading data...")
MODELS_5U = {'5U_Logic', '5ULTRA_Binary', '5ULTRA_Weighted', 'Flux_Joint'}
master    = pd.read_csv(DATA_DIR / 'Master_Results_Clean.csv.gz')
m5u_all   = master[master['model'].isin(MODELS_5U) & (master['true_k'] >= 5)]
gene_burden = (m5u_all.sort_values('logp', ascending=False)
               .drop_duplicates('gene').set_index('gene'))

pqtl = pd.read_csv(DATA_DIR / 'EUR.5UTRs.MANE.GENE.cis.tsv', sep='\t', low_memory=False)
pqtl.columns = [c.replace('#', '') for c in pqtl.columns]
for col in ('LOG10P', 'BETA', 'SE', 'ALTFREQ'):
    pqtl[col] = pd.to_numeric(pqtl[col], errors='coerce')

dfs = []
for f in glob.glob(str(DATA_DIR / '5ultra_output/*.tsv')):
    tmp = pd.read_csv(f, sep='\t', low_memory=False)
    tmp.columns = [c.replace('#', '') for c in tmp.columns]
    dfs.append(tmp[['GENE', 'ID', 'CSQ', 'Translation']])
db_all = pd.concat(dfs, ignore_index=True)
# Deduplicate per (GENE, ID) — the same variant can appear in multiple gene contexts
# with different CSQ/Translation; keep the first annotation for each gene-variant pair.
db = db_all.drop_duplicates(['GENE', 'ID'])
ultra_ids = set(db['ID'].astype(str))   # is_5ultra flag: variant in ANY gene's 5ultra output

pqtl['is_5ultra'] = pqtl['ID'].isin(ultra_ids)
pqtl_rare = pqtl[pqtl['ALTFREQ'] < 0.05].copy()
# Merge on GENE + ID so each variant gets the annotation specific to its gene,
# not an arbitrary annotation from a different gene that shares the same variant ID.
pqtl_rare  = pqtl_rare.merge(db[['GENE', 'ID', 'Translation', 'CSQ']],
                               on=['GENE', 'ID'], how='left')

# ── Panel a — rank enrichment ─────────────────────────────────────────────────
rank_rows = []
for gene, gdf in pqtl_rare.groupby('GENE'):
    u = gdf[gdf['is_5ultra']]
    if len(u) == 0:
        continue
    best = u['LOG10P'].max()
    rp   = (gdf['LOG10P'] <= best).sum() / len(gdf) * 100
    burden_lp = float(gene_burden.loc[gene, 'logp']) if gene in gene_burden.index else 0.0
    rank_rows.append({'gene': gene, 'rank_pct': rp, 'burden_logp': burden_lp})
rank_df    = pd.DataFrame(rank_rows)
rank_pcts  = rank_df['rank_pct'].values
n_genes    = len(rank_pcts)
_, wpval   = stats.wilcoxon(rank_pcts - 50)
hist_edges = np.linspace(0, 100, 11)
hist_counts, _ = np.histogram(rank_pcts, bins=hist_edges)
hist_expected  = n_genes / 10.0
print(f"a  n={n_genes} genes  Wilcoxon p={wpval:.2e}")

# ── Panel b — strip chart (best 5ULTRA per gene) ──────────────────────────────
strip_rows = []
for gene, gdf in pqtl_rare.groupby('GENE'):
    u = gdf[gdf['is_5ultra']]
    if len(u) == 0:
        continue
    best_row  = u.loc[u['LOG10P'].idxmax()]
    csq_hit   = db.loc[(db['GENE'] == gene) & (db['ID'] == str(best_row['ID'])), 'CSQ']
    csq       = csq_hit.iloc[0] if len(csq_hit) else ''
    burden_lp = float(gene_burden.loc[gene, 'logp']) if gene in gene_burden.index else 0.0
    strip_rows.append({'gene': gene, 'pqtl_logp': float(best_row['LOG10P']),
                       'csq': csq, 'burden_logp': burden_lp})
strip_df = pd.DataFrame(strip_rows)
LOG_FLOOR = 0.05
strip_df['pqtl_plot'] = np.maximum(strip_df['pqtl_logp'], LOG_FLOOR)
strip_df['group'] = pd.cut(strip_df['burden_logp'],
                            bins=[-0.01, 2.99, THRESH_U - 0.001, 1e9],
                            labels=GRP_ORDER)
gws_vals  = strip_df[strip_df['group'] == 'GWS\nburdenhit']['pqtl_plot']
none_vals = strip_df[strip_df['group'] == 'No burden\nsignal']['pqtl_plot']
_, mw_p   = stats.mannwhitneyu(gws_vals, none_vals, alternative='greater')
print(f"b  GWS median={gws_vals.median():.2f}  none median={none_vals.median():.2f}"
      f"  Mann-Whitney p={mw_p:.3e}")

# ── Panel c — per-gene GWS summary ───────────────────────────────────────────
gene_data = {}
for gene in GWS_GENES:
    best_b = m5u_all[m5u_all['gene'] == gene].sort_values('logp', ascending=False).iloc[0]
    gdf    = pqtl_rare[pqtl_rare['GENE'] == gene]
    u5     = gdf[gdf['is_5ultra']].dropna(subset=['LOG10P'])
    best_p = u5.loc[u5['LOG10P'].idxmax()] if len(u5) else None
    gene_data[gene] = {
        'burden_logp':  float(best_b['logp']),
        'burden_pheno': int(best_b['pheno']),
        'pqtl_logp':    float(best_p['LOG10P']) if best_p is not None else 0.0,
        'n_5ultra':     len(u5),
    }
    print(f"c  {gene}: burden={gene_data[gene]['burden_logp']:.1f} "
          f"({PHENO_LABELS.get(gene_data[gene]['burden_pheno'], gene_data[gene]['burden_pheno'])}), "
          f"pQTL={gene_data[gene]['pqtl_logp']:.2f}")

# ── Panel d — forest plot ─────────────────────────────────────────────────────
forest_rows = []
for gene in GWS_GENES:
    gdf = pqtl_rare[pqtl_rare['GENE'] == gene]
    u5  = gdf[gdf['is_5ultra']].dropna(subset=['BETA', 'SE'])
    for _, row in u5[u5['LOG10P'] >= 1.3].sort_values('LOG10P', ascending=False).iterrows():
        parts = str(row['ID']).split(':')
        id_short = (f"{parts[0]}:{parts[1]}  {parts[2]}>{parts[3]}"
                    if len(parts) == 4 else str(row['ID']))
        forest_rows.append({'gene': gene,
                            'id_short':    id_short,
                            'beta':        float(row['BETA']),
                            'ci95':        1.96 * float(row['SE']),
                            'logp':        float(row['LOG10P']),
                            'translation': str(row['Translation'])})
forest_df = pd.DataFrame(forest_rows)

# Build y-positions: gene header sits 0.52 above its block's top variant
yi = 0.0
plot_rows = []
for gene in GWS_GENES:
    gene_variants = forest_df[forest_df['gene'] == gene]
    plot_rows.append({'kind': 'header', 'gene': gene, 'y': yi + 0.52})
    for _, r in gene_variants.iterrows():
        plot_rows.append({'kind': 'variant', 'y': float(yi), **r.to_dict()})
        yi -= 1.0
    yi -= 0.55   # inter-gene gap
plot_rows_df = pd.DataFrame(plot_rows)
variant_rows = plot_rows_df[plot_rows_df['kind'] == 'variant'].copy()
header_rows  = plot_rows_df[plot_rows_df['kind'] == 'header'].copy()

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12), dpi=300)
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    width_ratios=[1.0, 1.2],
    height_ratios=[1.0, 1.6],
    hspace=0.48, wspace=0.38,
    left=0.07, right=0.96, top=0.95, bottom=0.07,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL a — rank enrichment histogram
# ══════════════════════════════════════════════════════════════════════════════
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
bar_colors  = [C_OBS if c >= 90 else '#AED6F1' for c in bin_centers]
ax_a.bar(bin_centers, hist_counts, width=8.5,
         color=bar_colors, zorder=2, edgecolor='white', linewidth=0.5)
ax_a.axhline(hist_expected, color=C_GWS, lw=1.2, ls='--', zorder=3)

# Stat annotation — no box, clean positioned text
ratio = hist_counts[-1] / hist_expected
ax_a.text(3, hist_counts.max() * 0.85,
          f'Top 10% bin: {ratio:.1f}× expected\n'
          f'Wilcoxon p = {wpval:.1e}   n = {n_genes} genes',
          ha='left', va='top', fontsize=13, color='#333', linespacing=1.5)

ax_a.set_xlim(0, 100)
ax_a.set_ylim(0, hist_counts.max() * 1.18)
ax_a.set_xticks([0, 25, 50, 75, 100])
ax_a.set_xticklabels(['Bottom\n10%', '25%', '50%', '75%', 'Top\n10%'], fontsize=14)
ax_a.set_xlabel("Rank percentile of best 5ULTRA variant\nwithin the gene's full pQTL distribution",
                fontsize=15, fontweight='bold')
ax_a.set_ylabel('Number of genes', fontsize=15, fontweight='bold')
ax_a.spines[['top', 'right']].set_visible(False)
ax_a.tick_params(axis='both', labelsize=14)
leg_a = [
    mpatches.Patch(facecolor=C_OBS,    edgecolor='none', label='Top 10% (observed)'),
    mpatches.Patch(facecolor='#AED6F1', edgecolor='none', label='Other deciles'),
    mlines.Line2D([0], [0], color=C_GWS, lw=1.2, ls='--',
                  label=f'Expected ({hist_expected:.0f} per bin)'),
]
ax_a.legend(handles=leg_a, frameon=False, fontsize=13,
            loc='upper left', labelspacing=0.3, handlelength=1.4, handletextpad=0.5)
ax_a.text(-0.15, 1.04, 'a', transform=ax_a.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL b — strip chart by burden tier
# ══════════════════════════════════════════════════════════════════════════════
jit_b = {}
for grp, sub in strip_df.groupby('group', observed=True):
    jit_b[str(grp)] = dict(zip(sub['gene'], rng.uniform(-0.18, 0.18, len(sub))))

gws_dot_positions = {}
for grp, sub in strip_df.groupby('group', observed=True):
    xi     = GRP_ORDER.index(str(grp))
    c      = GRP_COLOR[str(grp)]
    is_gws = str(grp) == 'GWS\nburdenhit'
    jmap   = jit_b[str(grp)]
    for _, row in sub.iterrows():
        jv    = jmap.get(row['gene'], 0)
        dot_c = CSQ_COLORS.get(row['csq'], c) if is_gws else c
        ax_b.scatter(xi + jv, row['pqtl_plot'],
                     color=dot_c,
                     s=55 if is_gws else 9,
                     alpha=1.0 if is_gws else 0.35,
                     edgecolors='white' if is_gws else 'none',
                     linewidths=0.6,
                     zorder=5 if is_gws else 3,
                     marker='D' if is_gws else 'o')
        if is_gws and row['gene'] in GWS_LABEL_GENES:
            prev_y, _ = gws_dot_positions.get(row['gene'], (-np.inf, None))
            if row['pqtl_plot'] >= prev_y:
                gws_dot_positions[row['gene']] = (row['pqtl_plot'],
                                                   (xi + jv, row['pqtl_plot'], dot_c))

for grp, sub in strip_df.groupby('group', observed=True):
    xi  = GRP_ORDER.index(str(grp))
    med = sub['pqtl_plot'].median()
    ax_b.plot([xi - 0.25, xi + 0.25], [med, med],
              color='#1a1a1a', lw=2.0, zorder=6, solid_capstyle='round')

label_nudge = {'GP1BA': (12, 'left'), 'IGDCC4': (12, 'left'),
               'BCL7A': (-12, 'right'), 'WDR46': (-12, 'right')}
for gene, (_, (xp, yp, dc)) in gws_dot_positions.items():
    xoff, ha = label_nudge.get(gene, (12, 'left'))
    ax_b.annotate(gene, xy=(xp, yp), xytext=(xoff, 0),
                  textcoords='offset points',
                  fontsize=13, fontweight='bold', color=dc, ha=ha, va='center',
                  arrowprops=dict(arrowstyle='-', color=dc, lw=0.5,
                                  shrinkA=0, shrinkB=3))

ax_b.set_yscale('log')
ax_b.set_ylim(LOG_FLOOR * 0.7, strip_df['pqtl_logp'].max() * 3.5)
ax_b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda y, _: f'{y:.0f}' if y >= 1 else f'{y:.2f}'))
ax_b.set_yticks([0.05, 0.1, 0.3, 1, 3, 10, 100])
ax_b.tick_params(axis='y', labelsize=14)

n_by_grp   = strip_df.groupby('group', observed=True).size()
xtick_lbls = [f'{GRP_LABELS[i]}\n(n = {n_by_grp.get(g, 0)})'
              for i, g in enumerate(GRP_ORDER)]
ax_b.set_xticks([0, 1, 2])
ax_b.set_xticklabels(xtick_lbls, fontsize=14)
ax_b.set_xlim(-0.55, 2.55)
for tick, grp in zip(ax_b.get_xticklabels(), GRP_ORDER):
    tick.set_color(GRP_COLOR[grp])
    if grp == 'GWS\nburdenhit':
        tick.set_fontweight('bold')
ax_b.set_ylabel('Best 5ULTRA cis-pQTL\n-log10P  (MAF < 5%)',
                fontsize=15, fontweight='bold')
# Stat annotation — no box
ax_b.text(0.97, 0.04,
          f'Mann–Whitney p = {mw_p:.2e}',
          ha='right', va='bottom', fontsize=13, color='#555',
          transform=ax_b.transAxes, style='italic')
ax_b.spines[['top', 'right']].set_visible(False)
ax_b.text(-0.13, 1.04, 'b', transform=ax_b.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL c — back-to-back burden vs pQTL (GWS genes)
# ══════════════════════════════════════════════════════════════════════════════
ys    = np.arange(len(GWS_GENES) - 1, -1, -1)
barh  = 0.44
MAX_LP = 75

for i, gene in enumerate(GWS_GENES):
    if i % 2 == 1:
        ax_c.axhspan(ys[i] - 0.5, ys[i] + 0.5, color=STRIPE, zorder=0)

for i, gene in enumerate(GWS_GENES):
    d    = gene_data[gene]
    y    = ys[i]
    b_lp = min(d['burden_logp'], MAX_LP)
    p_lp = d['pqtl_logp']
    trait = PHENO_LABELS.get(d['burden_pheno'], f"UKB {d['burden_pheno']}")

    ax_c.barh(y, -b_lp, height=barh, color=C_BURDEN, alpha=0.9,  zorder=2)
    ax_c.barh(y,  p_lp, height=barh, color=C_PQTL,   alpha=0.9,  zorder=2)

    ax_c.text(-b_lp - 0.8, y, f'{d["burden_logp"]:.0f}',
              ha='right', va='center', fontsize=14, fontweight='bold', color=C_BURDEN)
    ax_c.text(-20, y - barh * 0.62, trait,
              ha='center', va='top', fontsize=13.5, color='#666', style='italic', zorder=4)

    sig = '***' if p_lp > 10 else ('**' if p_lp > 5 else ('*' if p_lp > 1.3 else 'ns'))
    ax_c.text(p_lp + 0.8, y, f'{p_lp:.1f}  {sig}',
              ha='left', va='center', fontsize=14, fontweight='bold', color=C_PQTL)

ax_c.axvline( THRESH_U, color=C_GWS, lw=0.8, ls='--', alpha=0.65, zorder=3)
ax_c.axvline(-THRESH_U, color=C_GWS, lw=0.8, ls='--', alpha=0.65, zorder=3)
ax_c.axvline( 0, color='#2C2C2C', lw=0.8, zorder=3)
ax_c.text(-37, 3.42, f'GWS threshold\n(-log10P > {THRESH_U:.1f})',
          ha='left', va='bottom', fontsize=14, color=C_GWS, style='italic')

ax_c.set_yticks(ys)
ax_c.set_yticklabels(GWS_GENES, fontsize=16, fontweight='bold')
ax_c.tick_params(axis='y', length=0, pad=4)
ax_c.set_ylim(-0.72, len(GWS_GENES) - 0.28)
xticks = [-60, -40, -20, -round(THRESH_U, 1), 0, round(THRESH_U, 1), 20]
ax_c.set_xticks(xticks)
ax_c.set_xticklabels([str(abs(x)) for x in xticks], fontsize=14)
ax_c.set_xlim(-MAX_LP - 10, 28)
ax_c.set_xlabel('-log10P', fontsize=16, fontweight='bold')

mid_burden = (-MAX_LP - 10) / 2
ax_c.text(mid_burden, len(GWS_GENES) - 0.08,
          '5ULTRA burden association',
          ha='center', va='bottom', fontsize=15, fontweight='bold', color=C_BURDEN)
ax_c.text(14, len(GWS_GENES) - 0.08, 'cis-pQTL',
          ha='center', va='bottom', fontsize=15, fontweight='bold', color=C_PQTL)
# Significance key — clean text, no box
ax_c.text(0.56, 0.015,
          '* -log10P > 1.3 '
          '** > 5 '
          '*** > 10',
          ha='right', va='bottom', fontsize=12.5, color='#777',
          transform=ax_c.transAxes)
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.text(-0.19, 1.04, 'c', transform=ax_c.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ══════════════════════════════════════════════════════════════════════════════
# PANEL d — forest plot
# ══════════════════════════════════════════════════════════════════════════════
from matplotlib.transforms import blended_transform_factory
ltrans = blended_transform_factory(ax_d.transAxes, ax_d.transData)

XMIN_D, XMAX_D = -0.72, 0.60
ax_d.set_xlim(XMIN_D, XMAX_D)
ax_d.axvline(0, color='#BBBBBB', lw=0.8, ls='--', zorder=1)

# ── Gene name labels (bold, black, right-aligned into left margin) ────────────
GENE_X = -0.05   # axes fraction
for _, hr in header_rows.iterrows():
    ax_d.text(GENE_X, hr['y'],
              hr['gene'],
              ha='right', va='center',
              fontsize=15, fontweight='bold', color='#1a1a1a',
              transform=ltrans, clip_on=False)

# ── Variant ID labels (two lines: position / alleles) ─────────────────────────
for _, r in variant_rows.iterrows():
    parts = r['id_short'].split('  ')
    pos_str    = parts[0] if len(parts) >= 1 else r['id_short']
    allele_str = parts[1] if len(parts) == 2 else ''
    ax_d.text(GENE_X, r['y'] + 0.14, pos_str,
              ha='right', va='center',
              fontsize=12, fontfamily='monospace', color='#444',
              transform=ltrans, clip_on=False)
    if allele_str:
        ax_d.text(GENE_X, r['y'] - 0.18, allele_str,
                  ha='right', va='center',
                  fontsize=11.5, fontfamily='monospace', color='#999',
                  transform=ltrans, clip_on=False)

# ── CI bars and point estimates ───────────────────────────────────────────────
present_trans = set()
for _, row in variant_rows.iterrows():
    c    = TRANS_COLOR.get(str(row['translation']), '#999999')
    present_trans.add(str(row['translation']))
    beta = float(row['beta'])
    ci   = float(row['ci95'])
    y    = float(row['y'])

    ax_d.plot([beta - ci, beta + ci], [y, y],
              color=c, lw=1.6, solid_capstyle='round', alpha=0.7, zorder=3)
    for xc in [beta - ci, beta + ci]:
        ax_d.plot([xc, xc], [y - 0.10, y + 0.10],
                  color=c, lw=1.2, zorder=3)
    ax_d.scatter(beta, y, s=44, color=c,
                 edgecolors='white', linewidths=0.7, zorder=5)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax_d.set_yticks([])
y_all = plot_rows_df['y'].values
ax_d.set_ylim(y_all.min() - 0.65, y_all.max() + 0.80)
ax_d.tick_params(axis='x', labelsize=13)
ax_d.set_xlabel('cis-pQTL effect size  (SD per ALT allele)',
                fontsize=15, fontweight='bold')
ax_d.spines[['top', 'right', 'left']].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
TRANS_LABEL = {'decreased':            'Decreased translation',
               'increased':            'Increased translation',
               'N-terminal extension': 'N-terminal extension'}
leg_d = [mpatches.Patch(color=TRANS_COLOR[t], label=TRANS_LABEL[t])
         for t in ['decreased', 'increased', 'N-terminal extension']
         if t in present_trans]
ax_d.legend(handles=leg_d, frameon=False, fontsize=13,
            loc='lower right',
            labelspacing=0.35, handlelength=1.0, handletextpad=0.5,
            title='5ULTRA prediction', title_fontsize=14.5)
ax_d.text(-0.08, 1.04, 'd', transform=ax_d.transAxes,
          fontsize=18, fontweight='bold', va='top')

# ── Save ──────────────────────────────────────────────────────────────────────
for fmt in ('png', 'pdf'):
    out = f'Figure5_combined.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f'Saved: {out}')
