"""
Figure 1 — Publication-quality, print-native layout.

Panel a  Hexbin density: 5ULTRA score vs allele frequency (NFE cohort).
         Shows purifying selection acting on high-score 5'UTR variants.

Panel b  Horizontal bar: total unique variant count per mechanism.

Panel c  Ridge-line KDE grid (2 × 3 mechanisms × 6 ancestries).
         Reveals how the AF spectrum differs by mechanism and ancestry.

Output: Figure1_Publication.png (300 dpi)  +  Figure1_Publication.pdf
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

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
MECH_ORDER = [
    'uStart Gain', 'uStart Loss', 'uStop Loss',
    'uStop Gain',  'uKozak Change', 'mKozak Change',
]
ANC_ORDER = ['OTH', 'SAS', 'NFE', 'EAS', 'ASJ', 'AFR']

# Single unified palette — used in both panel b (bars) and panel c (ridges)
MECH_C = {
    'uStart Gain':   '#D62728',   # brick red
    'uStart Loss':   '#1F77B4',   # steel blue
    'uStop Loss':    '#FF7F0E',   # safety orange
    'uStop Gain':    '#2CA02C',   # green
    'uKozak Change': '#9467BD',   # purple
    'mKozak Change': '#8C564B',   # brown
}

# ── 2.  LOAD & PREPARE ────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_DIR / "Figure1_Data_Ancestry_Aggregated.csv")

min_af = df[df['ALT_FREQS'] > 0]['ALT_FREQS'].min()
df['AF_Log'] = np.log10(df['ALT_FREQS'].replace(0, min_af * 0.5))

def get_mech(csq):
    c = str(csq)
    if 'uStart_gain' in c: return 'uStart Gain'
    if 'uStart_loss' in c: return 'uStart Loss'
    if 'uStop_gain'  in c: return 'uStop Gain'
    if 'uStop_loss'  in c: return 'uStop Loss'
    if 'uKozak'      in c: return 'uKozak Change'
    if 'mKozak'      in c: return 'mKozak Change'
    return 'Other'

df['Mechanism'] = df['CSQ'].apply(get_mech)
df = df[df['Mechanism'].isin(MECH_ORDER)].copy()
df_nfe = df[df['Ancestry'] == 'NFE'].copy()
print(f"  Total rows: {len(df):,}  |  NFE: {len(df_nfe):,}")

# ── 2b. PURIFYING SELECTION STATS (NFE, deduplicated) ─────────────────────────
from scipy.stats import mannwhitneyu
nfe_u = df_nfe.drop_duplicates('ID').copy()
nfe_u = nfe_u[nfe_u['ALT_FREQS'] > 0]

high = nfe_u[nfe_u['5ULTRA_Score'] >= 0.5]['ALT_FREQS']
low  = nfe_u[nfe_u['5ULTRA_Score'] <  0.5]['ALT_FREQS']
fold_hl   = low.mean() / high.mean()
_, mw_p   = mannwhitneyu(high, low, alternative='less')
pct_high_rare = (high < 0.001).mean() * 100
pct_low_rare  = (low  < 0.001).mean() * 100

print(f"\n── Purifying selection stats (NFE, unique variants with AF > 0) ──")
print(f"  High-scoring (≥0.5): n={len(high):,}  mean AF={high.mean():.3e}")
print(f"  Low-scoring  (<0.5): n={len(low):,}   mean AF={low.mean():.3e}")
print(f"  Fold difference (low/high): {fold_hl:.2f}×")
print(f"  Mann-Whitney p: {mw_p:.2e}")
print(f"  % rare (AF<0.1%) high-scoring: {pct_high_rare:.1f}%")
print(f"  % rare (AF<0.1%) low-scoring:  {pct_low_rare:.1f}%")

deciles = pd.qcut(nfe_u['5ULTRA_Score'], 10, labels=False)
d_means = nfe_u.groupby(deciles)['ALT_FREQS'].mean()
fold_decile = d_means.iloc[0] / d_means.iloc[-1]
print(f"  Fold difference lowest/highest decile mean AF: {fold_decile:.2f}×")
print(f"  Decile mean AFs (0=lowest score … 9=highest score):")
for i, v in d_means.items():
    print(f"    decile {i}: {v:.3e}")
print()

# ── 3.  FIGURE SKELETON ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.0), dpi=300)

gs_main = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.4, 1.85],
    hspace=0.44,
)
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_main[0],
    width_ratios=[1.0, 1.45], wspace=0.52,
)
gs_bot = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_main[1],
    hspace=0.50, wspace=0.38,
)

# ── 4.  PANEL b: HEXBIN CONSTRAINT MAP ────────────────────────────────────────
ax_a = fig.add_subplot(gs_top[1])

hb = ax_a.hexbin(
    df_nfe['5ULTRA_Score'], df_nfe['AF_Log'],
    gridsize=50, cmap='magma',
    norm=LogNorm(vmin=1, vmax=2000),
    mincnt=1, edgecolors='none',
)

# Colorbar — tight, minimal ticks
cb = fig.colorbar(hb, ax=ax_a, shrink=0.90, pad=0.02, aspect=28,
                  ticks=[1, 10, 100, 1000])
cb.ax.set_yticklabels(['1', '10', '100', '1,000'], fontsize=7)
cb.set_label('Variants (log scale)', fontsize=8, labelpad=0)
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(width=0.5, length=2.5)

# Axis formatting
ax_a.set_xlim(-0.02, 1.02)
ax_a.set_ylim(-6.5, 0.35)
ax_a.set_xlabel('5ULTRA score', fontsize=9, fontweight='bold')
ax_a.set_ylabel(r'Allele frequency', fontsize=9, fontweight='bold')
ax_a.spines[['top', 'right']].set_visible(False)
ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

# Y-axis: use mathtext for proper superscripts
ax_a.set_yticks([-6, -5, -4, -3, -2, -1, 0])
ax_a.set_yticklabels(
    [r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$',
     r'$10^{-2}$', r'$10^{-1}$', r'$1$'],
    fontsize=7.5,
)

ax_a.text(-0.22, 1.04, 'b', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='top')

# ── 5.  PANEL a: MECHANISM ABUNDANCE ─────────────────────────────────────────
ax_b = fig.add_subplot(gs_top[0])

# Unique variants across all ancestries
counts = (
    df.drop_duplicates('ID')['Mechanism']
    .value_counts()
    .reindex(MECH_ORDER)
    .fillna(0)
    .reset_index()
)
counts.columns = ['Mechanism', 'Count']
counts = counts.sort_values('Count', ascending=True)   # ascending → largest at top
bar_colors = [MECH_C[m] for m in counts['Mechanism']]

bars = ax_b.barh(range(len(counts)), counts['Count'],
                 color=bar_colors, edgecolor='none', height=0.65)

# Count labels
max_c = counts['Count'].max()
for bar, cnt in zip(bars, counts['Count']):
    if cnt == 0:
        continue
    lbl = f'{cnt/1000:.0f}k' if cnt >= 1000 else f'{int(cnt):,}'
    ax_b.text(cnt + max_c * 0.025,
              bar.get_y() + bar.get_height() / 2,
              lbl, va='center', ha='left', fontsize=7.5, color='#333333')

ax_b.set_yticks(range(len(counts)))
ax_b.set_yticklabels(counts['Mechanism'], fontsize=8)
ax_b.set_xlim(0, max_c * 1.32)
ax_b.set_xlabel('Number of variants', fontsize=9, fontweight='bold')
ax_b.xaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))
    )
)
ax_b.spines[['top', 'right']].set_visible(False)

ax_b.text(-0.44, 1.04, 'a', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', va='top')

# ── 6.  PANEL c: RIDGELINE GRID ──────────────────────────────────────────────
print("Generating ridgelines …")
X_LO, X_HI = -6.5, 0.2

for m_idx, mech in enumerate(MECH_ORDER):
    row, col = divmod(m_idx, 3)
    color = MECH_C[mech]

    gs_sub = gridspec.GridSpecFromSubplotSpec(
        len(ANC_ORDER), 1,
        subplot_spec=gs_bot[row, col],
        hspace=-0.52,
    )

    # Invisible overlay carries the mechanism title (and panel letter for cell 0)
    ax_ov = fig.add_subplot(gs_bot[row, col], frame_on=False)
    ax_ov.set_xticks([]); ax_ov.set_yticks([])
    ax_ov.set_title(mech, fontsize=8.5, fontweight='bold',
                    color=color, pad=4, loc='left')
    if m_idx == 0:
        ax_ov.text(-0.18, 1.25, 'c', transform=ax_ov.transAxes,
                   fontsize=13, fontweight='bold', va='top')

    for a_idx, anc in enumerate(ANC_ORDER):
        ax = fig.add_subplot(gs_sub[a_idx])

        sub = df[(df['Mechanism'] == mech) &
                 (df['Ancestry']  == anc)]['AF_Log'].dropna()

        if len(sub) > 10:
            xg  = np.linspace(X_LO, X_HI, 500)
            kde = gaussian_kde(sub, bw_method=0.18)(xg)
            y   = np.sqrt(kde)            # sqrt-scale to show rare tail better
            y  /= y.max()
            ax.fill_between(xg, 0, y, color=color, alpha=0.72,
                            edgecolor='#222222', linewidth=0.35, zorder=2)

        # 0.1% MAF reference line
        ax.axvline(-3, color='#BBBBBB', ls=':', lw=0.7, zorder=1)

        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.patch.set_alpha(0)
        ax.spines[['top', 'right', 'left']].set_visible(False)

        ax.text(-0.05, 0.28, anc, fontsize=6.5, fontweight='bold',
                ha='right', va='bottom', color='#444444',
                transform=ax.transAxes)

        is_last = (a_idx == len(ANC_ORDER) - 1)
        if is_last:
            ax.tick_params(axis='x', labelsize=7, colors='#333333', length=2.5)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            if row == 1:
                ax.set_xlabel(r'$\log_{10}$(AF)', fontsize=8, fontweight='bold')
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

# ── 7.  FOOTER NOTE + SAVE ────────────────────────────────────────────────────
fig.text(0.13, 0.04,
         r'- - -  dotted line = 0.1% MAF ($\log_{10}$ = $-$3)',
         fontsize=7, color='#999999', va='bottom')

for fmt in ('png', 'pdf'):
    out = f'Figure1_Publication.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f"Saved: {out}")
