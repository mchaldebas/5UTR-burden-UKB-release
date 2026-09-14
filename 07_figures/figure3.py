import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis_config import DATA_DIR, MASTER_CSV, TRUE_K_MIN

matplotlib.rcParams.update({
    'font.family':       'Arial',
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

FILE   = MASTER_CSV
TRUE_K = TRUE_K_MIN
LP_FOC = 3.0

# Per-pair DN-vs-UP beta-equality test (06_rheostat_beta_equality.py output).
# This file drives the four-way classification of the highlighted pairs
# (rheostat / UP-driven / DN-driven / fragility, plus 'neither'); see
# rheostat_classify.py for the definitions. Highlighted pairs are classified on
# the JOINT refit coefficients, not on sign alone. Absent => pairs are marked
# 'untested' and only the population-level sign/binomial story is shown. Point
# RHEO_EQ_FILE at the cluster output to enable.
EQ_FILE  = Path(os.environ.get('RHEO_EQ_FILE',
                               DATA_DIR / "rheostat_beta_equality_nfe.tsv"))
EQ_ALPHA = 0.05

C_DN   = '#2E86C1'
C_UP   = '#C0392B'
C_TREND = '#16A085'
C_FRAG = '#7F8C8D'
C_BG   = '#E8ECF0'

from rheostat_classify import (classify, count_table, CATEGORY_ORDER,
                               CATEGORY_COLORS, CATEGORY_MARKERS,
                               CATEGORY_LABELS)

from pheno_labels import PHENO_LABELS as _PHENO_CANON

# Tighter abbreviations for the dense Z-score rows; every other field (incl.
# 30820 Rheumatoid factor, WBC, Glucose, Vitamin D) is inherited from the
# canonical mapping below so labels never silently fall back to numeric ids.
_PHENO_SHORT = {
    30010:'RBC count',        30020:'Haemoglobin',       30030:'Haematocrit',
    30040:'MCV',              30050:'MCH',               30060:'MCHC',
    30070:'RBC distrib w.',   30080:'Platelet count',    30090:'Platelet crit',
    30100:'Mean plt vol',     30110:'Plt distrib w.',    30120:'Lymphocyte count',
    30130:'Monocyte count',   30140:'Neutrophil count',  30150:'Eosinophil count',
    30160:'Basophil count',   30180:'Lymphocyte %',      30190:'Monocyte %',
    30200:'Neutrophil %',     30210:'Eosinophil %',      30220:'Basophil %',
    30240:'Reticulocyte %',   30250:'Reticulocyte ct',   30260:'Mean retic vol',
    30270:'Mean sph cell vol',30280:'Immature retic fr.',30290:'High lt scat retic %',
    30300:'High lt scat retic ct', 30600:'Albumin',      30610:'Alk phosphatase',
    30620:'ALT',              30630:'Apolipoprotein A',  30640:'Apolipoprotein B',
    30650:'AST',              30660:'Direct bilirubin',  30670:'Urea',
    30680:'Calcium',          30690:'Cholesterol',       30700:'Creatinine',
    30710:'CRP',              30720:'Cystatin C',        30730:'GGT',
    30750:'HbA1c',            30760:'HDL cholesterol',   30770:'IGF-1',
    30780:'LDL direct',       30790:'Lipoprotein A',     30800:'Oestradiol',
    30810:'Phosphate',        30830:'SHBG',
    30840:'Total bilirubin',  30850:'Testosterone',      30860:'Total protein',
    30870:'Triglycerides',    30880:'Urate',
}
PHENO_LABELS = {**_PHENO_CANON, **_PHENO_SHORT}

print("Loading data…")
df = pd.read_csv(FILE)
df = df[(df['true_k'] >= TRUE_K) & (df['spectrum'] == 'af5')].copy()

# NAMING: the results/code label this model '5U_Logic'; it is built from the
# 'Mirror' UP/DN anno (02_annotation/01_generate_anno_masks.py) and is referred
# to as the "Directional model" in the manuscript. Same model, three names —
# reader-facing text/labels use "Directional".
logic = df[df['model'] == '5U_Logic'].copy()   # == "Directional" (Mirror UP/DN)
dn    = logic[logic['mask'].str.contains('DN', case=False, na=False)].copy()
up    = logic[logic['mask'].str.contains('UP', case=False, na=False)].copy()

battle = pd.merge(dn, up, on=['gene','pheno'], suffixes=('_DN','_UP'))
battle['Max_LogP']    = battle[['logp_DN','logp_UP']].max(axis=1)
# Sign-only descriptor, genome-wide. This is NOT the rheostat classification --
# it carries no significance gate and exists solely for the population-level
# binomial / trend statistics in Panels A and C, where per-pair refits are not
# available. Pair-level categories come from classify() on the refit output.
battle['Opposite_Sign'] = np.sign(battle['beta_DN']) != np.sign(battle['beta_UP'])
battle['pheno_label'] = (battle['pheno'].astype(int)
                         .map(PHENO_LABELS)
                         .fillna(battle['pheno'].astype(str)))

n_all  = len(battle)
n_opposite = battle['Opposite_Sign'].sum()
p_all  = stats.binomtest(n_opposite, n_all, p=0.5).pvalue

focus = battle[battle['Max_LogP'] >= LP_FOC].copy()
n_f   = len(focus)
n_fr  = focus['Opposite_Sign'].sum()
p_foc = stats.binomtest(n_fr, n_f, p=0.5).pvalue

# --- per-pair classification from the joint refit ---------------------------
# Merge the 06_rheostat_beta_equality.py output onto the highlighted pairs and
# classify each pair on the significance of the two arms individually, not on
# sign alone (see rheostat_classify.py). The refit betas are kept alongside the
# REGENIE marginal betas under an _eq suffix, so the classification, the p-values
# and the coefficients it acts on all come from the same fit.
def load_equality(path):
    if not Path(path).exists():
        print(f"  [beta-equality] {path} not found; pairs marked 'untested'.")
        return None
    eq = pd.read_csv(path, sep='\t')
    # 06's pheno is the 'p'-prefixed trait name (p30300); battle uses the int id.
    eq['pheno'] = (eq['pheno'].astype(str).str.replace(r'^p', '', regex=True)
                   .astype(int))
    eq['gene'] = eq['gene'].astype(str)
    return eq[['gene', 'pheno', 'beta_UP', 'beta_DN', 'se_UP', 'se_DN',
               'p_UP', 'p_DN', 'p_symmetry', 'p_equality', 'opposite_sign']]

eq = load_equality(EQ_FILE)
if eq is not None:
    focus = focus.merge(eq, on=['gene', 'pheno'], how='left',
                        suffixes=('', '_eq'))
else:
    for _c in ('beta_UP_eq', 'beta_DN_eq', 'se_UP_eq', 'se_DN_eq',
               'p_UP', 'p_DN', 'p_symmetry', 'p_equality'):
        focus[_c] = np.nan

# Classify on the refit coefficients/p-values (beta_*_eq), never the marginal ones.
focus = classify(focus, alpha=EQ_ALPHA, beta_UP='beta_UP_eq', beta_DN='beta_DN_eq')

cat_counts = count_table(focus)
print(f"  [classification] {len(focus)} highlighted pairs "
      f"(alpha = {EQ_ALPHA} on each arm):")
for _, _r in cat_counts[cat_counts['n'] > 0].iterrows():
    print(f"      {_r['category']:<10s} {_r['category_detail']:<32s} n={_r['n']}")

slide_rows = []
for t in np.arange(0.0, 3.6, 0.5):
    sub = battle[battle['Max_LogP'] >= t]
    if len(sub) < 5: break
    r = sub['Opposite_Sign'].sum()
    p = stats.binomtest(r, len(sub), p=0.5).pvalue
    slide_rows.append({'thr': t, 'pct': r/len(sub)*100, 'n': len(sub), 'p': p, 'k': r})
slide = pd.DataFrame(slide_rows)

# Cochran-Armitage trend test on DISJOINT Max_LogP bins.
# NOTE: the `slide` table above is cumulative — each threshold nests the next,
# so the same gene-phenotype pair is recounted in every bin it clears. That is
# fine for the Panel-C visualisation, but a trend test requires *independent*
# groups. Here we re-bin every pair into non-overlapping Max_LogP intervals and
# test for an ordinal increase in the rheostat proportion across them.
_ca_edges  = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
_ca_scores = np.arange(len(_ca_edges) - 1)            # ordinal scores 0,1,2,3
_ca_bin    = pd.cut(battle['Max_LogP'], bins=_ca_edges,
                    right=False, labels=False)
_k = np.array([int(battle.loc[_ca_bin == i, 'Opposite_Sign'].sum()) for i in _ca_scores])
_n = np.array([int((_ca_bin == i).sum())                          for i in _ca_scores])
_nz = _n > 0                                          # ignore empty bins
_p_pool   = _k.sum() / _n.sum()
_w_mean_t = np.average(_ca_scores[_nz], weights=_n[_nz])
_ca_var   = _p_pool * (1 - _p_pool) * np.sum(_n[_nz] * (_ca_scores[_nz] - _w_mean_t)**2)
if _ca_var > 0:
    _T_CA    = np.sum(_ca_scores[_nz] * (_k[_nz] - _n[_nz] * _p_pool)) / np.sqrt(_ca_var)
    p_CA_one = stats.norm.sf(_T_CA)   # one-sided: increasing trend
    p_CA_two = 2 * stats.norm.sf(abs(_T_CA))
else:
    _T_CA, p_CA_one, p_CA_two = np.nan, np.nan, np.nan

print(f"  n={n_all}, opposite-sign={n_opposite} ({n_opposite/n_all*100:.1f}%), p={p_all:.3e}")
print(f"  logP≥{LP_FOC}: {n_fr}/{n_f} = {n_fr/n_f*100:.0f}%, p={p_foc:.3f}")
print(f"  Cochran-Armitage T={_T_CA:.3f}, p(one-sided)={p_CA_one:.3f}, p(two-sided)={p_CA_two:.3f}")

fig = plt.figure(figsize=(7.5, 8.2), dpi=300)
gs = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.05, 1.0],
    hspace=0.26,
    left=0.13, right=0.90, top=0.96, bottom=0.07,
)
# Bottom row: panel b = opposite-sign trend (left), panel c = per-pair
# Z-scores (right). The bar chart keeps the wider cell because its two-line
# gene/trait tick labels need the room.
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[1],
    width_ratios=[1.0, 1.4], wspace=0.55,
)

gs_top = gridspec.GridSpecFromSubplotSpec(
    4, 4, subplot_spec=gs[0],
    hspace=0.06, wspace=0.06,
)
ax_a     = fig.add_subplot(gs_top[1:, :-1])
ax_a_top = fig.add_subplot(gs_top[0,  :-1], sharex=ax_a)
ax_a_rgt = fig.add_subplot(gs_top[1:,  -1], sharey=ax_a)

ax_trend = fig.add_subplot(gs_bot[0])   # panel b
ax_bars  = fig.add_subplot(gs_bot[1])   # panel c

bg  = battle[battle['Max_LogP'] < LP_FOC]
ax_a.scatter(bg['z_DN'], bg['z_UP'],
             color=C_BG, s=8, alpha=0.55, rasterized=True, zorder=1, linewidths=0)

# Neutral shading of the two opposite-sign quadrants. Deliberately NOT the
# rheostat colour: opposite sign alone does not make a rheostat, and tinting
# these quadrants teal would re-assert the sign-only claim the categories drop.
ax_a.axhspan( 0,  9, xmin=0.0, xmax=0.5, color='#95A5A6', alpha=0.05, zorder=0)
ax_a.axhspan(-9,  0, xmin=0.5, xmax=1.0, color='#95A5A6', alpha=0.05, zorder=0)
ax_a.axhline(0, color='#AAAAAA', lw=0.7, zorder=2)
ax_a.axvline(0, color='#AAAAAA', lw=0.7, zorder=2)

for _, r in focus.iterrows():
    c = CATEGORY_COLORS[r['category']]
    m = CATEGORY_MARKERS[r['category']]
    # gold ring = asymmetric rheostat: opposite sign, BOTH arms significant, and
    # the mirror contrast beta_DN = -beta_UP rejected (drawn behind the marker).
    if r['category_detail'] == 'rheostat, asymmetric':
        ax_a.scatter(r['z_DN'], r['z_UP'], s=210, marker='o',
                     facecolors='none', edgecolors='#F1C40F',
                     linewidths=1.7, zorder=4)
    ax_a.scatter(r['z_DN'], r['z_UP'],
                 c=c, s=90, marker=m,
                 edgecolors='white', linewidths=0.7, zorder=5)

labeled = set()
offsets = {
    'TSEN2':  (-32,  6), 'BCL7A':  ( 7,  4),
    'SLC39A9':( 7,  4), 'OR2T6':  ( 7, -12),
}
for _, r in focus.sort_values('Max_LogP', ascending=False).iterrows():
    if r['gene'] not in labeled:
        ox, oy = offsets.get(r['gene'], (7, 4))
        ax_a.annotate(
            r['gene'],
            xy=(r['z_DN'], r['z_UP']),
            xytext=(ox, oy), textcoords='offset points',
            fontsize=7.5, fontweight='bold', color='#1A1A1A',
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.5),
        )
        labeled.add(r['gene'])

from scipy.stats import gaussian_kde
for z_vals, ax_m, vert in [(battle['z_DN'], ax_a_top, False),
                            (battle['z_UP'], ax_a_rgt, True)]:
    z_clean = z_vals.dropna()
    xg = np.linspace(-9, 9, 400)
    kde = gaussian_kde(z_clean, bw_method=0.35)(xg)
    if vert:
        ax_m.fill_betweenx(xg, 0, kde, color='#85929E', alpha=0.4)
        ax_m.axhline(0, color='#AAAAAA', lw=0.5)
    else:
        ax_m.fill_between(xg, 0, kde, color='#85929E', alpha=0.4)
        ax_m.axvline(0, color='#AAAAAA', lw=0.5)
    ax_m.axis('off')

ax_a.set_xlim(-9, 9); ax_a.set_ylim(-9, 9)
ax_a.set_xlabel(r'Repressor Z-score  ($Z_\mathrm{DN}$)', fontsize=9, fontweight='bold')
ax_a.set_ylabel(r'Enhancer Z-score  ($Z_\mathrm{UP}$)',  fontsize=9, fontweight='bold')
ax_a.spines[['top','right']].set_visible(False)

ax_a.text(-8.5, 8.2,
          f'n = {n_all:,} gene–phenotype pairs\n'
          f'{n_opposite/n_all*100:.1f}% opposite sign, unfiltered  (P = {p_all:.3f})',
          fontsize=7.2, va='top', color='#444444', linespacing=1.5)

# Legend lists only the categories actually present among the highlighted pairs.
_present = [c for c in CATEGORY_ORDER if (focus['category'] == c).any()]
leg_a = [mlines.Line2D([], [], marker=CATEGORY_MARKERS[c], linestyle='none',
                       markerfacecolor=CATEGORY_COLORS[c],
                       markeredgecolor='white', markeredgewidth=0.7,
                       markersize=7.5, label=CATEGORY_LABELS[c])
         for c in _present]
leg_a.append(mpatches.Patch(facecolor=C_BG, edgecolor='none',
                            label=f'Background (logP < {LP_FOC:.0f})'))
if (focus['category_detail'] == 'rheostat, asymmetric').any():
    leg_a.append(mlines.Line2D([], [], marker='o', linestyle='none',
                               markerfacecolor='none', markeredgecolor='#F1C40F',
                               markeredgewidth=1.7, markersize=9,
                               label='Asymmetric rheostat (β$_{DN}$ = −β$_{UP}$ rejected)'))
ax_a.legend(handles=leg_a, frameon=False, fontsize=6.8,
            loc='lower right', handlelength=1.0,
            labelspacing=0.3, borderpad=0.3)

ax_a.text(-0.18, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='top')

# Rows are grouped by category (CATEGORY_ORDER), ascending Max_LogP within each
# group, with same-gene rows kept adjacent (TSEN2 last within its block).
_blocks = []
for _cat in CATEGORY_ORDER:
    _sub = focus[focus['category'] == _cat]
    if _sub.empty:
        continue
    _non = _sub[_sub['gene'] != 'TSEN2'].sort_values('Max_LogP', ascending=True)
    _ts  = _sub[_sub['gene'] == 'TSEN2'].sort_values('Max_LogP', ascending=True)
    _blocks.append((_cat, pd.concat([_non, _ts])))
plot_df = pd.concat([b for _, b in _blocks]).reset_index(drop=True)

# gene name gets a trailing dagger when the pair is an ASYMMETRIC RHEOSTAT
_conf_glyph = np.where(plot_df['category_detail'] == 'rheostat, asymmetric',
                       ' ‡', '')
plot_df['row_label'] = (plot_df['gene'] + _conf_glyph + '\n' +
                        plot_df['pheno_label'].str[:20])

Z_CLIP = 5.0
h = 0.32

for i, (_, r) in enumerate(plot_df.iterrows()):
    stripe = '#F7F8F9' if i % 2 == 0 else 'white'
    ax_bars.axhspan(i - 0.5, i + 0.5, color=stripe, zorder=0)
    z_dn_plot = np.clip(r['z_DN'], -Z_CLIP, Z_CLIP)
    z_up_plot = np.clip(r['z_UP'], -Z_CLIP, Z_CLIP)
    ax_bars.barh(i + h/2, z_dn_plot, height=h, color=C_DN,
              edgecolor='none', alpha=0.88, zorder=2)
    ax_bars.barh(i - h/2, z_up_plot, height=h, color=C_UP,
              edgecolor='none', alpha=0.88, zorder=2)

ax_bars.set_xlim(-Z_CLIP - 2.6, Z_CLIP + 0.4)   # left gutter holds the category labels
ax_bars.axvline(0, color='#2C2C2C', lw=0.8, zorder=3)

# Dashed separator + italic label at each category boundary.
_row = 0
for _cat, _blk in _blocks:
    if _row > 0:
        ax_bars.axhline(_row - 0.5, color='#CCCCCC', lw=0.8, ls='--', zorder=3)
    # Sit the label just BELOW the divider, inside its own block, so it never
    # rides on the dashed line or on the block above it.
    ax_bars.text(-Z_CLIP - 2.45, _row - 0.30, _cat,
              fontsize=6.2, color=CATEGORY_COLORS[_cat],
              va='top', ha='left', fontstyle='italic', zorder=5)
    _row += len(_blk)

ax_bars.set_yticks(range(len(plot_df)))
ax_bars.set_yticklabels(plot_df['row_label'], fontsize=7.0)
ax_bars.invert_yaxis()
ax_bars.set_xlabel(r'Z-score  ($Z_\mathrm{DN}$  or  $Z_\mathrm{UP}$)',
                fontsize=9, fontweight='bold')
ax_bars.spines[['top','right']].set_visible(False)
ax_bars.tick_params(axis='y', length=0, pad=3)
ax_bars.set_ylim(len(plot_df) - 0.5, -0.5)

leg_bars = [
    mpatches.Patch(facecolor=C_DN, edgecolor='none', label='DN (repressor)'),
    mpatches.Patch(facecolor=C_UP, edgecolor='none', label='UP (enhancer)'),
]
# Inside the axes, lower-right — right of the x = 0 rule and over the emptiest
# region: the bottom two rows have essentially no positive bars. The left edge
# is pinned at x = 0.6 in DATA coords (blended transform: data-x, axes-y) so the
# box can never spill back across the zero rule onto the negative bars, however
# wide it is. A light white frame keeps the entries legible over the stripes.
from matplotlib.transforms import blended_transform_factory
_bt = blended_transform_factory(ax_bars.transData, ax_bars.transAxes)
_leg = ax_bars.legend(handles=leg_bars, fontsize=6.3,
            loc='lower left', bbox_to_anchor=(0.6, 0.015), bbox_transform=_bt,
            handlelength=0.9, labelspacing=0.3, borderaxespad=0.0,
            frameon=True, framealpha=0.9, edgecolor='none')
_leg.get_frame().set_facecolor('white')
_leg.set_zorder(6)

ax_bars.text(-0.30, 1.02, 'c', transform=ax_bars.transAxes,
          fontsize=13, fontweight='bold', va='top')

_foot = (f'Categories from the joint refit y ~ covariates + burden$_{{UP}}$ + '
         f'burden$_{{DN}}$, on the significance of each arm individually '
         f'(P < {EQ_ALPHA}): rheostat = opposite sign with both arms '
         f'significant; UP-driven / DN-driven = that arm alone significant '
         f'(sign not part of the definition); genetic fragility = same sign '
         f'with both arms significant. A rejected β-equality test indicates '
         f'that the two coefficients differ in magnitude, not that they oppose.')
if (plot_df['category_detail'] == 'rheostat, asymmetric').any():
    _foot += ('  ‡ asymmetric rheostat: the mirror contrast β$_{DN}$ = '
              '−β$_{UP}$ is rejected.')
fig.text(0.02, -0.012, _foot, fontsize=5.6, color='#666666',
         va='top', ha='left', wrap=True)

# Significance is shown by the 95% CI, not by a separate marker: a point is
# significantly different from the 50% null exactly where its CI excludes 50%.
# Wilson score interval (better than normal-approx at small n and near 0/1);
# _k opposite-sign out of _n at each threshold.
_z95   = stats.norm.ppf(0.975)
_kk    = slide['k'].to_numpy(dtype=float)
_nn    = slide['n'].to_numpy(dtype=float)
_phat  = _kk / _nn
_denom = 1.0 + _z95**2 / _nn
_centre = (_phat + _z95**2 / (2 * _nn)) / _denom
_hw     = (_z95 * np.sqrt(_phat * (1 - _phat) / _nn
                          + _z95**2 / (4 * _nn**2))) / _denom
_lo = np.clip((_centre - _hw) * 100, 0, 100)
_hi = np.clip((_centre + _hw) * 100, 0, 100)
_pct = slide['pct'].to_numpy(dtype=float)

ax_trend.axhline(50, color='#AAAAAA', lw=0.9, ls='--', zorder=2)
ax_trend.errorbar(slide['thr'], _pct,
                  yerr=[_pct - _lo, _hi - _pct],
                  fmt='o-', color=C_TREND, lw=1.8, ms=5.0,
                  markeredgecolor='white', markeredgewidth=0.6,
                  ecolor='#F1C40F', elinewidth=1.4,
                  capsize=2.8, capthick=1.2, zorder=4)

ax_trend.set_xlabel(r'Min $-\log_{10}P$ threshold', fontsize=9, fontweight='bold')
ax_trend.set_ylabel('Opposite-sign rate (%)', fontsize=9, fontweight='bold')
ax_trend.set_xlim(-0.3, 3.6)
ax_trend.set_ylim(30, 98)
ax_trend.spines[['top','right']].set_visible(False)
ax_trend.text(0.03, 0.03,
          '95% CI (Wilson); significant where the CI excludes 50%',
          transform=ax_trend.transAxes, fontsize=5.6, color='#888888',
          va='bottom', ha='left')

ax_trend.text(0.97, 0.97,
          f'Overall: {n_opposite/n_all*100:.1f}%  (P = {p_all:.3f})',
          transform=ax_trend.transAxes, fontsize=7.0,
          ha='right', va='top', color=C_TREND, fontweight='bold')

ax_trend.text(-0.26, 1.02, 'b', transform=ax_trend.transAxes,
          fontsize=13, fontweight='bold', va='top')

for fmt in ('png', 'pdf'):
    out = f'Figure3_Publication.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f'Saved: {out}')

# Source-data / supp record for the highlighted pairs.
#
# SCALE WARNING. The joint refit (06_rheostat_beta_equality.py) is NOT on the
# REGENIE scale: REGENIE's --build-mask sum caps the summed weighted dosage at 2
# (the mask is a pseudo-variant), --apply-rint normalises the RESIDUALS, and the
# Step-1 LOCO offset is applied; the refit uses an uncapped weighted burden,
# RINTs the phenotype before regressing covariates, and carries no LOCO offset.
# Its coefficients therefore differ from the REGENIE betas in magnitude and, on
# these data, in sign. They are exported only so the tests can be reproduced,
# and are named *_refit_unscaled to stop them being read as effect sizes.
#
# What IS comparable, and what the classification uses: p_UP, p_DN and the two
# Wald contrasts. Multiplying both coefficients by a common constant (including
# a negative one) leaves the two-sided p-values unchanged, so the categories are
# invariant to the scale and sign difference.
_src_cols = ['gene', 'pheno', 'pheno_label', 'Max_LogP',
             'beta_DN', 'z_DN', 'logp_DN', 'beta_UP', 'z_UP', 'logp_UP',
             'Opposite_Sign',
             'beta_UP_eq', 'se_UP_eq', 'p_UP',
             'beta_DN_eq', 'se_DN_eq', 'p_DN',
             'p_symmetry', 'p_equality',
             'category', 'category_detail']
src = (focus[[c for c in _src_cols if c in focus.columns]]
       .assign(_ord=focus['category'].map(
           {c: i for i, c in enumerate(CATEGORY_ORDER)}))
       .sort_values(['_ord', 'Max_LogP'], ascending=[True, False])
       .drop(columns='_ord'))
src = src.rename(columns={
    'beta_UP_eq': 'beta_UP_refit_unscaled', 'se_UP_eq': 'se_UP_refit_unscaled',
    'beta_DN_eq': 'beta_DN_refit_unscaled', 'se_DN_eq': 'se_DN_refit_unscaled',
})
src.to_csv('Figure3_rheostat_equality.tsv', sep='\t', index=False)

cat_counts.to_csv('Figure3_category_counts.tsv', sep='\t', index=False)
print(f'Saved: Figure3_category_counts.tsv  '
      f'({int(cat_counts["n"].sum())} pairs across '
      f'{int((cat_counts["n"] > 0).sum())} category/sub-flag rows)')
print(f'Saved: Figure3_rheostat_equality.tsv  ({len(src)} highlighted pairs)')
