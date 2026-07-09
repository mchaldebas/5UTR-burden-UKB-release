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

DATA_DIR = Path(os.environ.get('DATA_DIR', '/Volumes/MCHALDEBAS3/UKB-500k/UKB-data'))

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

FILE   = DATA_DIR / "Master_Results_Clean.csv.gz"
TRUE_K = 5.0
LP_FOC = 3.0

# Per-pair DN-vs-UP beta-equality test (06_rheostat_beta_equality.py output).
# Optional overlay: when this file is present, highlighted pairs whose UP and DN
# effects are each individually significant, of opposing sign, AND for which a
# single shared burden effect is rejected (equality contrast) are flagged as
# beta-equality-CONFIRMED rheostats. Absent => the figure renders exactly as
# before (population-level sign/binomial story only). Point RHEO_EQ_FILE at the
# cluster output to enable.
EQ_FILE  = Path(os.environ.get('RHEO_EQ_FILE',
                               DATA_DIR / "rheostat_beta_equality_nfe.tsv"))
EQ_ALPHA = 0.05

C_DN   = '#2E86C1'
C_UP   = '#C0392B'
C_RHEO = '#16A085'
C_FRAG = '#7F8C8D'
C_BG   = '#E8ECF0'

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
battle['Is_Rheostat'] = np.sign(battle['beta_DN']) != np.sign(battle['beta_UP'])
battle['pheno_label'] = (battle['pheno'].astype(int)
                         .map(PHENO_LABELS)
                         .fillna(battle['pheno'].astype(str)))

n_all  = len(battle)
n_rheo = battle['Is_Rheostat'].sum()
p_all  = stats.binomtest(n_rheo, n_all, p=0.5).pvalue

focus = battle[battle['Max_LogP'] >= LP_FOC].copy()
n_f   = len(focus)
n_fr  = focus['Is_Rheostat'].sum()
p_foc = stats.binomtest(n_fr, n_f, p=0.5).pvalue

# --- optional per-pair beta-equality overlay --------------------------------
# Merge the 06_rheostat_beta_equality.py output onto the highlighted pairs. The
# formal test shows these effects are DN-dominant: the UP arm is not
# individually significant in any pair, so we do NOT claim symmetric rheostats.
# Instead we flag ASYMMETRIC-DIRECTIONAL pairs — opposing sign, the DN arm
# significant, and a single shared UP=DN effect rejected (p_equality). This
# marks a genuinely directional effect whose two masks differ, without
# over-claiming a mirror. p_symmetry is carried through descriptively.
def load_equality(path):
    if not Path(path).exists():
        print(f"  [beta-equality] {path} not found; overlay OFF.")
        return None
    eq = pd.read_csv(path, sep='\t')
    # 06's pheno is the 'p'-prefixed trait name (p30300); battle uses the int id.
    eq['pheno'] = (eq['pheno'].astype(str).str.replace(r'^p', '', regex=True)
                   .astype(int))
    eq['gene'] = eq['gene'].astype(str)
    eq['eq_asym'] = (
        eq['opposite_sign'].astype(bool)
        & (eq['p_DN'] < EQ_ALPHA)
        & (eq['p_equality'] < EQ_ALPHA)
    )
    return eq[['gene', 'pheno', 'beta_UP', 'beta_DN', 'p_UP', 'p_DN',
               'p_symmetry', 'p_equality', 'opposite_sign', 'eq_asym']]

eq = load_equality(EQ_FILE)
if eq is not None:
    focus = focus.merge(eq, on=['gene', 'pheno'], how='left',
                        suffixes=('', '_eq'))
    focus['eq_asym'] = focus['eq_asym'].fillna(False)
    n_asym = int(focus['eq_asym'].sum())
    print(f"  [beta-equality] {n_asym}/{n_fr} opposing-sign pairs are "
          f"asymmetric-directional (DN P<{EQ_ALPHA}, shared UP=DN effect "
          f"rejected); UP arm not individually significant in any pair.")
else:
    focus['eq_asym'] = False

slide_rows = []
for t in np.arange(0.0, 3.6, 0.5):
    sub = battle[battle['Max_LogP'] >= t]
    if len(sub) < 5: break
    r = sub['Is_Rheostat'].sum()
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
_k = np.array([int(battle.loc[_ca_bin == i, 'Is_Rheostat'].sum()) for i in _ca_scores])
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

print(f"  n={n_all}, rheostat={n_rheo} ({n_rheo/n_all*100:.1f}%), p={p_all:.3e}")
print(f"  logP≥{LP_FOC}: {n_fr}/{n_f} = {n_fr/n_f*100:.0f}%, p={p_foc:.3f}")
print(f"  Cochran-Armitage T={_T_CA:.3f}, p(one-sided)={p_CA_one:.3f}, p(two-sided)={p_CA_two:.3f}")

fig = plt.figure(figsize=(7.5, 9.0), dpi=300)
gs = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.05, 1.0],
    hspace=0.52,
    left=0.13, right=0.97, top=0.96, bottom=0.07,
)
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[1],
    width_ratios=[1.4, 1.0], wspace=0.52,
)

gs_top = gridspec.GridSpecFromSubplotSpec(
    4, 4, subplot_spec=gs[0],
    hspace=0.06, wspace=0.06,
)
ax_a     = fig.add_subplot(gs_top[1:, :-1])
ax_a_top = fig.add_subplot(gs_top[0,  :-1], sharex=ax_a)
ax_a_rgt = fig.add_subplot(gs_top[1:,  -1], sharey=ax_a)

ax_b = fig.add_subplot(gs_bot[0])
ax_c = fig.add_subplot(gs_bot[1])

bg  = battle[battle['Max_LogP'] < LP_FOC]
ax_a.scatter(bg['z_DN'], bg['z_UP'],
             color=C_BG, s=8, alpha=0.55, rasterized=True, zorder=1, linewidths=0)

ax_a.axhspan( 0,  9, xmin=0.0, xmax=0.5, color=C_RHEO, alpha=0.04, zorder=0)
ax_a.axhspan(-9,  0, xmin=0.5, xmax=1.0, color=C_RHEO, alpha=0.04, zorder=0)
ax_a.axhline(0, color='#AAAAAA', lw=0.7, zorder=2)
ax_a.axvline(0, color='#AAAAAA', lw=0.7, zorder=2)

for _, r in focus.iterrows():
    c = C_RHEO if r['Is_Rheostat'] else C_FRAG
    m = 'o'    if r['Is_Rheostat'] else 's'
    # gold ring = asymmetric-directional (β-equality) (drawn behind the marker)
    if r.get('eq_asym', False):
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
          f'{n_rheo/n_all*100:.1f}% opposing direction  (P = {p_all:.3f})',
          fontsize=7.2, va='top', color='#444444', linespacing=1.5)

leg_a = [
    mpatches.Patch(facecolor=C_RHEO, edgecolor='none', label=f'Opposing direction  (logP ≥ {LP_FOC:.0f})'),
    mpatches.Patch(facecolor=C_FRAG, edgecolor='none', label=f'Same direction  (logP ≥ {LP_FOC:.0f})'),
    mpatches.Patch(facecolor=C_BG,   edgecolor='none', label='Background'),
]
if bool(focus['eq_asym'].any()):
    leg_a.append(mlines.Line2D([], [], marker='o', linestyle='none',
                               markerfacecolor='none', markeredgecolor='#F1C40F',
                               markeredgewidth=1.7, markersize=9,
                               label='Asymmetric directional (β-equality)'))
ax_a.legend(handles=leg_a, frameon=False, fontsize=6.8,
            loc='lower right', handlelength=1.0,
            labelspacing=0.3, borderpad=0.3)

ax_a.text(-0.18, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='top')

rheo_df = focus[focus['Is_Rheostat']].copy()
frag_df = focus[~focus['Is_Rheostat']].copy()

tsen2 = rheo_df[rheo_df['gene'] == 'TSEN2'].sort_values('Max_LogP', ascending=True)
other = rheo_df[rheo_df['gene'] != 'TSEN2'].sort_values('Max_LogP', ascending=True)
frag_s = frag_df.sort_values('Max_LogP', ascending=True)
plot_df = pd.concat([frag_s, other, tsen2]).reset_index(drop=True)

# gene name gets a trailing dagger when the pair is asymmetric-directional
_conf_glyph = plot_df['eq_asym'].map({True: ' ‡', False: ''})
plot_df['row_label'] = (plot_df['gene'] + _conf_glyph + '\n' +
                        plot_df['pheno_label'].str[:20])

Z_CLIP = 5.0
h = 0.32
n_frag  = len(frag_s)
n_other = len(other)
n_tsen2 = len(tsen2)

tsen2_start = n_frag + n_other
if n_tsen2 > 0:
    ax_b.axhspan(tsen2_start - 0.5, tsen2_start + n_tsen2 - 0.5,
                 color=C_RHEO, alpha=0.05, zorder=0)

for i, (_, r) in enumerate(plot_df.iterrows()):
    stripe = '#F7F8F9' if i % 2 == 0 else 'white'
    ax_b.axhspan(i - 0.5, i + 0.5, color=stripe, zorder=0)
    z_dn_plot = np.clip(r['z_DN'], -Z_CLIP, Z_CLIP)
    z_up_plot = np.clip(r['z_UP'], -Z_CLIP, Z_CLIP)
    ax_b.barh(i + h/2, z_dn_plot, height=h, color=C_DN,
              edgecolor='none', alpha=0.88, zorder=2)
    ax_b.barh(i - h/2, z_up_plot, height=h, color=C_UP,
              edgecolor='none', alpha=0.88, zorder=2)

ax_b.set_xlim(-Z_CLIP - 0.8, Z_CLIP + 0.4)
ax_b.axvline(0, color='#2C2C2C', lw=0.8, zorder=3)

if n_frag > 0:
    ax_b.axhline(n_frag - 0.5, color='#CCCCCC', lw=0.8, ls='--', zorder=3)
    ax_b.text(-Z_CLIP - 0.35, n_frag - 0.6,
              'Same direction', fontsize=6.2, color=C_FRAG,
              va='bottom', ha='left', fontstyle='italic')

sep2 = n_frag + n_other - 0.5
if n_other > 0 and n_tsen2 > 0:
    ax_b.axhline(sep2, color='#CCCCCC', lw=0.8, ls='--', zorder=3)

if n_tsen2 > 0:
    mid = tsen2_start + (n_tsen2 - 1) / 2
    ax_b.text(Z_CLIP + 0.3, mid, 'TSEN2\nmulti-trait',
              ha='left', va='center', fontsize=6.2, color=C_RHEO,
              fontweight='bold', fontstyle='italic', clip_on=False)

ax_b.set_yticks(range(len(plot_df)))
ax_b.set_yticklabels(plot_df['row_label'], fontsize=7.0)
ax_b.invert_yaxis()
ax_b.set_xlabel(r'Z-score  ($Z_\mathrm{DN}$  or  $Z_\mathrm{UP}$)',
                fontsize=9, fontweight='bold')
ax_b.spines[['top','right']].set_visible(False)
ax_b.tick_params(axis='y', length=0, pad=3)
ax_b.set_ylim(len(plot_df) - 0.5, -0.5)

leg_b = [
    mpatches.Patch(facecolor=C_DN, edgecolor='none',
                   label='DN mask  (repressor variants)'),
    mpatches.Patch(facecolor=C_UP, edgecolor='none',
                   label='UP mask  (enhancer variants)'),
]
ax_b.legend(handles=leg_b, frameon=False, fontsize=6.5,
            loc='upper left', handlelength=1.0, labelspacing=0.3)

ax_b.text(-0.32, 1.02, 'b', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', va='top')

if bool(plot_df['eq_asym'].any()):
    fig.text(0.02, 0.008,
             '‡ asymmetric directional: DN arm significant, opposing sign, '
             'single shared UP=DN effect rejected (β-equality test); UP arm '
             'not independently significant',
             fontsize=5.6, color='#666666', va='bottom', ha='left')

ax_c.plot(slide['thr'], slide['pct'],
          'o-', color=C_RHEO, lw=1.8, ms=5.5,
          markeredgecolor='white', markeredgewidth=0.6, zorder=4)
ax_c.axhline(50, color='#AAAAAA', lw=0.9, ls='--', zorder=2)

sig = slide[slide['p'] < 0.05]
if not sig.empty:
    ax_c.scatter(sig['thr'], sig['pct'],
                 s=160, c='#F1C40F', zorder=6,
                 edgecolors='#2C2C2C', linewidths=0.9)
    for k, (_, sr) in enumerate(sig.iterrows()):
        ax_c.text(sr['thr'], sr['pct'] + 2.2 + k * 2.5, 'p<0.05',
                  ha='center', va='bottom', fontsize=6.0,
                  color='#D4AC0D', fontweight='bold')

for _, sr in slide.iterrows():
    ax_c.text(sr['thr'], sr['pct'] - 2.5,
              f"n={sr['n']:.0f}",
              ha='center', va='top', fontsize=6.0, color='#888888')

ax_c.set_xlabel(r'Min $-\log_{10}P$ threshold', fontsize=9, fontweight='bold')
ax_c.set_ylabel('Opposing-direction rate (%)', fontsize=9, fontweight='bold')
ax_c.set_xlim(-0.3, 3.6)
ax_c.set_ylim(42, 82)
ax_c.spines[['top','right']].set_visible(False)

ax_c.text(0.97, 0.97,
          f'Overall: {n_rheo/n_all*100:.1f}%  (P = {p_all:.3f})',
          transform=ax_c.transAxes, fontsize=7.0,
          ha='right', va='top', color=C_RHEO, fontweight='bold')

ax_c.text(-0.30, 1.02, 'c', transform=ax_c.transAxes,
          fontsize=13, fontweight='bold', va='top')

for fmt in ('png', 'pdf'):
    out = f'Figure3_Publication.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f'Saved: {out}')

# Source-data / supp record for the highlighted pairs, incl. the beta-equality
# columns when available (empty if the overlay was OFF).
_src_cols = ['gene', 'pheno', 'pheno_label', 'Max_LogP',
             'beta_DN', 'z_DN', 'logp_DN', 'beta_UP', 'z_UP', 'logp_UP',
             'Is_Rheostat', 'eq_asym']
_src_cols += [c for c in ('p_UP', 'p_DN', 'p_symmetry', 'p_equality')
              if c in focus.columns]
src = (focus[[c for c in _src_cols if c in focus.columns]]
       .sort_values(['Is_Rheostat', 'Max_LogP'], ascending=[False, False]))
src.to_csv('Figure3_rheostat_equality.tsv', sep='\t', index=False)
print(f'Saved: Figure3_rheostat_equality.tsv  ({len(src)} highlighted pairs)')
