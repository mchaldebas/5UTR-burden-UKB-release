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
from adjustText import adjust_text

DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))

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

FILE_TRIAD  = DATA_DIR / 'Final_Triad_Comparison.csv.gz'
FILE_NATURE = DATA_DIR / 'Nature_2025_Reproduction_Data.csv'
K_MIN = 15.0

C_5U   = '#2E86C1'   # matches C_DN / C_OBS in Figs 3–5
C_UP   = '#C0392B'   # 5'UTR-UP negative control (matches C_UP in Fig 3)
C_PTV  = '#85929E'   # matches C_FRAG / C_SHARED / C_NONE in Figs 2–5
C_DISC = '#E67E22'   # matches C_SUG / endocrine orange in Figs 2–5
C_MIS  = '#8E44AD'   # exome missense (consortium), Panel B negative control

# Consortium damaging-missense / nonsynonymous CDS masks. The reproduction file
# stores only the single BEST CDS model per gene-phenotype, so a missense beta
# exists ONLY where one of these masks won (13/23 Figure-4 pairs); the bar is
# omitted otherwise. NOTE for the caption: these are the *best* missense mask
# (selected for significance), not a single fixed mask — see Methods.
MISSENSE_MASKS = {'flexdmg', 'flexnonsynmtr', 'raredmg', 'raredmgmtr'}

from pheno_labels import PHENO_LABELS as _PHENO_CANON

# Tighter abbreviations for the dense forest-plot rows; every other field (incl.
# 30820 Rheumatoid factor, WBC, Glucose, Vitamin D) is inherited from the
# canonical mapping below so labels never silently fall back to numeric ids.
_PHENO_SHORT = {
    30010:'RBC count',          30020:'Haemoglobin',        30030:'Haematocrit',
    30040:'MCV',                30050:'MCH',                30060:'MCHC',
    30070:'RBC distrib w.',     30080:'Platelet count',     30090:'Platelet crit',
    30100:'Mean plt vol',       30110:'Plt distrib w.',     30120:'Lymphocyte ct',
    30130:'Monocyte ct',        30140:'Neutrophil ct',      30150:'Eosinophil ct',
    30160:'Basophil ct',        30180:'Lymphocyte %',       30190:'Monocyte %',
    30200:'Neutrophil %',       30210:'Eosinophil %',       30220:'Basophil %',
    30240:'Reticulocyte %',     30250:'Reticulocyte ct',    30260:'Mean retic vol',
    30270:'Mean sph cell vol',  30280:'Immature retic fr.', 30290:'High lt scat %',
    30300:'High lt scat ct',    30600:'Albumin',            30610:'Alk phosphatase',
    30620:'ALT',                30630:'Apolipoprotein A',   30640:'Apolipoprotein B',
    30650:'AST',                30660:'Direct bilirubin',   30670:'Urea',
    30680:'Calcium',            30690:'Cholesterol',        30700:'Creatinine',
    30710:'CRP',                30720:'Cystatin C',         30730:'GGT',
    30750:'HbA1c',              30760:'HDL cholesterol',    30770:'IGF-1',
    30780:'LDL direct',         30790:'Lipoprotein A',      30800:'Oestradiol',
    30810:'Phosphate',          30830:'SHBG',
    30840:'Total bilirubin',    30850:'Testosterone',       30860:'Total protein',
    30870:'Triglycerides',      30880:'Urate',
}
PHENO_SHORT = {**_PHENO_CANON, **_PHENO_SHORT}

print("Loading data…")
df      = pd.read_csv(FILE_TRIAD)
nature  = pd.read_csv(FILE_NATURE)

pheno_map = {}
for _, r in nature.iterrows():
    pheno_map[int(r['phenotype'])] = str(r['phenotype_name']).split('#')[-1].strip()

nature['pheno']  = nature['phenotype'].astype(int)
nature['z_ptv']  = nature['beta.CDS_PTV'] / nature['se.CDS_PTV']
nature['beta_missense'] = np.where(
    nature['model.CDS'].isin(MISSENSE_MASKS), nature['beta.CDS'], np.nan)
nature_ref = nature[['gene','pheno','beta.CDS_PTV','se.CDS_PTV','z_ptv',
                     'model.CDS','beta_missense']].copy()
nature_ref.columns = ['gene','pheno','beta_ptv','se_ptv','z_ptv',
                      'cds_model','beta_missense']

dn_af5 = df[(df['model'] == 'Binary_DN') & (df['mask'] == 'Binary_DN_AF5')].copy()
dn_af5['z'] = dn_af5['beta'] / dn_af5['se']
sub_af5 = pd.merge(dn_af5, nature_ref, on=['gene','pheno'])
sub_af5 = sub_af5[sub_af5['k'] >= K_MIN].copy()
sub_af5['pheno_label'] = sub_af5['pheno'].map(PHENO_SHORT).fillna(
    sub_af5['pheno'].map(pheno_map).fillna(sub_af5['pheno'].astype(str)))

sub = sub_af5.copy()

r_val, p_val = stats.pearsonr(sub['z_ptv'], sub['z'])
print(f"  n={len(sub)}, r={r_val:.3f}, p={p_val:.2e}")

# --- 5'UTR-UP negative control (Aurelie comments #13/#14) --------------------
# PTV is a loss-of-function mask; the enhancer/UP mask should NOT track it. We
# show UP only on the scale-free Z-score axis, not the beta forest: no unweighted
# Binary_UP mask exists, and comparing weighted-UP betas against the unweighted
# Binary_DN betas would be apples-to-oranges (Aurelie comment #15 on weighting).
up_af5 = df[df['mask'] == 'Mirror_UP_AF5'].copy()
up_af5['z_up'] = up_af5['beta'] / up_af5['se']
up_sub = sub[['gene', 'pheno', 'z_ptv']].merge(
    up_af5[['gene', 'pheno', 'z_up']], on=['gene', 'pheno'], how='inner')
if len(up_sub) >= 3:
    r_up, p_up = stats.pearsonr(up_sub['z_ptv'], up_sub['z_up'])
else:
    r_up, p_up = np.nan, np.nan
print(f"  UP negative control: n={len(up_sub)}, r={r_up:.3f}, p={p_up:.3f}")

top23 = (sub_af5.sort_values('logp', ascending=False)
         .head(23))
top23['concordant'] = top23['beta'] * top23['beta_ptv'] > 0
top23 = (top23.sort_values(['concordant', 'logp'], ascending=[False, False])
         .reset_index(drop=True))
top23['row_label']  = top23['gene'] + '\n' + top23['pheno_label'].str[:18]

fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        width_ratios=[1.05, 1.0], wspace=0.50,
                        left=0.13, right=0.97, top=0.96, bottom=0.07)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

x_all  = sub['z_ptv'].values
y_all  = sub['z'].values
slope, intercept = np.polyfit(x_all, y_all, 1)
x_fit  = np.linspace(x_all.min() - 1, x_all.max() + 1, 300)

ax_a.axhline(0, color='#CCCCCC', lw=0.7, zorder=1)
ax_a.axvline(0, color='#CCCCCC', lw=0.7, zorder=1)

# --- UP negative control (drawn underneath the DN layer) --------------------
if len(up_sub):
    ax_a.scatter(up_sub['z_ptv'], up_sub['z_up'],
                 facecolors='none', edgecolors=C_UP, s=42,
                 linewidths=1.2, alpha=0.9, zorder=3)
if np.isfinite(r_up) and len(up_sub) >= 2:
    xu = up_sub['z_ptv'].values
    su, iu = np.polyfit(xu, up_sub['z_up'].values, 1)
    xu_fit = np.linspace(xu.min() - 1, xu.max() + 1, 200)   # UP data range only
    ax_a.plot(xu_fit, su * xu_fit + iu, color=C_UP, lw=1.5, ls='--', zorder=3)

# --- DN signal --------------------------------------------------------------
ax_a.plot(x_fit, slope * x_fit + intercept, color=C_5U, lw=1.8, zorder=5)
for _, row in top23.iterrows():
    c = C_5U if row['concordant'] else C_DISC
    ax_a.scatter(row['z_ptv'], row['z'],
                 color=c, s=55, zorder=6,
                 edgecolors='white', linewidths=0.7)

# Headroom above the topmost point so the stat lines sit clear of GP1BA.
_y0, _y1 = ax_a.get_ylim()
ax_a.set_ylim(_y0, _y1 + 0.07 * (_y1 - _y0))

# Correlation stat boxes — created BEFORE the labels so adjustText treats them
# as obstacles (keeps the top GP1BA label off the "(n = 23)" line).
p_str_dn = f'P = {p_val:.2e}' if p_val < 0.001 else f'P = {p_val:.4f}'
p_str_up = f'P = {p_up:.2f}' if np.isfinite(p_up) else 'P = n/a'
t_dn = ax_a.text(0.05, 0.95,
                 f"5'UTR-DN vs PTV:  r = {r_val:.2f}, {p_str_dn}  (n = {len(sub)})",
                 transform=ax_a.transAxes, fontsize=8.3, va='top',
                 color=C_5U, fontweight='bold', zorder=8)
t_up = ax_a.text(0.05, 0.935,
                 f"5'UTR-UP vs PTV:  r = {r_up:.2f}, {p_str_up}  (n = {len(up_sub)})",
                 transform=ax_a.transAxes, fontsize=8.3, va='top',
                 color=C_UP, fontweight='bold', zorder=8)

# Gene labels (DN only) with adjustText to avoid overlap
texts_a = []
for _, row in top23.iterrows():
    c = C_5U if row['concordant'] else C_DISC
    texts_a.append(ax_a.text(row['z_ptv'], row['z'], row['gene'],
                             fontsize=6.5, fontweight='bold', color=c, zorder=7))
# Repel labels from *every* marker (DN + UP) and from the stat boxes, so the
# dense origin cluster, the red UP circles, and the "(n = 23)" line stay clear.
pts_x = np.concatenate([top23['z_ptv'].values, up_sub['z_ptv'].values])
pts_y = np.concatenate([top23['z'].values,     up_sub['z_up'].values])
adjust_text(
    texts_a, x=pts_x, y=pts_y, ax=ax_a, objects=[t_dn, t_up],
    expand=(1.6, 1.9),
    force_text=(0.5, 0.7),
    force_static=(0.35, 0.5),
    min_arrow_len=8,
    arrowprops=dict(arrowstyle='-', color='#BBBBBB', lw=0.5),
)

ax_a.set_xlabel(r"Exome PTV Z-score ($Z_\mathrm{PTV}$)",
                fontsize=9, fontweight='bold')
ax_a.set_ylabel(r"5'UTR burden Z-score ($Z_\mathrm{5'UTR}$)",
                fontsize=9, fontweight='bold')
ax_a.spines[['top', 'right']].set_visible(False)

leg_a = [
    mlines.Line2D([], [], marker='o', ls='none', mfc=C_5U, mec='white',
                  ms=7, label="5'UTR-DN (signal)"),
    mlines.Line2D([], [], marker='o', ls='none', mfc=C_DISC, mec='white',
                  ms=7, label='DN discordant with PTV'),
    mlines.Line2D([], [], marker='o', ls='none', mfc='none', mec=C_UP,
                  mew=1.2, ms=7, label="5'UTR-UP (negative control)"),
]
ax_a.legend(handles=leg_a, loc='lower right', frameon=False, fontsize=7.0,
            handlelength=1.0, labelspacing=0.35, borderpad=0.3)

ax_a.text(-0.12, 1.04, 'a', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='top')

bh = 0.26   # three bars per row: 5'UTR-DN (top), PTV (mid), missense (bottom)
n_mis = int(top23['beta_missense'].notna().sum())
for i, (_, row) in enumerate(top23.iterrows()):
    stripe = '#F7F8F9' if i % 2 == 0 else 'white'
    ax_b.axhspan(i - 0.5, i + 0.5, color=stripe, zorder=0)

    c_5u  = C_5U  if row['concordant'] else C_DISC

    ax_b.barh(i + bh, row['beta'],     height=bh,
              color=c_5u,  edgecolor='none', alpha=0.88, zorder=3)
    ax_b.barh(i,      row['beta_ptv'], height=bh,
              color=C_PTV, edgecolor='none', alpha=0.78, zorder=3)
    if pd.notna(row['beta_missense']):
        ax_b.barh(i - bh, row['beta_missense'], height=bh,
                  color=C_MIS, edgecolor='none', alpha=0.82, zorder=3)

ax_b.axvline(0, color='#2C2C2C', lw=0.8, zorder=4)

x_min = np.nanmin([top23['beta'].min(), top23['beta_ptv'].min(),
                   top23['beta_missense'].min()])
x_max = np.nanmax([top23['beta'].max(), top23['beta_ptv'].max(),
                   top23['beta_missense'].max()])
pad = (x_max - x_min) * 0.14
ax_b.set_xlim(x_min - pad - 0.5, x_max + pad)

ax_b.set_yticks(range(len(top23)))
ax_b.set_yticklabels(top23['row_label'], fontsize=8.5, fontweight='bold')
ax_b.invert_yaxis()
ax_b.set_xlabel(r"Effect size ($\beta$, SD units)",
                fontsize=9, fontweight='bold')
ax_b.spines[['top', 'right']].set_visible(False)
ax_b.tick_params(axis='y', length=0, pad=2)

leg_b = [
    mpatches.Patch(facecolor=C_5U,   edgecolor='none', label="5'UTR-DN (5ULTRA)"),
    mpatches.Patch(facecolor=C_PTV,  edgecolor='none', label='Exome PTV (consortium)'),
    mpatches.Patch(facecolor=C_MIS,  edgecolor='none',
                   label=f'Exome missense (consortium, n={n_mis})'),
    mpatches.Patch(facecolor=C_DISC, edgecolor='none', label='Discordant direction'),
]
ax_b.legend(handles=leg_b, frameon=False, fontsize=9,
            loc='upper left', handlelength=1.0, labelspacing=0.25,
            borderpad=0.3)

ax_b.text(-0.30, 1.04, 'b', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', va='top')

for fmt in ('png', 'pdf'):
    out = f'Figure4_Publication.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f'Saved: {out}')
