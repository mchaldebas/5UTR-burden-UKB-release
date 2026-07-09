import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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

FILE      = DATA_DIR / "Master_Results_Clean.csv.gz"
THRESHOLD = 6.0
TRUE_K    = 5.0

# Model-selection correction: each suite draws the best of K model×spectrum combinations.
# Bonferroni adds log10(K) to the base threshold to ensure fair comparison.
N_MODELS_5ULTRA   = 8   # 4 models × 2 spectra (rare, af5)
N_MODELS_CADD     = 4   # 2 models × 2 spectra
N_MODELS_ABLATION = 4   # 2 models × 2 spectra
THRESH_U = THRESHOLD + np.log10(N_MODELS_5ULTRA)    # 6.903
THRESH_C = THRESHOLD + np.log10(N_MODELS_CADD)      # 6.602
THRESH_A = THRESHOLD + np.log10(N_MODELS_ABLATION)  # 6.602

from pheno_labels import PHENO_LABELS
KNOWN_PHENOS = set(PHENO_LABELS.keys())

_HEMA  = ['WBC count',
          'RBC count','Haemoglobin','Haematocrit','MCV','MCH','MCHC',
          'RBC distrib width','Platelet count','Platelet crit','Plt distrib width',
          'Mean platelet vol',
          'Lymphocyte count','Monocyte count','Neutrophil count','Eosinophil count',
          'Basophil count','Lymphocyte %','Monocyte %','Neutrophil %',
          'Eosinophil %','Basophil %','Reticulocyte %','Reticulocyte count',
          'Mean reticuloc vol','Mean sphered cell vol','Immature retic fr',
          'High lt scat %','High lt scat count']
_LIPID = ['Apolipoprotein A','Apolipoprotein B','Cholesterol','HDL cholesterol',
          'LDL direct','Triglycerides','Lipoprotein A']
_LIVER = ['Albumin','Total protein','ALT','Alk phosphatase','GGT','AST',
          'Direct bilirubin','Total bilirubin']
_RENAL = ['Urea','Creatinine','Cystatin C','Urate','Urate (alt)']
_BONE  = ['Calcium','Phosphate','Vitamin D']
_ENDO  = ['Glucose','HbA1c','IGF-1','Testosterone','Oestradiol','SHBG']
_INFL  = ['CRP', 'Rheumatoid factor']

PHENO_CATEGORIES = {}
for p in _HEMA:  PHENO_CATEGORIES[p] = 'Hematology'
for p in _LIPID: PHENO_CATEGORIES[p] = 'Lipid'
for p in _LIVER: PHENO_CATEGORIES[p] = 'Liver/Metabolic'
for p in _RENAL: PHENO_CATEGORIES[p] = 'Renal'
for p in _BONE:  PHENO_CATEGORIES[p] = 'Bone/Mineral'
for p in _ENDO:  PHENO_CATEGORIES[p] = 'Endocrine'
for p in _INFL:  PHENO_CATEGORIES[p] = 'Inflammation'

CAT_ORDER  = ['Hematology','Lipid','Liver/Metabolic','Renal','Bone/Mineral','Endocrine','Inflammation']
CAT_LABELS = ['Hematology','Lipid','Liver /\nMetabolic','Renal','Bone /\nMineral','Endocrine','Inflammation']

CATEGORY_COLORS = {
    'Hematology':      '#2E86C1',
    'Lipid':           '#E67E22',
    'Liver/Metabolic': '#229954',
    'Renal':           '#8E44AD',
    'Bone/Mineral':    '#C0392B',
    'Endocrine':       '#117A65',
    'Inflammation':    '#B7950B',
    'Other':           '#7F8C8D',
}

MODELS_5ULTRA   = {'5U_Logic','5ULTRA_Binary','5ULTRA_Weighted','Flux_Joint'}
MODELS_CADD     = {'CADD_Binary','CADD_Weighted'}
MODELS_ABLATION = {'Ablation', 'Ablation_Weighted'}

C_5U_EX  = '#C0392B'
C_SHARED = '#85929E'
C_CD_EX  = '#CCD1D1'
C_ABL    = '#E5E8EA'

print("Loading data…")
df = pd.read_csv(FILE)
df = df[df['true_k'] >= TRUE_K].copy()
df['pheno'] = pd.to_numeric(df['pheno'], errors='coerce')

df['target_group'] = 'Other'
df.loc[df['model'].isin(MODELS_5ULTRA),   'target_group'] = '5ULTRA'
df.loc[df['model'].isin(MODELS_CADD),     'target_group'] = 'CADD'
df.loc[df['model'].isin(MODELS_ABLATION), 'target_group'] = 'Ablation'

u_best = df[df['target_group']=='5ULTRA'].sort_values('logp', ascending=False).drop_duplicates(['gene','pheno'])
c_base = df[df['target_group']=='CADD'].sort_values('logp', ascending=False).drop_duplicates(['gene','pheno'])
c_abl  = df[df['target_group']=='Ablation'].sort_values('logp', ascending=False).drop_duplicates(['gene','pheno'])

def pair_hits(s, threshold):
    return set(s[s['logp'] > threshold].apply(lambda r: f"{r['gene']}_{r['pheno']}", axis=1))

u_hits = pair_hits(u_best, THRESH_U)
c_hits = pair_hits(c_base, THRESH_C)
a_hits = pair_hits(c_abl,  THRESH_A)

n_u_total = len(u_hits)
n_c_total = len(c_hits)
n_shared  = len(u_hits & c_hits)
n_u_ex    = len(u_hits - c_hits)
n_c_ex    = len(c_hits - u_hits)
n_abl     = len(a_hits)
ratio     = n_u_total / n_c_total

u_sig = u_best[u_best['logp'] > THRESH_U].copy()
c_sig = c_base[c_base['logp'] > THRESH_C].copy()
for s in [u_sig, c_sig]:
    s['trait']    = s['pheno'].astype(int).map(PHENO_LABELS).fillna('Other')
    s['category'] = s['trait'].map(PHENO_CATEGORIES).fillna('Other')

u_sig['status'] = u_sig.apply(
    lambda r: 'shared' if f"{r['gene']}_{r['pheno']}" in c_hits else '5u_ex', axis=1)
c_sig['status'] = c_sig.apply(
    lambda r: 'shared' if f"{r['gene']}_{r['pheno']}" in u_hits else 'cadd_ex', axis=1)

cat_data = {}
for cat in CAT_ORDER:
    u_cat = u_sig[u_sig['category'] == cat]
    c_cat = c_sig[c_sig['category'] == cat]
    cat_data[cat] = {
        'shared': len(u_cat[u_cat['status'] == 'shared']),
        'u_ex':   len(u_cat[u_cat['status'] == '5u_ex']),
        'c_ex':   len(c_cat[c_cat['status'] == 'cadd_ex']),
    }

print("  Category data:")
for cat, d in cat_data.items():
    print(f"    {cat:20s}  5ULTRA excl={d['u_ex']:2d}  shared={d['shared']:2d}  CADD excl={d['c_ex']:2d}")
print(f"  Totals: 5ULTRA={n_u_total}, CADD={n_c_total}, shared={n_shared}, ablation={n_abl}")

# Drop categories with zero associations in both scorers
active_idx  = [i for i, cat in enumerate(CAT_ORDER)
               if cat_data[cat]['u_ex'] + cat_data[cat]['shared'] + cat_data[cat]['c_ex'] > 0]
CAT_ORDER  = [CAT_ORDER[i]  for i in active_idx]
CAT_LABELS = [CAT_LABELS[i] for i in active_idx]

novel_all = u_sig[u_sig['status'] == '5u_ex'].copy()
novel_all['pheno_int']   = novel_all['pheno'].astype(int)
novel_all['known_pheno'] = novel_all['pheno_int'].isin(KNOWN_PHENOS)
novel = novel_all[novel_all['known_pheno']].copy()
novel_gene = novel.sort_values('logp', ascending=False).drop_duplicates('gene')
novel_gene['bar_color'] = novel_gene['category'].map(CATEGORY_COLORS)
print(f"  Novel genes (known pheno, 5ULTRA exclusive): {len(novel_gene)}")

# ── FIGURE SKELETON ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.5, 9.5), dpi=300)

gs = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[0.68, 1.80],
    hspace=0.30,
    left=0.13, right=0.97, top=0.96, bottom=0.06,
)
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[1],
    width_ratios=[1.35, 1.0], wspace=0.54,
)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs_bot[0])
ax_c = fig.add_subplot(gs_bot[1])

# ── PANEL a: BUTTERFLY CHART — DISCOVERIES BY PHENOTYPE SYSTEM ───────────────
n_cats = len(CAT_ORDER)
h_bar  = 0.50

x_max_r =  70
x_max_l = -42

for i, cat in enumerate(CAT_ORDER):
    y   = n_cats - 1 - i
    sv  = cat_data[cat]['shared']
    uv  = cat_data[cat]['u_ex']
    cv  = cat_data[cat]['c_ex']

    if i % 2 == 1:
        ax_a.axhspan(y - 0.5, y + 0.5, color='#F4F6F7', zorder=0)

    # RIGHT (5ULTRA): shared from 0→sv, exclusive sv→sv+uv
    if sv > 0:
        ax_a.barh(y, sv, height=h_bar, left=0,
                  color=C_SHARED, edgecolor='none', zorder=2)
    if uv > 0:
        ax_a.barh(y, uv, height=h_bar, left=sv,
                  color=CATEGORY_COLORS[cat], alpha=0.88, edgecolor='none', zorder=2)

    u_total = sv + uv
    lbl_col = CATEGORY_COLORS[cat] if uv > 0 else C_SHARED
    ax_a.text(u_total + 0.8, y, str(u_total),
              va='center', ha='left', fontsize=8.5, fontweight='bold', color=lbl_col)

    # LEFT (CADD): shared from 0→-sv, exclusive -sv→-(sv+cv)
    if sv > 0:
        ax_a.barh(y, -sv, height=h_bar, left=0,
                  color=C_SHARED, edgecolor='none', zorder=2)
    if cv > 0:
        ax_a.barh(y, -cv, height=h_bar, left=-sv,
                  color=C_CD_EX, edgecolor='#BBBBBB', linewidth=0.3, zorder=2)

    c_total = sv + cv
    if c_total > 0:
        ax_a.text(-(c_total + 0.8), y, str(c_total),
                  va='center', ha='right', fontsize=8.5, fontweight='bold', color='#666666')

ax_a.axvline(0, color='#AAAAAA', lw=0.9, zorder=3)

# Column headers — placed at top of plot, outside the bar area
ax_a.text(-21, n_cats, f'← CADD  (n = {n_c_total})',
          ha='center', va='bottom', fontsize=10, fontweight='bold', color='#666666')
ax_a.text(35, n_cats, f'5ULTRA  (n = {n_u_total},  {ratio:.1f}× CADD) →',
          ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_5U_EX)

ax_a.set_yticks(range(n_cats))
ax_a.set_yticklabels(list(reversed(CAT_LABELS)), fontsize=8.2)
ax_a.set_xlim(x_max_l, x_max_r)
ax_a.set_ylim(-0.55, n_cats + 0.45)
ax_a.set_xlabel('Gene–phenotype associations', fontsize=9, fontweight='bold')
ax_a.spines[['top', 'right']].set_visible(False)
ax_a.tick_params(axis='y', length=0, pad=4)

xticks = [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
ax_a.set_xticks(xticks)
ax_a.set_xticklabels([str(abs(x)) for x in xticks])

# Legend placed below the x-axis
legend_a = [
    mpatches.Patch(facecolor=C_SHARED, edgecolor='none',
                   label=f'Shared by both  (n = {n_shared})'),
    mpatches.Patch(facecolor=C_CD_EX, edgecolor='#BBBBBB', linewidth=0.5,
                   label=f'CADD only  (n = {n_c_ex})'),
    mpatches.Patch(facecolor='#2E86C1', edgecolor='none',
                   label=f'5ULTRA only  (n = {n_u_ex}, colored by trait group)'),
]
ax_a.legend(handles=legend_a, frameon=False, fontsize=9.5,
            loc='lower center', bbox_to_anchor=(0.5, -0.33),
            ncol=3, handlelength=1.2, labelspacing=0.3,
            columnspacing=1.2, borderpad=0.2)

ax_a.text(-0.12, 1.04, 'a', transform=ax_a.transAxes,
          fontsize=13, fontweight='bold', va='top')

# ── PANEL b: 5ULTRA-EXCLUSIVE GENE DISCOVERY LANDSCAPE ──────────────────────
panel_b_df = novel_gene.sort_values('logp', ascending=True).reset_index(drop=True)

for i in range(len(panel_b_df)):
    if i % 2 == 0:
        ax_b.axhspan(i - 0.5, i + 0.5, color='#F7F8F9', zorder=0)

bars_b = ax_b.barh(
    range(len(panel_b_df)), panel_b_df['logp'],
    color=panel_b_df['bar_color'], alpha=0.85,
    edgecolor='none', height=0.68, zorder=2,
)

xmax_b = panel_b_df['logp'].max()
for bar, (_, row) in zip(bars_b, panel_b_df.iterrows()):
    val = row['logp']
    bx  = val + xmax_b * 0.016
    by  = bar.get_y() + bar.get_height() / 2
    ax_b.text(bx, by + 0.14, f'{val:.0f}',
              va='bottom', ha='left', fontsize=7.5, fontweight='bold', color='#1A1A1A')
    if row['trait']:
        ax_b.text(bx, by + 0.02, row['trait'],
                  va='top', ha='left', fontsize=8.0,
                  fontstyle='italic', color='#AAAAAA')

ax_b.set_yticks(range(len(panel_b_df)))
ax_b.set_yticklabels(panel_b_df['gene'], fontsize=9, fontweight='bold')
ax_b.axvline(THRESH_U, color='#555555', ls='--', lw=0.75, alpha=0.45, zorder=1)
ax_b.set_xlabel(r'5ULTRA  ($-\log_{10}\,P$)', fontsize=9, fontweight='bold')
ax_b.set_xlim(0, xmax_b * 1.28)
ax_b.set_ylim(-0.6, len(panel_b_df) - 0.4)
ax_b.spines[['top', 'right']].set_visible(False)
ax_b.tick_params(axis='y', length=0, pad=2)

present_cats = [c for c in CAT_ORDER if c in panel_b_df['category'].values]
legend_b = [
    mpatches.Patch(facecolor=CATEGORY_COLORS[c], edgecolor='none', label=c, alpha=0.85)
    for c in present_cats
]
ax_b.legend(handles=legend_b, frameon=False, fontsize=10,
            loc='lower right', handlelength=1.0, labelspacing=0.25,
            borderpad=0.3)

ax_b.text(-0.30, 1.02, 'b', transform=ax_b.transAxes,
          fontsize=13, fontweight='bold', va='top')

# ── PANEL c: MECHANISTIC ABLATION CASCADE ────────────────────────────────────
x_pos = np.arange(3)
width = 0.62
y_max = n_u_total + 40

ax_c.bar(x_pos[0], n_abl, width=width, color=C_ABL, edgecolor='#CCCCCC',
         linewidth=0.4, zorder=3)
ax_c.text(x_pos[0], n_abl + 1.2, str(n_abl),
          ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#888888')

ax_c.bar(x_pos[1], n_shared, width=width, color=C_SHARED, edgecolor='none', zorder=3)
ax_c.bar(x_pos[1], n_c_ex, bottom=n_shared, width=width,
         color=C_CD_EX, edgecolor='#AAAAAA', linewidth=0.3, zorder=3)
if n_shared >= 6:
    ax_c.text(x_pos[1], n_shared / 2, str(n_shared),
              ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
if n_c_ex >= 4:
    ax_c.text(x_pos[1], n_shared + n_c_ex / 2, f'+{n_c_ex}',
              ha='center', va='center', fontsize=8, color='#2C2C2C', fontweight='bold')
ax_c.text(x_pos[1], n_c_total + 1.2, str(n_c_total),
          ha='center', va='bottom', fontsize=11, fontweight='bold', color='#444444')

ax_c.bar(x_pos[2], n_shared, width=width, color=C_SHARED, edgecolor='none', zorder=3)
ax_c.bar(x_pos[2], n_u_ex, bottom=n_shared, width=width,
         color=C_5U_EX, edgecolor='none', zorder=3)
if n_shared >= 6:
    ax_c.text(x_pos[2], n_shared / 2, str(n_shared),
              ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
ax_c.text(x_pos[2], n_shared + n_u_ex / 2, f'+{n_u_ex}',
          ha='center', va='center', fontsize=13, color='white', fontweight='bold')
ax_c.text(x_pos[2], n_u_total + 1.2, str(n_u_total),
          ha='center', va='bottom', fontsize=12, fontweight='bold', color=C_5U_EX)

ax_c.plot([x_pos[1] + width / 2 + 0.04, x_pos[2] - width / 2 - 0.04],
          [n_shared, n_shared],
          color='#555555', lw=0.65, ls=':', zorder=4)

a1_y = n_c_total + 11
a2_y = n_u_total + 11

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(['CADD\nablated', 'CADD\nbaseline', '5ULTRA'], fontsize=8.2)
ax_c.set_ylabel('Significant gene–phenotype associations',
                fontsize=12, fontweight='bold', linespacing=1.3)
ax_c.set_ylim(0, y_max)
ax_c.set_xlim(-0.55, 2.55)
ax_c.spines[['top', 'right']].set_visible(False)

legend_c = [
    mpatches.Patch(facecolor=C_SHARED, edgecolor='none', label='Shared'),
    mpatches.Patch(facecolor=C_CD_EX, edgecolor='#AAAAAA', linewidth=0.4, label='CADD only'),
    mpatches.Patch(facecolor=C_5U_EX, edgecolor='none', label='5ULTRA only'),
]
ax_c.legend(handles=legend_c, frameon=False, fontsize=10,
            loc='upper left', handlelength=1.0, labelspacing=0.3, borderpad=0.3)

ax_c.text(-0.36, 1.02, 'c', transform=ax_c.transAxes,
          fontsize=13, fontweight='bold', va='top')

# ── SAVE ─────────────────────────────────────────────────────────────────────
for fmt in ('png', 'pdf'):
    out = f'Figure2_Publication_corrected.{fmt}'
    plt.savefig(out, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    print(f'Saved: {out}')
