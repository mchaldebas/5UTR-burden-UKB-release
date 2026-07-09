"""
102.nature2025_overlap.py  (v7 — final)
-----------------------------------------
Design rationale:
  Panel A  — 5ULTRA signal for each Nature 2025 hit is recomputed LIVE from the
             master CSV via a numeric gene + field_id join (field_id == master
             'pheno'). No phenotype-NAME matching is involved, so the result is
             fully reproducible from Master_Results_Clean.csv.gz. The baked-in
             '-log10P_me' column is read only as an audit cross-check.

  Panels B/C — use the live master CSV only; no Nature 2025 pheno mapping needed.

  k reporting — uses the Weighted model true_k (discrete-ish carrier count)
               wherever available; falls back to Logic model true_k only when
               Weighted model has no result at the same gene-pheno. Merge is
               on ['gene','pheno'] to avoid picking the wrong phenotype row.

  v7 fix — GP1BA k=511 display bug: the Weighted model merge was on gene only,
           accidentally picking a phenotype with k≈0. Now merges on gene+pheno.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

DATA_DIR = Path(os.environ.get('DATA_DIR', '/Volumes/MCHALDEBAS3/UKB-500k/UKB-data'))

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
NATURE_FILE  = DATA_DIR / "Nature_2025_Reproduction_Data.csv"
MASTER_CSV   = DATA_DIR / "Master_Results_Clean.csv.gz"
PHENO_FILE   = DATA_DIR / "pheno_covar_final_clean_all.txt"
TRUE_K_MIN   = 5.0
N_MODELS_5ULTRA = 8    # 4 models × 2 spectra
GWS_ME       = 6.0 + np.log10(N_MODELS_5ULTRA)   # 6.903 — model-selection corrected
GWS_NAT      = 8.0
K_SUSPICIOUS = 1000

PHENO_LABELS = {
    30000: 'WBC count',
    30010: 'RBC count',         30020: 'Haemoglobin',        30030: 'Haematocrit',
    30040: 'MCV',               30050: 'MCH',                30060: 'MCHC',
    30070: 'RBC distrib width', 30080: 'Platelet count',     30090: 'Platelet crit',
    30100: 'Mean platelet vol', 30110: 'Plt distrib width',  30120: 'Lymphocyte count',
    30130: 'Monocyte count',    30140: 'Neutrophil count',   30150: 'Eosinophil count',
    30160: 'Basophil count',    30180: 'Lymphocyte %',       30190: 'Monocyte %',
    30200: 'Neutrophil %',      30210: 'Eosinophil %',       30220: 'Basophil %',
    30240: 'Reticulocyte %',    30250: 'Reticulocyte count', 30260: 'Mean reticuloc vol',
    30270: 'Mean sphered cell vol', 30280: 'Immature retic fr', 30290: 'High lt scat %',
    30300: 'High lt scat count',
    30600: 'Albumin',           30610: 'Alk phosphatase',    30620: 'ALT',
    30630: 'Apolipoprotein A',  30640: 'Apolipoprotein B',   30650: 'AST',
    30660: 'Direct bilirubin',  30670: 'Urea',               30680: 'Calcium',
    30690: 'Cholesterol',       30700: 'Creatinine',         30710: 'CRP',
    30720: 'Cystatin C',        30730: 'GGT',                30740: 'Glucose',
    30750: 'HbA1c',             30760: 'HDL cholesterol',    30770: 'IGF-1',
    30780: 'LDL direct',        30790: 'Lipoprotein A',      30800: 'Oestradiol',
    30810: 'Phosphate',         30820: 'Rheumatoid factor',       30830: 'SHBG',
    30840: 'Total bilirubin',   30850: 'Testosterone',       30860: 'Total protein',
    30870: 'Triglycerides',     30880: 'Urate',              30890: 'Vitamin D',
}
known_phenos = set(PHENO_LABELS.keys())

MODELS_5ULTRA = {'5U_Logic', '5ULTRA_Binary', '5ULTRA_Weighted', 'Flux_Joint'}
MODELS_CADD   = {'CADD_Binary', 'CADD_Weighted'}

# Binary-weight models give a true physical carrier count; all others report a
# W_PHRED-weighted burden SUM (NOT carriers). Used to label the reported k.
PHYSICAL_MODELS = {'5ULTRA_Binary', 'CADD_Binary', 'Binary_DN'}
def k_label(model):
    return 'carriers' if model in PHYSICAL_MODELS else 'W_PHRED_burden'

# ── 1. LOAD MASTER CSV ────────────────────────────────────────────────────────
print("Loading Master_Results_Clean.csv.gz...")
df_all = pd.read_csv(MASTER_CSV)
df_all['gene']  = df_all['gene'].astype(str).str.strip()
df_all['pheno'] = pd.to_numeric(df_all['pheno'], errors='coerce')
df_all['target_group'] = 'Other'
df_all.loc[df_all['model'].isin(MODELS_5ULTRA), 'target_group'] = '5ULTRA'
df_all.loc[df_all['model'].isin(MODELS_CADD),   'target_group'] = 'CADD'

# Apply k-filter BEFORE drop_duplicates so the best k-passing model wins,
# not the highest-logP model that may fail the k filter.
u_all = df_all[df_all['target_group'] == '5ULTRA'].copy()
u_best_all = (u_all.sort_values('logp', ascending=False)
                   .drop_duplicates(['gene', 'pheno']).copy())

u_kpass = u_all[u_all['true_k'] >= TRUE_K_MIN].copy()
u_best_kpass = (u_kpass.sort_values('logp', ascending=False)
                        .drop_duplicates(['gene', 'pheno']).copy())

# GWS hits use k-filtered best; u_best_all kept for missed-hit diagnosis
u_gws = u_best_kpass[u_best_kpass['logp'] > GWS_ME].copy()
u_gws['pheno_label']  = u_gws['pheno'].map(PHENO_LABELS).fillna('UKB-' + u_gws['pheno'].astype(str))
u_gws['suspicious_k'] = u_gws['true_k'] > K_SUSPICIOUS
u_gws['unknown_pheno']= ~u_gws['pheno'].isin(known_phenos)

# k reporting: use true_k from the best-performing model for each gene–phenotype pair.
# No cross-model substitution — the best model's own true_k is the most consistent count.
# (Previous logic substituted Weighted k, which gave fractional k=2.73 for CBFA2T3
#  even though the significant hit came from Binary model with k=68.)
# u_weighted kept as empty placeholder so downstream merge code is harmless.
u_weighted = pd.DataFrame(columns=['gene','pheno','logP_wtd','k_wtd'])

print(f"  5ULTRA GWS hits (true_k ≥ {TRUE_K_MIN}): {len(u_gws)}")

# ── 2. LOAD NATURE 2025 DATA ──────────────────────────────────────────────────
print("\nLoading Nature 2025 data...")
nat_all = pd.read_csv(NATURE_FILE)
nat_all['gene']            = nat_all['gene'].astype(str).str.strip()
nat_all['field_id']        = pd.to_numeric(nat_all['field_id'], errors='coerce')
nat_all['logP_consortium'] = pd.to_numeric(nat_all['-log10P_consortium'], errors='coerce')

# ── LIVE 5ULTRA signal — recomputed from the master CSV (reproducible) ─────────
# Merge each Nature 2025 row to the best 5ULTRA result on gene + numeric
# field_id (== master 'pheno' code). This is a pure numeric join, so the
# phenotype-NAME mismatch that motivated the old pre-computed '-log10P_me'
# workaround no longer applies. The baked-in column is kept only as an audit.
#
# FAIR-COMPARISON FILTERS (replication panel only): Nature 2025's 5'UTR masks
# impose NO minimum-carrier floor, so here we use u_best_all (i.e. we DROP our
# own k≥5 discovery filter for the replication comparison). Our k≥5 filter is
# still applied to the novel / high-gain panels below (u_gws). Replication also
# requires DIRECTION concordance (5ULTRA burden beta sign == Nature 5'UTR beta).
_u_lookup = (u_best_all[['gene', 'pheno', 'logp', 'beta']]
             .rename(columns={'logp': 'logP_me', 'beta': 'beta_me'}))
nat_all = pd.merge(nat_all, _u_lookup,
                   left_on=['gene', 'field_id'], right_on=['gene', 'pheno'],
                   how='left').drop(columns=['pheno'])
nat_all['logP_me']       = nat_all['logP_me'].fillna(0)
nat_all['beta_nat']      = pd.to_numeric(nat_all['beta.5UTR'], errors='coerce')
nat_all['concordant']    = np.sign(nat_all['beta_me']) == np.sign(nat_all['beta_nat'])
nat_all['logP_me_baked'] = pd.to_numeric(nat_all['-log10P_me'], errors='coerce').fillna(0)

nat = nat_all[nat_all['logP_consortium'] >= GWS_NAT].copy()
print(f"  Total rows in CSV: {len(nat_all)}  |  Significant (≥{GWS_NAT}): {len(nat)}")

# Replication thresholds (fair comparison): nominal P<0.05 and Bonferroni for
# the number of Nature genome-wide-significant 5'UTR hits being tested.
REPL_NOMINAL = -np.log10(0.05)               # 1.301
REPL_BONF    = -np.log10(0.05 / len(nat))    # ≈2.92 for 42 hits
print(f"  Replication thresholds: nominal −log10P>{REPL_NOMINAL:.2f}, "
      f"Bonferroni(0.05/{len(nat)}) −log10P>{REPL_BONF:.2f}")

nat['pheno_label'] = nat['field_id'].map(PHENO_LABELS).fillna(
    '(' + nat['phenotype_name'].astype(str).str.split('#').str[-1].str[:20] + ')')

# ── 3. CLASSIFY NATURE 2025 HITS (direction-concordant; Nature-matched) ───────
def classify_repl(row):
    if not bool(row['concordant']):
        return 'Missed'                       # no 5ULTRA signal, or opposite direction
    if row['logP_me'] > REPL_BONF:
        return 'Replicated'                   # Bonferroni — primary
    if row['logP_me'] > REPL_NOMINAL:
        return 'Replicated (nominal only)'    # nominal — sensitivity tier
    return 'Missed'

nat['Class3'] = nat.apply(classify_repl, axis=1)

n_rep_bonf = int((nat['Class3'] == 'Replicated').sum())
n_rep_nom  = int(nat['Class3'].isin(['Replicated', 'Replicated (nominal only)']).sum())
n_concord  = int(nat['concordant'].sum())
print(f"\n--- Replication of {len(nat)} Nature 2025 GWS 5'UTR hits "
      f"(filters matched to Nature; direction-concordant) ---")
print(f"  Bonferroni (−log10P>{REPL_BONF:.2f}) : {n_rep_bonf}/{len(nat)} "
      f"({n_rep_bonf/len(nat)*100:.0f}%)   [primary]")
print(f"  Nominal    (−log10P>{REPL_NOMINAL:.2f}) : {n_rep_nom}/{len(nat)} "
      f"({n_rep_nom/len(nat)*100:.0f}%)   [sensitivity]")
print(f"  Direction concordance overall     : {n_concord}/{len(nat)}")

# ── 4. DIAGNOSE NON-REPLICATED HITS ───────────────────────────────────────────
missed = nat[nat['Class3'] == 'Missed'].copy()
print(f"\n--- Diagnosis of {len(missed)} non-replicated Nature 2025 hits ---")
miss_reasons = {}
for _, r in missed.sort_values('logP_consortium', ascending=False).iterrows():
    gene = r['gene']
    if pd.isna(r['beta_me']) or r['logP_me'] == 0:
        diag = "No 5ULTRA test at this phenotype"
    elif not bool(r['concordant']):
        diag = f"Opposite direction (5ULTRA logP={r['logP_me']:.1f})"
    elif r['logP_me'] > REPL_NOMINAL:
        diag = f"Concordant but sub-Bonferroni (logP={r['logP_me']:.1f})"
    else:
        diag = f"No 5ULTRA signal (logP={r['logP_me']:.1f})"
    miss_reasons[gene] = diag
    print(f"  {gene:12s}  Nat logP={r['logP_consortium']:5.1f}  {diag}")

# ── 5. NOVEL HITS (Panel B) ───────────────────────────────────────────────────
nat_all_genes = set(nat_all['gene'])
novel = u_gws[~u_gws['gene'].isin(nat_all_genes)].drop_duplicates('gene').copy()

# Attach Weighted model k for better carrier reporting (merge on gene+pheno)
novel = pd.merge(novel, u_weighted, on=['gene','pheno'], how='left')
novel['k_report'] = np.where(novel['k_wtd'].notna() & (novel['k_wtd'] >= 1),
                              novel['k_wtd'], novel['true_k'])

novel_clean = novel[~novel['suspicious_k'] & ~novel['unknown_pheno']].copy()
novel_flag  = novel[ novel['suspicious_k'] |  novel['unknown_pheno']].copy()

print(f"\n--- Novel 5ULTRA hits (absent from Nature 2025 entirely) ---")
print(f"  Clean: {len(novel_clean)}  |  Flagged (k>{K_SUSPICIOUS} or unknown pheno): {len(novel_flag)}")
for _, r in novel_clean.sort_values('logp', ascending=False).iterrows():
    tag = 'carr' if k_label(r['model']) == 'carriers' else 'wtd-burden'
    print(f"  {r['gene']:12s}  {r['pheno_label']:25s}  logP={r['logp']:.1f}  "
          f"k={r['k_report']:.0f}({tag})  {r['model']}")
for _, r in novel_flag.sort_values('logp', ascending=False).iterrows():
    reasons = []
    if r['suspicious_k']:   reasons.append(f"k={r['true_k']:.0f}>{K_SUSPICIOUS}")
    if r['unknown_pheno']:  reasons.append(f"unknown pheno {int(r['pheno'])}")
    print(f"  ⚠ {r['gene']:10s}  logP={r['logp']:.1f}  [{', '.join(reasons)}]")

# ── 6. HIGH-GAIN HITS (Panel C) ───────────────────────────────────────────────
# GWS genes that Nature 2025 tested but was sub-threshold for
high_gain = u_gws[u_gws['gene'].isin(nat_all_genes)].copy()

nat_gene_best = (nat_all.groupby('gene')['logP_consortium'].max()
                         .reset_index()
                         .rename(columns={'logP_consortium': 'logP_nat_best'}))
high_gain = pd.merge(high_gain, nat_gene_best, on='gene', how='left')
high_gain['logP_nat_best'] = high_gain['logP_nat_best'].fillna(0)
high_gain = high_gain[high_gain['logP_nat_best'] < GWS_NAT].copy()

# ── KEY FIX (v7): merge Weighted k on gene+pheno, not gene alone ──────────────
high_gain = pd.merge(high_gain, u_weighted, on=['gene','pheno'], how='left')
high_gain['k_report'] = np.where(
    high_gain['k_wtd'].notna() & (high_gain['k_wtd'] >= 1),
    high_gain['k_wtd'],
    high_gain['true_k']
)
high_gain['suspicious_k_c'] = high_gain['k_report'] > K_SUSPICIOUS

high_gain_dedup = (high_gain.sort_values('logp', ascending=False)
                             .drop_duplicates('gene').copy())
clean_hg = high_gain_dedup[~high_gain_dedup['suspicious_k_c'] &
                            ~high_gain_dedup['unknown_pheno']].copy()
susp_hg  = high_gain_dedup[ high_gain_dedup['suspicious_k_c'] |
                              high_gain_dedup['unknown_pheno']].copy()

print(f"\n--- High-gain hits (5ULTRA GWS, Nature 2025 sub-threshold) ---")
print(f"  Clean: {len(clean_hg)}  |  Flagged: {len(susp_hg)}")
for _, r in clean_hg.sort_values('logp', ascending=False).iterrows():
    tag = 'carr' if k_label(r['model']) == 'carriers' else 'wtd-burden'
    print(f"  {r['gene']:12s}  {r['pheno_label']:25s}  logP={r['logp']:.1f}  "
          f"k={r['k_report']:.0f}({tag})  Nat_best={r['logP_nat_best']:.1f}")

# ── 7. SAVE TABLES ────────────────────────────────────────────────────────────
nat[['gene','field_id','pheno_label','logP_consortium','logP_me','Class3']]\
    .to_csv("nature2025_comparison_table.tsv", sep='\t', index=False)

novel_clean['k_type']  = novel_clean['model'].map(k_label)
clean_hg['k_type']     = clean_hg['model'].map(k_label)

novel_clean[['gene','pheno','pheno_label','logp','beta','true_k','k_report','k_type','model']]\
    .sort_values('logp', ascending=False)\
    .to_csv("nature2025_novel_hits.tsv", sep='\t', index=False)

clean_hg[['gene','pheno_label','logp','k_report','k_type','logP_nat_best']]\
    .sort_values('logp', ascending=False)\
    .to_csv("nature2025_highgain_hits.tsv", sep='\t', index=False)

missed[['gene','field_id','logP_consortium','logP_me']]\
    .to_csv("nature2025_missed.tsv", sep='\t', index=False)

print("\nSaved: nature2025_comparison_table.tsv, novel_hits.tsv, highgain_hits.tsv, missed.tsv")

# ── 8. FIGURE ─────────────────────────────────────────────────────────────────
C3_PAL = {
    'Replicated':               '#1A5276',   # Bonferroni, concordant
    'Replicated (nominal only)':'#85C1E9',   # nominal sensitivity tier
    'Missed':                   '#BDC3C7',   # no signal or opposite direction
}

plt.rcParams['font.family'] = 'DejaVu Sans'
fig = plt.figure(figsize=(20, 10), dpi=300)
gs  = gridspec.GridSpec(1, 3, wspace=0.38, width_ratios=[1.3, 1, 1])

# Panel A — replication scatter
ax1 = fig.add_subplot(gs[0])
lim = max(nat['logP_consortium'].max(), nat['logP_me'].max()) * 1.08

for cls, grp in nat.groupby('Class3'):
    ax1.scatter(grp['logP_consortium'], grp['logP_me'],
                color=C3_PAL.get(cls, '#ccc'), s=120,
                edgecolor='black', linewidth=0.6, alpha=0.85,
                label=f"{cls} (n={len(grp)})", zorder=5)

ax1.plot([0, lim], [0, lim], 'k--', alpha=0.2, lw=1.5)
ax1.axhline(REPL_BONF,    color='gray', linestyle=':',  alpha=0.5)
ax1.axhline(REPL_NOMINAL, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(GWS_NAT,      color='gray', linestyle=':',  alpha=0.4)

for _, r in nat[nat['Class3'].str.startswith('Replicated')].iterrows():
    ax1.annotate(r['gene'],
                 xy=(r['logP_consortium'], r['logP_me']),
                 xytext=(4, 2), textcoords='offset points',
                 fontsize=7.5, color='#1A5276', fontweight='bold')

for _, r in nat[nat['Class3'] == 'Missed'].nlargest(5, 'logP_consortium').iterrows():
    ax1.annotate(r['gene'],
                 xy=(r['logP_consortium'], r['logP_me']),
                 xytext=(3, -12), textcoords='offset points',
                 fontsize=8, color='#7F8C8D', style='italic')

ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
ax1.set_xlabel(r"Nature 2025 5′UTR Signal ($-\log_{10}P$)", fontsize=13, fontweight='bold')
ax1.set_ylabel(r"5ULTRA Signal ($-\log_{10}P$)", fontsize=13, fontweight='bold')
ax1.set_title(f"A: 5ULTRA vs Nature 2025 Significant 5′UTR Associations\n"
              f"(n={len(nat)}, Nature logP≥{GWS_NAT}; filters matched to Nature, "
              f"direction-concordant; {n_rep_bonf}/{len(nat)} replicate at Bonferroni, "
              f"{n_rep_nom}/{len(nat)} at nominal)",
              fontweight='bold', loc='left', fontsize=12)
ax1.legend(frameon=False, fontsize=9, loc='upper left')

# Panel B — novel genes
ax2 = fig.add_subplot(gs[1])
novel_plot = novel_clean.sort_values('logp', ascending=False).head(12)
bar_labels = [f"{r['gene']}\n({r['pheno_label']})" for _, r in novel_plot.iterrows()]
ax2.barh(range(len(novel_plot)), novel_plot['logp'],
         color='#E74C3C', edgecolor='black', linewidth=0.6, height=0.7)
ax2.set_yticks(range(len(novel_plot)))
ax2.set_yticklabels(bar_labels, fontsize=9)
ax2.invert_yaxis()
ax2.axvline(GWS_ME, color='black', linestyle='--', alpha=0.4)
ax2.set_xlabel(r"5ULTRA Power ($-\log_{10}P$)", fontsize=13, fontweight='bold')
ax2.set_title(f"B: Novel 5ULTRA Genes\n(absent from Nature 2025; k≤{K_SUSPICIOUS}; known pheno)",
              fontweight='bold', loc='left', fontsize=12)
for i, (_, r) in enumerate(novel_plot.iterrows()):
    ax2.text(r['logp'] + 0.3, i, f"k={r['k_report']:.0f}",
             va='center', fontsize=8, color='#1A5276')

if len(novel_flag):
    ax2.text(0.02, 0.02,
             f"+ {len(novel_flag)} flagged: {', '.join(novel_flag['gene'].head(3))}",
             transform=ax2.transAxes, fontsize=8, color='#922B21', va='bottom',
             style='italic',
             bbox=dict(facecolor='#FDEDEC', edgecolor='#E74C3C',
                       boxstyle='round,pad=0.3', alpha=0.8))

# Panel C — high-gain hits
ax3 = fig.add_subplot(gs[2])
hg_plot = clean_hg.sort_values('logp', ascending=False).head(10)
hg_labels = [f"{r['gene']}\n({r['pheno_label']})" for _, r in hg_plot.iterrows()]

ax3.barh(range(len(hg_plot)), hg_plot['logp'],
         color='#2980B9', edgecolor='black', linewidth=0.6, height=0.7,
         label='5ULTRA')
ax3.barh(range(len(hg_plot)), hg_plot['logP_nat_best'],
         color='#BDC3C7', edgecolor='black', linewidth=0.4, height=0.7,
         alpha=0.7, label='Nature 2025 (best signal for gene)')
ax3.set_yticks(range(len(hg_plot)))
ax3.set_yticklabels(hg_labels, fontsize=9)
ax3.invert_yaxis()
ax3.axvline(GWS_ME, color='black', linestyle='--', alpha=0.4)
ax3.set_xlabel(r"$-\log_{10}P$", fontsize=13, fontweight='bold')
ax3.set_title("C: High-Gain Hits\n(5ULTRA GWS, Nature 2025 sub-threshold for gene)",
              fontweight='bold', loc='left', fontsize=12)
ax3.legend(frameon=False, fontsize=9, loc='lower right')
for i, (_, r) in enumerate(hg_plot.iterrows()):
    ax3.text(r['logp'] + 0.3, i, f"k={r['k_report']:.0f}",
             va='center', fontsize=8, color='#1A5276')

sns.despine()
plt.savefig("Nature2025_Overlap_Figure.png", bbox_inches='tight', dpi=300)
print("Saved: Nature2025_Overlap_Figure.png")

# ── 9. MANUSCRIPT SUMMARY ─────────────────────────────────────────────────────
n_miss = int((nat['Class3'] == 'Missed').sum())

print(f"""
======================================================
MANUSCRIPT SUMMARY — Nature 2025 replication
======================================================
Nature 2025 GWS 5'UTR associations (logP≥{GWS_NAT}): {len(nat)}
Filters matched to Nature (no k≥5 floor); direction-concordant.

Replication (Bonferroni 0.05/{len(nat)}, −log10P>{REPL_BONF:.2f}):  {n_rep_bonf}/{len(nat)} ({n_rep_bonf/len(nat)*100:.0f}%)  [primary]
Replication (nominal P<0.05, −log10P>{REPL_NOMINAL:.2f}):          {n_rep_nom}/{len(nat)} ({n_rep_nom/len(nat)*100:.0f}%)  [sensitivity]
Direction concordance overall:                          {n_concord}/{len(nat)}
Not replicated:                                         {n_miss}/{len(nat)} ({n_miss/len(nat)*100:.0f}%)

Non-replication breakdown (per gene-phenotype pair, sums to {n_miss}):
  No 5ULTRA test at phenotype:  {int((missed['beta_me'].isna() | (missed['logP_me'] == 0)).sum())}
  Opposite direction:           {int(((~missed['concordant']) & ~(missed['beta_me'].isna() | (missed['logP_me'] == 0))).sum())}
  Concordant but no signal:     {int((missed['concordant'] & (missed['logP_me'] <= REPL_NOMINAL)).sum())}

Novel genes (absent from Nature 2025, clean): {len(novel_clean)}
High-gain hits (clean):                        {len(clean_hg)}

Top replicated hits (B=Bonferroni, n=nominal-only):""")
rep_sorted = nat[nat['Class3'].str.startswith('Replicated')]\
    .sort_values('logP_me', ascending=False)
for _, r in rep_sorted.iterrows():
    tier = 'B' if r['Class3'] == 'Replicated' else 'n'
    print(f"  [{tier}] {r['gene']:12s}  {r['pheno_label']:25s}  "
          f"Nat={r['logP_consortium']:.1f}  Us={r['logP_me']:.1f}")
