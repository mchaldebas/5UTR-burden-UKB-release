"""
Generate Supplementary Tables S3 and S4.

S3 — All 58 GWS 5ULTRA gene-phenotype associations
S4 — All 37 GWS CADD gene-phenotype associations

Added as new sheets to a new Excel workbook:
  Supp_Tables_S3_S4.xlsx

Thresholds (Bonferroni-corrected for model selection):
  THRESH_U = 6.0 + log10(8) = 6.903   (4 models × 2 spectra)
  THRESH_C = 6.0 + log10(4) = 6.602   (2 models × 2 spectra)

k filter applied BEFORE deduplication to avoid the order-of-operations bug.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR     = Path(os.environ.get('DATA_DIR', '/Volumes/MCHALDEBAS3/UKB-500k/UKB-data'))
MASTER_CSV   = DATA_DIR / 'Master_Results_Clean.csv.gz'
OUT_XLSX     = 'Supp_Tables_S3_S4.xlsx'
TRUE_K_MIN   = 5
THRESH_U     = 6.0 + np.log10(8)   # 6.903
THRESH_C     = 6.0 + np.log10(4)   # 6.602

MODELS_5U = {'5U_Logic', '5ULTRA_Binary', '5ULTRA_Weighted', 'Flux_Joint'}
MODELS_CA = {'CADD_Binary', 'CADD_Weighted'}

# Models whose burden uses binary (weight=1) weights, so k is a TRUE physical
# carrier count. All other models (Weighted, Logic/Mirror, Flux) report a
# W_PHRED-weighted burden SUM — not a carrier count. See the 'k type' column.
PHYSICAL_MODELS = {'5ULTRA_Binary', 'CADD_Binary', 'Binary_DN'}

PHENO_LABELS = {
    30000:'WBC count',
    30010:'RBC count',          30020:'Haemoglobin',         30030:'Haematocrit',
    30040:'MCV',                30050:'MCH',                 30060:'MCHC',
    30070:'RBC distrib. width', 30080:'Platelet count',      30090:'Platelet crit',
    30100:'Mean platelet vol.', 30110:'Plt distrib. width',  30120:'Lymphocyte count',
    30130:'Monocyte count',     30140:'Neutrophil count',    30150:'Eosinophil count',
    30160:'Basophil count',     30180:'Lymphocyte %',        30190:'Monocyte %',
    30200:'Neutrophil %',       30210:'Eosinophil %',        30220:'Basophil %',
    30240:'Reticulocyte %',     30250:'Reticulocyte count',  30260:'Mean retic. vol.',
    30270:'Mean sph. cell vol.',30280:'Immature retic. fr.', 30290:'High lt. scat. %',
    30300:'High lt. scat. count',
    30600:'Albumin',            30610:'Alk. phosphatase',    30620:'ALT',
    30630:'Apolipoprotein A',   30640:'Apolipoprotein B',    30650:'AST',
    30660:'Direct bilirubin',   30670:'Urea',                30680:'Calcium',
    30690:'Cholesterol',        30700:'Creatinine',          30710:'CRP',
    30720:'Cystatin C',         30730:'GGT',                 30740:'Glucose',
    30750:'HbA1c',              30760:'HDL cholesterol',     30770:'IGF-1',
    30780:'LDL direct',         30790:'Lipoprotein A',       30800:'Oestradiol',
    30810:'Phosphate',          30820:'Rheumatoid factor',        30830:'SHBG',
    30840:'Total bilirubin',    30850:'Testosterone',        30860:'Total protein',
    30870:'Triglycerides',      30880:'Urate',               30890:'Vitamin D',
}

HAEM_PHENOS = {30000} | set(range(30010, 30310, 10))   # haematology codes (incl. WBC 30000)

MODEL_LABELS = {
    '5U_Logic':        'Directional',   # manuscript term for the Mirror UP/DN model
    '5ULTRA_Binary':   'Binary',
    '5ULTRA_Weighted': 'Weighted',
    'Flux_Joint':      'Flux Joint',
    'CADD_Binary':     'Binary',
    'CADD_Weighted':   'Weighted',
}

MASK_LABELS = {
    'Mirror_DN':   "5'UTR-DN",
    'Mirror_UP':   "5'UTR-UP",
    '5U_Binary':   'Binary',
    '5U_Weighted': 'Weighted',
    'Flux_Joint':  'Flux Joint',
    'CD_Binary':   'Binary',
    'CD_Weighted': 'Weighted',
}

SPEC_LABELS  = {'rare': 'AF < 1%', 'af5': 'AF < 5%'}

# ── Colour palette ─────────────────────────────────────────────────────────────
C_HEADER_U   = 'BDD7EE'   # light blue  – 5ULTRA header
C_HEADER_C   = 'D9D9D9'   # light grey  – CADD header
C_ROW_HAEM   = 'EAF4FB'   # very light blue – haematology rows
C_ROW_BIOCH  = 'FEF9E7'   # very light yellow – biochemistry rows
C_ROW_SHARED = 'E8F8E8'   # very light green – shared hits
C_ROW_WHITE  = 'FFFFFF'
C_BORDER     = 'BFBFBF'

def thin_border():
    s = Side(style='thin', color=C_BORDER)
    return Border(left=s, right=s, top=s, bottom=s)

def header_fill(hex_color):
    return PatternFill('solid', fgColor=hex_color)

def row_fill(hex_color):
    return PatternFill('solid', fgColor=hex_color)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading Master_Results_Clean.csv.gz …")
df = pd.read_csv(MASTER_CSV)
df_k = df[df['true_k'] >= TRUE_K_MIN].copy()

# 5ULTRA hits
u_all  = df_k[df_k['model'].isin(MODELS_5U)]
u_best = (u_all.sort_values('logp', ascending=False)
               .drop_duplicates(['gene','pheno']).copy())
u_gws  = u_best[u_best['logp'] > THRESH_U].sort_values('logp', ascending=False).reset_index(drop=True)

# CADD hits
c_all  = df_k[df_k['model'].isin(MODELS_CA)]
c_best = (c_all.sort_values('logp', ascending=False)
               .drop_duplicates(['gene','pheno']).copy())
c_gws  = c_best[c_best['logp'] > THRESH_C].sort_values('logp', ascending=False).reset_index(drop=True)

print(f"5ULTRA GWS hits: {len(u_gws)}")
print(f"CADD   GWS hits: {len(c_gws)}")

# Shared gene-pheno pairs (for cross-reference column)
u_keys = set(zip(u_gws['gene'], u_gws['pheno']))
c_keys = set(zip(c_gws['gene'], c_gws['pheno']))

# ── Build tidy DataFrames ──────────────────────────────────────────────────────
def build_table(gws_df, shared_keys, is_cadd=False):
    rows = []
    for _, r in gws_df.iterrows():
        key = (r['gene'], r['pheno'])
        shared = key in shared_keys
        rows.append({
            'Gene':          r['gene'],
            'Phenotype':     PHENO_LABELS.get(int(r['pheno']), str(r['pheno'])),
            'Pheno code':    int(r['pheno']),
            '−log₁₀P':      round(float(r['logp']), 2),
            'β (SD units)':  round(float(r['beta']), 3),
            'SE':            round(float(r['se']),   3),
            'k':             round(float(r['true_k']), 1),
            'k type':        'carriers' if r['model'] in PHYSICAL_MODELS else 'W_PHRED burden',
            'Model':         MODEL_LABELS.get(r['model'], r['model']),
            'Mask':          MASK_LABELS.get(r['mask'],  r['mask']),
            'Spectrum':      SPEC_LABELS.get(r['spectrum'], r['spectrum']),
            'Shared':        shared,          # internal flag, not written as column
            'Is_haem':       int(r['pheno']) in HAEM_PHENOS,
        })
    return rows

u_rows = build_table(u_gws, c_keys, is_cadd=False)
c_rows = build_table(c_gws, u_keys, is_cadd=True)

# ── Excel writer helper ─────────────────────────────────────────────────────────
def write_sheet(ws, rows, header_color, cross_col_label, title_text):
    COLS = ['Gene','Phenotype','Pheno code','−log₁₀P','β (SD units)',
            'SE','k','k type','Model','Mask','Spectrum', cross_col_label]
    COL_WIDTHS = [12, 24, 12, 12, 14, 10, 9, 15, 14, 14, 12, 16]

    # ── Title row ──
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(COLS))
    title_cell = ws.cell(row=1, column=1, value=title_text)
    title_cell.font      = Font(name='Calibri', bold=True, size=11)
    title_cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
    title_cell.fill      = header_fill(header_color)
    ws.row_dimensions[1].height = 28

    # ── Threshold note ──
    thresh_val = THRESH_U if 'ULTRA' in title_text else THRESH_C
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(COLS))
    note_cell = ws.cell(row=2, column=1,
        value=f'Genome-wide significance threshold: −log₁₀P > {thresh_val:.3f} '
              f'(Bonferroni-corrected for model selection). '
              f'k filter: true_k ≥ {TRUE_K_MIN} (k is a physical carrier count only for '
              f'Binary masks; a W_PHRED-weighted burden sum otherwise — see "k type"). '
              f'Best-performing model × spectrum per gene-phenotype pair shown.')
    note_cell.font      = Font(name='Calibri', italic=True, size=9, color='595959')
    note_cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
    ws.row_dimensions[2].height = 30

    # ── Column headers ──
    hdr_fill = header_fill(header_color)
    for ci, col in enumerate(COLS, start=1):
        cell = ws.cell(row=3, column=ci, value=col)
        cell.font      = Font(name='Calibri', bold=True, size=10)
        cell.fill      = hdr_fill
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border    = thin_border()
    ws.row_dimensions[3].height = 32

    # ── Data rows ──
    for ri, row in enumerate(rows, start=4):
        shared  = row['Shared']
        is_haem = row['Is_haem']

        if shared:
            fill = row_fill(C_ROW_SHARED)
        elif is_haem:
            fill = row_fill(C_ROW_HAEM)
        else:
            fill = row_fill(C_ROW_BIOCH)

        values = [
            row['Gene'],
            row['Phenotype'],
            row['Pheno code'],
            row['−log₁₀P'],
            row['β (SD units)'],
            row['SE'],
            row['k'],
            row['k type'],
            row['Model'],
            row['Mask'],
            row['Spectrum'],
            'Yes' if shared else 'No',
        ]
        for ci, val in enumerate(values, start=1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill      = fill
            cell.border    = thin_border()
            cell.alignment = Alignment(horizontal='center' if ci > 2 else 'left',
                                       vertical='center')
            cell.font      = Font(name='Calibri', size=10,
                                  bold=(ci == 1))   # bold gene name

    # ── Column widths ──
    for ci, w in enumerate(COL_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = w

    # ── Freeze header ──
    ws.freeze_panes = 'A4'

    # ── Auto-filter on header row ──
    ws.auto_filter.ref = f'A3:{get_column_letter(len(COLS))}3'

    return len(rows)

# ── Legend helper ──────────────────────────────────────────────────────────────
def write_legend(ws, is_cadd):
    legends = [
        ('Colour coding', ''),
        ('  Green rows',  'Gene-phenotype pair significant in BOTH 5ULTRA and CADD'),
        ('  Blue rows',   'Haematological trait (pheno codes 30010–30300), 5ULTRA-only or not shared'),
        ('  Yellow rows', 'Biochemistry trait (pheno codes 30600–30880), 5ULTRA-only or not shared'),
        ('', ''),
        ('Columns', ''),
        ('  Gene',          'HGNC gene symbol'),
        ('  Phenotype',     'UK Biobank trait name'),
        ('  Pheno code',    'UK Biobank field ID'),
        ('  −log₁₀P',      'Best −log₁₀(P-value) across all model × spectrum combinations for this gene-phenotype pair'),
        ('  β (SD units)',  'REGENIE effect size estimate in standard deviation units of the phenotype'),
        ('  SE',            'Standard error of β'),
        ('  k',             'Burden magnitude of the best model × spectrum (interpretation depends on "k type")'),
        ('  k type',        'carriers = physical carrier count (Binary masks, weight=1, score-thresholded). '
                            'W_PHRED burden = PHRED-weighted burden sum across all in-mask variants — NOT a '
                            'carrier count (Weighted / Logic / Flux masks).'),
        ('  Model',         '5ULTRA scoring model: Logic (rule-based), Binary (RF binary mask), Weighted (RF score-weighted), Flux Joint' if not is_cadd
                            else 'CADD scoring model: Binary (CADD mask) or Weighted (CADD score-weighted)'),
        ('  Mask',          "Burden mask applied: 5'UTR-DN (repressor class), 5'UTR-UP (enhancer class), Binary, or Weighted"),
        ('  Spectrum',      'Allele frequency spectrum: AF < 1% (rare) or AF < 5% (af5)'),
        ('  Shared',        'Yes = this gene-phenotype pair is also genome-wide significant in the other scorer'),
    ]
    ws.column_dimensions['A'].width = 22
    ws.column_dimensions['B'].width = 80
    ws.cell(row=1, column=1, value='Legend').font = Font(bold=True, size=11)
    for ri, (key, val) in enumerate(legends, start=2):
        ws.cell(row=ri, column=1, value=key).font  = Font(bold=not key.startswith('  '), size=9)
        ws.cell(row=ri, column=2, value=val).font  = Font(size=9)

# ── Build workbook ─────────────────────────────────────────────────────────────
wb = Workbook()
wb.remove(wb.active)   # remove default empty sheet

# Sheet 1: S3 — 5ULTRA GWS hits
ws3 = wb.create_sheet('S3 – 5ULTRA GWS hits')
n3  = write_sheet(
    ws3, u_rows,
    header_color   = C_HEADER_U,
    cross_col_label= 'Also in CADD?',
    title_text     = (f'Supplementary Table S3. All {len(u_rows)} genome-wide significant 5ULTRA '
                      f'gene-phenotype associations (−log₁₀P > {THRESH_U:.3f}, k ≥ {TRUE_K_MIN}), '
                      f'sorted by −log₁₀P. Green = also significant in CADD; '
                      f'blue = haematology; yellow = biochemistry.'),
)

# Legend for S3
ws3_leg = wb.create_sheet('S3 Legend')
write_legend(ws3_leg, is_cadd=False)

# Sheet 2: S4 — CADD GWS hits
ws4 = wb.create_sheet('S4 – CADD GWS hits')
n4  = write_sheet(
    ws4, c_rows,
    header_color   = C_HEADER_C,
    cross_col_label= 'Also in 5ULTRA?',
    title_text     = (f'Supplementary Table S4. All {len(c_rows)} genome-wide significant CADD '
                      f'gene-phenotype associations (−log₁₀P > {THRESH_C:.3f}, k ≥ {TRUE_K_MIN}), '
                      f'sorted by −log₁₀P. Green = also significant in 5ULTRA; '
                      f'blue = haematology; yellow = biochemistry.'),
)

# Legend for S4 (reuse same sheet)
ws4_leg = wb.create_sheet('S4 Legend')
write_legend(ws4_leg, is_cadd=True)

wb.save(OUT_XLSX)
print(f"\nSaved: {OUT_XLSX}")
print(f"  S3: {n3} rows  |  S4: {n4} rows")

# ── Quick sanity print ─────────────────────────────────────────────────────────
shared_gp = u_keys & c_keys
print(f"\nShared gene-pheno pairs: {len(shared_gp)}")
print(f"5ULTRA-only: {len(u_keys - c_keys)}  |  CADD-only: {len(c_keys - u_keys)}")
print("\nTop 10 5ULTRA hits:")
for r in u_rows[:10]:
    sh = '✓' if r['Shared'] else ' '
    print(f"  {sh} {r['Gene']:12s}  {r['Phenotype']:26s}  logP={r['−log₁₀P']:7.2f}  "
          f"β={r['β (SD units)']:+.3f}  k={r['k']:6.1f}({r['k type'][:4]})  {r['Model']}/{r['Mask']}/{r['Spectrum']}")
print("\nTop 10 CADD hits:")
for r in c_rows[:10]:
    sh = '✓' if r['Shared'] else ' '
    print(f"  {sh} {r['Gene']:12s}  {r['Phenotype']:26s}  logP={r['−log₁₀P']:7.2f}  "
          f"β={r['β (SD units)']:+.3f}  k={r['k']:6.1f}({r['k type'][:4]})  {r['Model']}/{r['Mask']}/{r['Spectrum']}")
