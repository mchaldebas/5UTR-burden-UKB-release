"""
Generate Supplementary Tables S3 and S4.

S3 - All 58 GWS 5ULTRA gene-phenotype associations
S4 - All 38 GWS CADD gene-phenotype associations

Added as new sheets to a new Excel workbook:
  Supp_Tables_S3_S4.xlsx

Thresholds: Bonferroni over the exact number of tests performed per suite,
imported from analysis_config.py (THRESH_U = 7.016, THRESH_C = 6.331).

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
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from analysis_config import (DATA_DIR, MASTER_CSV, TRUE_K_MIN,
                             THRESH_U, THRESH_C, MODELS_5ULTRA, MODELS_CADD,
                             N_TESTS_5ULTRA, N_TESTS_CADD,
                             verify_test_counts, describe)

OUT_XLSX     = 'Supp_Tables_S3_S4.xlsx'
MODELS_5U = MODELS_5ULTRA
MODELS_CA = MODELS_CADD
print(describe())

# Every model reported in S3/S4 builds its burden as a weighted sum of allele
# dosages (REGENIE --build-mask sum --weights-col 4), so k is a weighted burden
# sum throughout and needs no per-row qualifier. The unweighted Binary-DN model
# used for the exome-PTV comparison (Table S2) is not part of either suite here.

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

HEM_PHENOS = {30000} | set(range(30010, 30310, 10))   # hematology codes (incl. WBC 30000)

# The mask identifies the burden model in the manuscript's vocabulary
# (Materials and Methods, "Burden test design"). Keep these strings identical to
# the model names used there, so a reader can map a row straight onto the text.
MODEL_LABELS = {
    'Mirror_DN':   '5ULTRA DN',
    'Mirror_UP':   '5ULTRA UP',
    '5U_Binary':   '5ULTRA High-confidence',
    '5U_Weighted': '5ULTRA All',
    'Flux_Joint':  '5ULTRA Joint (ACAT)',
    'CD_Binary':   'CADD High-confidence',
    'CD_Weighted': 'CADD All',
}

# Frequency spectra as defined in 02_annotation/01_generate_anno_masks.py
# (freq_cut = 0.001 for 'rare', 0.05 for 'af5').
SPEC_LABELS  = {'rare': 'MAF < 0.1%', 'af5': 'MAF < 5%'}

# ── Colour palette ─────────────────────────────────────────────────────────────
C_HEADER_U   = 'BDD7EE'   # light blue  – 5ULTRA header
C_HEADER_C   = 'D9D9D9'   # light grey  – CADD header
C_ROW_HEM    = 'EAF4FB'   # very light blue – hematology rows
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
verify_test_counts(df)
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
            'Model':         MODEL_LABELS.get(r['mask'], r['mask']),
            'Spectrum':      SPEC_LABELS.get(r['spectrum'], r['spectrum']),
            'Shared':        shared,          # internal flag, not written as column
            'Is_hem':        int(r['pheno']) in HEM_PHENOS,
        })
    return rows

u_rows = build_table(u_gws, c_keys, is_cadd=False)
c_rows = build_table(c_gws, u_keys, is_cadd=True)

# ── Excel writer helper ─────────────────────────────────────────────────────────
def write_sheet(ws, rows, header_color, cross_col_label, title_text,
                thresh_val, n_tests):
    COLS = ['Gene','Phenotype','Pheno code','−log₁₀P','β (SD units)',
            'SE','k','Model','Spectrum', cross_col_label]
    COL_WIDTHS = [12, 24, 12, 12, 14, 10, 9, 24, 14, 16]

    # ── Title row ──
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(COLS))
    title_cell = ws.cell(row=1, column=1, value=title_text)
    title_cell.font      = Font(name='Calibri', bold=True, size=11)
    title_cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
    title_cell.fill      = header_fill(header_color)
    ws.row_dimensions[1].height = 28

    # ── Threshold note ──
    # thresh_val is passed in explicitly: inferring it from title_text was wrong,
    # because the CADD sheet's title also contains the string "5ULTRA".
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(COLS))
    note_cell = ws.cell(row=2, column=1,
        value=f'Genome-wide significance threshold: −log₁₀P > {thresh_val:.3f} '
              f'(Bonferroni correction for the {n_tests:,} association tests '
              f'performed in this suite at k ≥ {TRUE_K_MIN}: 0.05/{n_tests:,}). '
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
        shared = row['Shared']
        is_hem = row['Is_hem']

        if shared:
            fill = row_fill(C_ROW_SHARED)
        elif is_hem:
            fill = row_fill(C_ROW_HEM)
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
            row['Model'],
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
    this_tool  = 'CADD'   if is_cadd else '5ULTRA'
    other_tool = '5ULTRA' if is_cadd else 'CADD'
    cross_col  = f'Also in {other_tool}?'
    legends = [
        ('Colour coding', ''),
        ('  Green rows',  'Gene-phenotype pair significant in BOTH 5ULTRA and CADD'),
        ('  Blue rows',   f'Hematological trait (pheno codes 30000–30300), {this_tool}-only'),
        ('  Yellow rows', f'Biochemistry trait (pheno codes 30600–30890), {this_tool}-only'),
        ('', ''),
        ('Columns', ''),
        ('  Gene',          'HGNC gene symbol'),
        ('  Phenotype',     'UK Biobank trait name'),
        ('  Pheno code',    'UK Biobank field ID'),
        ('  −log₁₀P',      'Best −log₁₀(P-value) across all model × spectrum combinations for this gene-phenotype pair'),
        ('  β (SD units)',  'REGENIE effect size estimate in standard deviation units of the phenotype'),
        ('  SE',            'Standard error of β'),
        ('  k',             'Aggregate weighted allele count for the best model × spectrum: the sum of weighted '
                            'allele dosages across all individuals (REGENIE --build-mask sum), not a raw carrier count'),
        ('  Model',         'Burden model, named as in Materials and Methods ("Burden test design"): '
                            + ('CADD All or CADD High-confidence' if is_cadd else
                               '5ULTRA All, 5ULTRA High-confidence, 5ULTRA UP, 5ULTRA DN or 5ULTRA Joint (ACAT)')),
        ('  Spectrum',      'Allele frequency spectrum: MAF < 0.1% (rare) or MAF < 5% (af5)'),
        (f'  {cross_col}',  f'Yes = this gene-phenotype pair is also genome-wide significant in {other_tool}'),
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
ws3 = wb.create_sheet('S3 - 5ULTRA GWS hits')
n3  = write_sheet(
    ws3, u_rows,
    header_color   = C_HEADER_U,
    cross_col_label= 'Also in CADD?',
    title_text     = (f'Supplementary Table S3. All {len(u_rows)} genome-wide significant 5ULTRA '
                      f'gene-phenotype associations (−log₁₀P > {THRESH_U:.3f}, k ≥ {TRUE_K_MIN}), '
                      f'sorted by −log₁₀P. Green = also significant in CADD; '
                      f'blue = hematology; yellow = biochemistry.'),
    thresh_val     = THRESH_U,
    n_tests        = N_TESTS_5ULTRA,
)

# Legend for S3
ws3_leg = wb.create_sheet('S3 Legend')
write_legend(ws3_leg, is_cadd=False)

# Sheet 2: S4 — CADD GWS hits
ws4 = wb.create_sheet('S4 - CADD GWS hits')
n4  = write_sheet(
    ws4, c_rows,
    header_color   = C_HEADER_C,
    cross_col_label= 'Also in 5ULTRA?',
    title_text     = (f'Supplementary Table S4. All {len(c_rows)} genome-wide significant CADD '
                      f'gene-phenotype associations (−log₁₀P > {THRESH_C:.3f}, k ≥ {TRUE_K_MIN}), '
                      f'sorted by −log₁₀P. Green = also significant in 5ULTRA; '
                      f'blue = hematology; yellow = biochemistry.'),
    thresh_val     = THRESH_C,
    n_tests        = N_TESTS_CADD,
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
          f"β={r['β (SD units)']:+.3f}  k={r['k']:6.1f}  {r['Model']} / {r['Spectrum']}")
print("\nTop 10 CADD hits:")
for r in c_rows[:10]:
    sh = '✓' if r['Shared'] else ' '
    print(f"  {sh} {r['Gene']:12s}  {r['Phenotype']:26s}  logP={r['−log₁₀P']:7.2f}  "
          f"β={r['β (SD units)']:+.3f}  k={r['k']:6.1f}  {r['Model']} / {r['Spectrum']}")
