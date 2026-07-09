"""
Canonical UK Biobank phenotype field-ID -> trait-name mapping.

Single source of truth for the 59 quantitative traits analysed in this study,
so the field labels cannot drift (or be mistyped) across figure scripts.

Notes:
  - Field 30820 is *Rheumatoid factor*, NOT Testosterone (30850). An earlier
    copy of this dict mislabelled 30820 as Testosterone in several figures.
  - Figures that need shorter labels for tight layouts (e.g. forest-plot rows)
    overlay their own abbreviations on top of this dict, e.g.:
        from pheno_labels import PHENO_LABELS as PHENO_CANON
        PHENO_LABELS = {**PHENO_CANON, **PHENO_SHORT}
    so the canonical name is always the fallback and corrections live here only.
"""

PHENO_LABELS = {
    # ── Blood count ───────────────────────────────────────────────────────────
    30000: 'WBC count',
    30010: 'RBC count',          30020: 'Haemoglobin',        30030: 'Haematocrit',
    30040: 'MCV',                30050: 'MCH',                30060: 'MCHC',
    30070: 'RBC distrib width',  30080: 'Platelet count',     30090: 'Platelet crit',
    30100: 'Mean platelet vol',  30110: 'Plt distrib width',  30120: 'Lymphocyte count',
    30130: 'Monocyte count',     30140: 'Neutrophil count',   30150: 'Eosinophil count',
    30160: 'Basophil count',     30180: 'Lymphocyte %',       30190: 'Monocyte %',
    30200: 'Neutrophil %',       30210: 'Eosinophil %',       30220: 'Basophil %',
    30240: 'Reticulocyte %',     30250: 'Reticulocyte count', 30260: 'Mean reticuloc vol',
    30270: 'Mean sphered cell vol', 30280: 'Immature retic fr', 30290: 'High lt scat %',
    30300: 'High lt scat count',
    # ── Biochemistry ──────────────────────────────────────────────────────────
    30600: 'Albumin',            30610: 'Alk phosphatase',    30620: 'ALT',
    30630: 'Apolipoprotein A',   30640: 'Apolipoprotein B',   30650: 'AST',
    30660: 'Direct bilirubin',   30670: 'Urea',               30680: 'Calcium',
    30690: 'Cholesterol',        30700: 'Creatinine',         30710: 'CRP',
    30720: 'Cystatin C',         30730: 'GGT',                30740: 'Glucose',
    30750: 'HbA1c',              30760: 'HDL cholesterol',    30770: 'IGF-1',
    30780: 'LDL direct',         30790: 'Lipoprotein A',      30800: 'Oestradiol',
    30810: 'Phosphate',          30820: 'Rheumatoid factor',  30830: 'SHBG',
    30840: 'Total bilirubin',    30850: 'Testosterone',       30860: 'Total protein',
    30870: 'Triglycerides',      30880: 'Urate',              30890: 'Vitamin D',
}
