import argparse
import pandas as pd
import glob, os, gzip, re
import numpy as np

# --- 1. CONFIGURATION ---
# Cohort-parametrized: NFE results live in results_final_clean/ (manuscript
# layout); non-NFE cohorts in results_final_clean_<cohort>/. Output is one
# Master_Results_Clean[_<cohort>].csv.gz per cohort, each tagged with a 'cohort'
# column so they concatenate cleanly for cross-ancestry comparison.
ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="nfe",
                help="nfe / nonnfe / afr / sas / eas / oth")
ap.add_argument("--input-dir", default=None,
                help="override results dir (default results_final_clean[_<cohort>])")
ap.add_argument("--output", default=None,
                help="override output (default Master_Results_Clean[_<cohort>].csv.gz)")
args = ap.parse_args()

COHORT = args.cohort
INPUT_DIR = args.input_dir or (
    "./results_final_clean" if COHORT == "nfe"
    else f"./results_final_clean_{COHORT}")
OUTPUT_FILE = args.output or (
    "Master_Results_Clean.csv.gz" if COHORT == "nfe"
    else f"Master_Results_Clean_{COHORT}.csv.gz")

def parse_final_results():
    all_data = []
    files = glob.glob(os.path.join(INPUT_DIR, "*.regenie.gz"))
    print(f"Cohort: {COHORT}  |  input: {INPUT_DIR}  |  output: {OUTPUT_FILE}")
    print(f"Found {len(files)} files. Starting Rigorous Physical Carrier Audit...")

    for i, f in enumerate(files):
        filename = os.path.basename(f)

        # --- Phenotype extraction ---
        m = re.search(r'p(\d+)', filename)
        if m is None:
            print(f"  [WARN] Could not extract pheno from: {filename} — skipping.")
            continue
        pheno = m.group(1)

        spectrum = "rare" if "rare" in filename.lower() else "af5"

        # --- Model identification (ordered: most-specific first) ---
        # Must match the --out prefixes in script 82:
        #   chr{CHR}_5U_Mirror_{SPEC}
        #   chr{CHR}_5U_Weighted_{SPEC}
        #   chr{CHR}_5U_Binary_{SPEC}
        #   chr{CHR}_CD_Weighted_{SPEC}
        #   chr{CHR}_CD_Binary_{SPEC}
        #   chr{CHR}_Ablation_{SPEC} / chr{CHR}_Ablation_Weighted_{SPEC}
        #   chr{CHR}_Flux_{SPEC}
        #   chr{CHR}_Binary_DN_Model
        #   chr{CHR}_Combined_Max_{SPEC} / chr{CHR}_Combined_Sum_{SPEC}  (CADD+5ULTRA)
        # ORDER MATTERS: check the more specific names first (e.g. Ablation_Weighted
        # before Ablation, since "Ablation" is a substring of "Ablation_Weighted").
        if   "Binary_DN"        in filename: model = "Binary_DN"         # PTV equivalence control
        elif "Combined_Max"     in filename: model = "Combined_Max"      # max(W_PHRED, CADD)
        elif "Combined_Sum"     in filename: model = "Combined_Sum"      # W_PHRED + CADD
        elif "CD_Binary"        in filename: model = "CADD_Binary"       # CADD binary (physical k)
        elif "CD_Weighted"      in filename: model = "CADD_Weighted"     # CADD continuous weights
        elif "5U_Binary"        in filename: model = "5ULTRA_Binary"     # 5ULTRA binary (physical k)
        elif "5U_Mirror"        in filename: model = "5U_Logic"          # 5ULTRA mirror (continuous)
        elif "5U_Weighted"      in filename: model = "5ULTRA_Weighted"   # 5ULTRA continuous weights
        elif "Ablation_Weighted" in filename: model = "Ablation_Weighted" # CADD-weighted ablation
        elif "Ablation"         in filename: model = "Ablation"          # CADD binary ablation control
        elif "Flux"             in filename: model = "Flux_Joint"        # ACAT joint test
        else:                                model = "Other"

        # --- Physical-k flag ---
        # TRUE only for models that use binary (0/1) weights, so AF*2N == real carrier count.
        # Mirror, Weighted, and Flux use --weights-col 4 (continuous scores) → NOT physical.
        is_physical_k = any(x in filename for x in ["Binary", "Binary_DN"])

        try:
            with gzip.open(f, 'rt') as f_in:
                # Locate header
                cols = None
                for line in f_in:
                    if line.startswith('CHROM'):
                        cols = line.split()
                        break
                if cols is None:
                    print(f"  [WARN] No header found in {filename} — skipping.")
                    continue

                idx_id = cols.index('ID')
                idx_af = cols.index('A1FREQ')
                idx_n  = cols.index('N')
                idx_b  = cols.index('BETA')
                idx_se = cols.index('SE')
                idx_lp = cols.index('LOG10P')

                for line in f_in:
                    parts = line.split()
                    if len(parts) <= idx_lp:
                        continue

                    logp = float(parts[idx_lp])
                    af   = float(parts[idx_af])
                    n    = float(parts[idx_n])
                    k_reported = af * 2 * n

                    id_parts = parts[idx_id].split('.')
                    gene = id_parts[0]
                    mask = id_parts[1] if len(id_parts) > 1 else "Unknown"

                    all_data.append([
                        gene, pheno, model, mask, spectrum, logp,
                        float(parts[idx_b]), float(parts[idx_se]),
                        k_reported, is_physical_k
                    ])

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            continue

    df = pd.DataFrame(
        all_data,
        columns=['gene', 'pheno', 'model', 'mask', 'spectrum',
                 'logp', 'beta', 'se', 'k', 'is_physical']
    )
    df.insert(0, 'cohort', COHORT)

    print(f"\nRaw rows parsed: {len(df):,}")
    print("Model counts:\n", df['model'].value_counts().to_string())

    # --- PHYSICAL CARRIER COUNT (true_k) ---
    # IMPORTANT: true_k is a physical carrier count ONLY for binary-weight masks
    # (is_physical == True: 5U_Binary, CD_Binary, Binary_DN). For weighted/logic
    # masks (Weighted, Mirror, Flux) 'k' = A1FREQ*2N is a W_PHRED-WEIGHTED burden
    # SUM, not a carrier count. The binary masks use DIFFERENT variant sets /
    # score thresholds than the weighted masks (e.g. 5U_Binary is score≥0.5 only,
    # 5U_Weighted is all variants), so a binary mask's k cannot be substituted as
    # the carrier count of a different mask. We therefore map physical k only
    # within the SAME (gene, mask, spectrum); weighted masks keep their own k.
    # ==> Always gate "carrier count" interpretation on the is_physical column.
    physical_map = (
        df[df['is_physical']]
        .groupby(['gene', 'mask', 'spectrum'])['k']
        .max()
        .reset_index()
        .rename(columns={'k': 'true_k'})
    )
    df = pd.merge(df, physical_map, on=['gene', 'mask', 'spectrum'], how='left')
    df['true_k'] = df['true_k'].fillna(df['k'])   # weighted masks: == weighted burden sum

    df['z'] = df['beta'] / df['se']

    n_missing = df['true_k'].isna().sum()
    if n_missing:
        print(f"  [WARN] {n_missing} rows have no true_k — check binary model coverage.")

    return df


df_master = parse_final_results()
df_master.to_csv(OUTPUT_FILE, index=False, compression='gzip')
print(f"\nSUCCESS: {OUTPUT_FILE} written ({len(df_master):,} rows). Rigor check complete.")
