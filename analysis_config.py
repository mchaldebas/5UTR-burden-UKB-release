"""
Single source of truth for the significance thresholds used across the paper.

Rationale
---------
Genome-wide significance is set by Bonferroni correction for the EXACT number of
association tests performed in each suite, counted after the k >= TRUE_K_MIN
filter. This replaces the earlier construction (P < 1e-6 divided by the number of
model x spectrum combinations), which divided by the number of MODELS rather than
the number of TESTS and therefore held the two suites to bars differing by ~2.4x.

Test counts (NFE, Master_Results_Clean.csv.gz, true_k >= 5), from
06_analysis/09_threshold_sensitivity.py:

    5ULTRA    518,380 tests   ->  alpha = 0.05/518,380 = 9.645e-08  -log10P > 7.016
    CADD      107,033 tests   ->  alpha = 0.05/107,033 = 4.671e-07  -log10P > 6.331
    Ablation   68,510 tests   ->  alpha = 0.05/ 68,510 = 7.298e-07  -log10P > 6.137

Call verify_test_counts(df) after loading the master table to assert that these
counts still hold; if the pipeline is rerun and the counts move, the thresholds
must move with them.

Usage from a script in 06_analysis/ or 07_figures/:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from analysis_config import (DATA_DIR, TRUE_K_MIN, THRESH_U, THRESH_C,
                                 THRESH_A, MODELS_5ULTRA, MODELS_CADD,
                                 MODELS_ABLATION, verify_test_counts)

Set THRESH_SCHEME=current to reproduce the pre-revision numbers, or
THRESH_SCHEME=pooled for the one-bar-for-both-suites sensitivity analysis.
"""

import os
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
MASTER_CSV = DATA_DIR / "Master_Results_Clean.csv.gz"

TRUE_K_MIN = 5.0
ALPHA = 0.05

MODELS_5ULTRA   = {"5U_Logic", "5ULTRA_Binary", "5ULTRA_Weighted", "Flux_Joint"}
MODELS_CADD     = {"CADD_Binary", "CADD_Weighted"}
MODELS_ABLATION = {"Ablation", "Ablation_Weighted"}

# Realised test counts at true_k >= TRUE_K_MIN (see module docstring).
N_TESTS_5ULTRA   = 518_380
N_TESTS_CADD     = 107_033
N_TESTS_ABLATION = 68_510

SCHEME = os.environ.get("THRESH_SCHEME", "exact").lower()
assert SCHEME in {"current", "exact", "pooled"}, f"bad THRESH_SCHEME: {SCHEME}"

if SCHEME == "current":
    # Pre-revision: 1e-6 base, divided by models x spectra per suite.
    THRESH_U = 6.0 + np.log10(8)   # 6.903
    THRESH_C = 6.0 + np.log10(4)   # 6.602
    THRESH_A = 6.0 + np.log10(4)   # 6.602
elif SCHEME == "exact":
    THRESH_U = -np.log10(ALPHA / N_TESTS_5ULTRA)    # 7.016
    THRESH_C = -np.log10(ALPHA / N_TESTS_CADD)      # 6.331
    THRESH_A = -np.log10(ALPHA / N_TESTS_ABLATION)  # 6.137
else:  # pooled — one identical bar for both suites
    _pooled = -np.log10(ALPHA / (N_TESTS_5ULTRA + N_TESTS_CADD))  # 7.097
    THRESH_U = THRESH_C = THRESH_A = _pooled

# Alpha equivalents, for quoting in table footnotes and legends.
ALPHA_U = 10 ** -THRESH_U
ALPHA_C = 10 ** -THRESH_C
ALPHA_A = 10 ** -THRESH_A


def suite_of(model_series):
    """Map a Series of model names onto suite labels."""
    import pandas as pd
    return pd.Series(
        np.where(model_series.isin(MODELS_5ULTRA), "5ULTRA",
        np.where(model_series.isin(MODELS_CADD), "CADD",
        np.where(model_series.isin(MODELS_ABLATION), "Ablation", "Other"))),
        index=model_series.index,
    )


def verify_test_counts(df, strict=True):
    """Check the realised test counts still match the hard-coded ones.

    df must be the full master table BEFORE the k filter (the filter is applied
    here). Returns the observed counts as a dict.
    """
    kept = df[df["true_k"] >= TRUE_K_MIN]
    obs = {
        "5ULTRA": int(kept["model"].isin(MODELS_5ULTRA).sum()),
        "CADD": int(kept["model"].isin(MODELS_CADD).sum()),
        "Ablation": int(kept["model"].isin(MODELS_ABLATION).sum()),
    }
    exp = {
        "5ULTRA": N_TESTS_5ULTRA,
        "CADD": N_TESTS_CADD,
        "Ablation": N_TESTS_ABLATION,
    }
    bad = {k: (obs[k], exp[k]) for k in exp if obs[k] != exp[k]}
    if bad:
        msg = ("Test counts have changed since the thresholds were fixed: "
               + ", ".join(f"{k} observed {o:,} vs expected {e:,}"
                           for k, (o, e) in bad.items())
               + ". Rerun 06_analysis/09_threshold_sensitivity.py and update "
                 "N_TESTS_* in analysis_config.py.")
        if strict:
            raise AssertionError(msg)
        print("[WARN] " + msg)
    return obs


def describe():
    return (f"scheme={SCHEME}  "
            f"5ULTRA -log10P>{THRESH_U:.3f} (p<{ALPHA_U:.2e})  "
            f"CADD -log10P>{THRESH_C:.3f} (p<{ALPHA_C:.2e})  "
            f"Ablation -log10P>{THRESH_A:.3f} (p<{ALPHA_A:.2e})")


if __name__ == "__main__":
    print(describe())
