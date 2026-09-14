"""
rheostat_classify.py
-------------------------------------------------------------------------------
Significance-based classification of UP/DN gene-phenotype pairs.

Replaces the old sign-only rule (`sign(beta_DN) != sign(beta_UP)`), which called
a pair a "rheostat" whenever the two coefficients happened to point in opposite
directions, regardless of whether either arm was individually significant.

The categories are defined on the JOINT refit from
06_analysis/06_rheostat_beta_equality.py (y ~ covars + burden_UP + burden_DN),
which supplies beta_UP/p_UP, beta_DN/p_DN and two Wald contrasts:

    p_symmetry : beta_UP + beta_DN = 0   i.e.  beta_DN = -beta_UP  (mirror)
    p_equality : beta_UP - beta_DN = 0   i.e.  beta_DN =  beta_UP  (identical)

Categories (alpha = 0.05 on each arm):

  rheostat    opposite sign AND both arms significant.
              sub-flag: asymmetric if the SYMMETRY contrast is rejected
              (beta_DN = -beta_UP does not hold), symmetric if it is not.
  DN-driven   only the DN arm significant. Sign is not part of the definition.
              sub-flag: whether the EQUALITY contrast is rejected, i.e. whether
              the silent arm is distinguishable from the driving one.
  UP-driven   only the UP arm significant. Sign is not part of the definition.
              sub-flag: as for DN-driven.
  fragility   same sign AND both arms significant.
              sub-flag: whether the EQUALITY contrast is rejected, i.e. whether
              the two arms are of the same order of magnitude.
  class-nonspecific   neither arm significant (association is not specific to
              the UP or DN class).

INTERPRETATION NOTE. Rejecting beta_UP = beta_DN means only that the two
coefficients DIFFER; it does not mean they OPPOSE. A pair in which only one arm
is significant must never be described as showing an "opposing effect" on the
strength of a rejected equality test. Opposition requires opposite sign AND both
arms individually significant -- that is the rheostat category and nothing else.
"""

import numpy as np
import pandas as pd

ALPHA = 0.05

# Top-level categories, in the order they should appear in tables and figures.
CATEGORY_ORDER = ['rheostat', 'UP-driven', 'DN-driven', 'fragility',
                  'class-nonspecific', 'untested']

CATEGORY_COLORS = {
    'rheostat':  '#16A085',   # teal
    'UP-driven': '#C0392B',   # red   (matches the UP mask colour)
    'DN-driven': '#2E86C1',   # blue  (matches the DN mask colour)
    'fragility': '#7F8C8D',   # grey
    'class-nonspecific': '#BFC9D1',   # pale grey
    'untested':  '#E8ECF0',   # background
}

CATEGORY_MARKERS = {
    'rheostat':  'o',
    'UP-driven': '^',
    'DN-driven': 'v',
    'fragility': 's',
    'class-nonspecific': 'X',
    'untested':  '.',
}

CATEGORY_LABELS = {
    'rheostat':  'Rheostat (opposite sign, both arms P < 0.05)',
    'UP-driven': 'UP-driven (only UP arm P < 0.05)',
    'DN-driven': 'DN-driven (only DN arm P < 0.05)',
    'fragility': 'Genetic fragility (same sign, both arms P < 0.05)',
    'class-nonspecific': 'Class-nonspecific (UP and DN P > 0.05)',
    'untested':  'Not refit (no beta-equality test)',
}


def classify_row(beta_UP, beta_DN, p_UP, p_DN, p_symmetry=np.nan,
                 p_equality=np.nan, alpha=ALPHA):
    """Return (category, detail) for one gene-phenotype pair.

    `detail` carries the sub-flag where the scheme defines one, and repeats the
    category otherwise. Any missing coefficient/p-value yields 'untested' rather
    than a silent drop.
    """
    vals = [beta_UP, beta_DN, p_UP, p_DN]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        return 'untested', 'untested'

    sig_UP = p_UP < alpha
    sig_DN = p_DN < alpha
    opposite = (beta_UP * beta_DN) < 0

    if sig_UP and sig_DN:
        if opposite:
            # Sub-flag on the mirror contrast beta_DN = -beta_UP.
            if np.isnan(p_symmetry):
                return 'rheostat', 'rheostat (symmetry untested)'
            return ('rheostat', 'rheostat, asymmetric') if p_symmetry < alpha \
                else ('rheostat', 'rheostat, symmetric')
        # Same sign, both significant: genetic fragility. Note whether the two
        # arms are of the same order of magnitude (equality contrast).
        if np.isnan(p_equality):
            return 'fragility', 'fragility (magnitude untested)'
        return ('fragility', 'fragility, differing magnitude') if p_equality < alpha \
            else ('fragility', 'fragility, similar magnitude')

    # Exactly one arm significant. The category is set by WHICH arm drives the
    # signal; the equality contrast is reported alongside it as a sub-flag, so a
    # single-arm result carries the same beta_UP = beta_DN annotation the
    # two-arm categories do. Rejecting it means the silent arm is distinguishable
    # from the driving one -- NOT that the two effects oppose.
    if sig_DN or sig_UP:
        cat = 'DN-driven' if sig_DN else 'UP-driven'
        if np.isnan(p_equality):
            return cat, f'{cat} (equality untested)'
        return (cat, f'{cat}, betas differ') if p_equality < alpha \
            else (cat, f'{cat}, betas not distinguishable')
    return 'class-nonspecific', 'class-nonspecific'


def classify(df, alpha=ALPHA, beta_UP='beta_UP', beta_DN='beta_DN',
             p_UP='p_UP', p_DN='p_DN', p_symmetry='p_symmetry',
             p_equality='p_equality'):
    """Add 'category' and 'category_detail' columns to a copy of `df`.

    Column names default to the 06_rheostat_beta_equality.py output; pass
    overrides when the frame carries suffixed copies (e.g. beta_UP_eq).
    """
    out = df.copy()

    def _get(row, col):
        return row[col] if col in out.columns else np.nan

    pairs = [classify_row(_get(r, beta_UP), _get(r, beta_DN),
                          _get(r, p_UP), _get(r, p_DN),
                          _get(r, p_symmetry), _get(r, p_equality), alpha)
             for _, r in out.iterrows()]
    out['category'] = [p[0] for p in pairs]
    out['category_detail'] = [p[1] for p in pairs]
    return out


def count_table(df, category_col='category', detail_col='category_detail'):
    """Category / sub-flag counts, in CATEGORY_ORDER, as a tidy frame."""
    rows = []
    for cat in CATEGORY_ORDER:
        sub = df[df[category_col] == cat]
        if sub.empty:
            rows.append({'category': cat, 'category_detail': '', 'n': 0})
            continue
        for detail, n in sub[detail_col].value_counts().items():
            rows.append({'category': cat, 'category_detail': detail, 'n': int(n)})
    return pd.DataFrame(rows)
