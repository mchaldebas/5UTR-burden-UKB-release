"""
07_cross_ancestry_replication.py
--------------------------------
Replication of the NFE-discovered 5ULTRA hits in the non-NFE ancestry cohorts.

Discovery set (NFE):
    models  = {5U_Logic, 5ULTRA_Binary, 5ULTRA_Weighted, Flux_Joint}
    filter  = true_k >= 5
    best model per (gene, pheno) by logp
    GWS     = logp > 6.0 + log10(8) = 6.903   (model-selection corrected)

For every discovery hit we pull the SAME (gene, pheno, model, spectrum) row from
each replication cohort and evaluate three replication rules, so the criterion
can be chosen post-hoc from the printed summary:

    sign        : effect direction concordant with NFE
    nominal     : concordant AND p_rep < 0.05
    bonferroni  : concordant AND p_rep < 0.05 / n_hits

Replication cohorts: afr, sas, eas, oth (each separately) + nonnfe (pooled).

Outputs (written next to this script):
    cross_ancestry_replication_perhit.csv   one row per (hit x cohort)
    cross_ancestry_replication_summary.csv  replication rates per cohort x rule
    cross_ancestry_replication.png/.pdf     status heatmap + summary bars
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# House style — matches 07_figures/figure*.py (the paper figures).
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#2C2C2C",
    "xtick.color": "#2C2C2C",
    "ytick.color": "#2C2C2C",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/Volumes/Dropbox Chaldeb/UKB-burden"))
OUT_DIR = Path(__file__).resolve().parent

MODELS_5U = {"5U_Logic", "5ULTRA_Binary", "5ULTRA_Weighted", "Flux_Joint"}
TRUE_K_MIN = 5.0
N_MODELS = 8  # 4 models x 2 spectra
GWS = 6.0 + np.log10(N_MODELS)  # 6.903

REP_COHORTS = ["afr", "sas", "eas", "oth", "nonnfe"]
ANCESTRY_COHORTS = ["afr", "sas", "eas", "oth"]  # distinct ancestries (exclude pooled)

# A hit is "testable" in a cohort iff a true_k>=5 burden result exists for the
# same gene-pheno-model-mask-spectrum (manuscript convention; the true_k>=5
# filter also removes REGENIE's ultra-rare singleton-AAF-bin artifacts that the
# aggregator collapses onto the same mask label).


def master_path(cohort):
    return (
        DATA_DIR / "Master_Results_Clean.csv.gz"
        if cohort == "nfe"
        else DATA_DIR / f"Master_Results_Clean_{cohort}.csv.gz"
    )


# ── Discovery set (NFE) ─────────────────────────────────────────────────────
print(f"Loading NFE master from {master_path('nfe')} …")
nfe = pd.read_csv(master_path("nfe"))
nfe_k = nfe[nfe["true_k"] >= TRUE_K_MIN]
u = (
    nfe_k[nfe_k["model"].isin(MODELS_5U)]
    .sort_values("logp", ascending=False)
    .drop_duplicates(["gene", "pheno"])
    .copy()
)
hits = u[u["logp"] > GWS].sort_values("logp", ascending=False).reset_index(drop=True)
n_hits = len(hits)
print(f"NFE 5ULTRA GWS hits: {n_hits}  (threshold logp>{GWS:.3f}, true_k>={TRUE_K_MIN:.0f})")

disc = hits[["gene", "pheno", "model", "mask", "spectrum", "logp", "beta", "se", "true_k", "z"]].rename(
    columns={
        "logp": "logp_nfe",
        "beta": "beta_nfe",
        "se": "se_nfe",
        "true_k": "true_k_nfe",
        "z": "z_nfe",
    }
)
BONF_P = 0.05 / max(n_hits, 1)

# ── Evaluate replication in each cohort ─────────────────────────────────────
records = []
for cohort in REP_COHORTS:
    fp = master_path(cohort)
    if not fp.exists():
        print(f"  [WARN] {fp.name} missing — skipping {cohort}.")
        continue
    print(f"Loading {cohort} master …")
    rep = pd.read_csv(fp)
    rep_sub = rep[["gene", "pheno", "model", "mask", "spectrum", "logp", "beta", "se", "true_k"]].rename(
        columns={
            "logp": "logp_rep",
            "beta": "beta_rep",
            "se": "se_rep",
            "true_k": "true_k_rep",
        }
    )
    # MANUSCRIPT CONVENTION. The aggregator collapses REGENIE's two AAF bins
    # (GENE.MASK.singleton / GENE.MASK.0.01) onto one 'mask' label, so each
    # (gene,pheno,model,mask,spectrum) key carries TWO rows. Reproduce the paper:
    # filter true_k>=5 (drops the ultra-rare singleton-bin artifacts), then take
    # the most significant remaining row per key — exactly what Supp Table S3 does
    # on the NFE discovery side.
    key = ["gene", "pheno", "model", "mask", "spectrum"]
    rep_sub = rep_sub[rep_sub["true_k_rep"] >= TRUE_K_MIN]
    rep_sub = rep_sub.sort_values("logp_rep", ascending=False).drop_duplicates(key)
    m = disc.merge(rep_sub, on=key, how="left")
    assert len(m) == len(disc), f"{cohort}: merge fanned out ({len(m)} != {len(disc)})"
    m["cohort"] = cohort

    # testable: a true_k>=5 burden result exists for this hit in the cohort
    m["tested"] = m["beta_rep"].notna() & m["se_rep"].notna()
    m["p_rep"] = np.power(10.0, -m["logp_rep"])
    m["concordant"] = np.sign(m["beta_rep"]) == np.sign(m["beta_nfe"])
    m["concordant"] = m["concordant"] & m["tested"]

    m["rep_sign"] = m["tested"] & m["concordant"]
    m["rep_nominal"] = m["rep_sign"] & (m["p_rep"] < 0.05)
    m["rep_bonf"] = m["rep_sign"] & (m["p_rep"] < BONF_P)
    records.append(m)

perhit = pd.concat(records, ignore_index=True)
perhit_cols = [
    "gene", "pheno", "model", "mask", "spectrum", "cohort",
    "logp_nfe", "beta_nfe", "z_nfe", "true_k_nfe",
    "logp_rep", "beta_rep", "se_rep", "true_k_rep", "p_rep",
    "tested", "concordant", "rep_sign", "rep_nominal", "rep_bonf",
]
perhit = perhit[perhit_cols]
perhit_out = OUT_DIR / "cross_ancestry_replication_perhit.csv"
perhit.to_csv(perhit_out, index=False)

# ── Per-cohort summary ──────────────────────────────────────────────────────
rows = []
for cohort in REP_COHORTS:
    c = perhit[perhit["cohort"] == cohort]
    if c.empty:
        continue
    n_tested = int(c["tested"].sum())
    for rule, col in [("sign", "rep_sign"), ("nominal", "rep_nominal"), ("bonferroni", "rep_bonf")]:
        n_rep = int(c[col].sum())
        rows.append(
            {
                "cohort": cohort,
                "rule": rule,
                "n_hits": n_hits,
                "n_tested": n_tested,
                "n_replicated": n_rep,
                "rate_of_tested": n_rep / n_tested if n_tested else np.nan,
                "rate_of_all": n_rep / n_hits if n_hits else np.nan,
            }
        )

# "replicated in >=1 ancestry cohort" (afr/sas/eas/oth, pooled nonnfe excluded)
anc = perhit[perhit["cohort"].isin(ANCESTRY_COHORTS)]
for rule, col in [("sign", "rep_sign"), ("nominal", "rep_nominal"), ("bonferroni", "rep_bonf")]:
    any_rep = anc.groupby(["gene", "pheno"])[col].any()
    tested_any = anc.groupby(["gene", "pheno"])["tested"].any()
    rows.append(
        {
            "cohort": "any_ancestry(>=1)",
            "rule": rule,
            "n_hits": n_hits,
            "n_tested": int(tested_any.sum()),
            "n_replicated": int(any_rep.sum()),
            "rate_of_tested": any_rep.sum() / tested_any.sum() if tested_any.sum() else np.nan,
            "rate_of_all": any_rep.sum() / n_hits if n_hits else np.nan,
        }
    )

summary = pd.DataFrame(rows)
summary_out = OUT_DIR / "cross_ancestry_replication_summary.csv"
summary.to_csv(summary_out, index=False)

print("\n=== Replication summary (Bonferroni p < {:.2e}) ===".format(BONF_P))
with pd.option_context("display.width", 200, "display.max_rows", None):
    print(
        summary.assign(
            rate_of_tested=lambda d: (100 * d["rate_of_tested"]).round(1),
            rate_of_all=lambda d: (100 * d["rate_of_all"]).round(1),
        ).to_string(index=False)
    )

# ── FIGURE (house style; addresses the "NFE-only" Discussion limitation) ─────
perhit["z_rep"] = perhit["beta_rep"] / perhit["se_rep"]
C_CONC = "#2E86C1"   # concordant — house hematology blue
C_DISC = "#E67E22"   # discordant — house lipid orange

COH_LABEL = {"afr": "AFR", "sas": "SAS", "eas": "EAS", "oth": "OTH",
             "nonnfe": "non-NFE\n(pooled)", "any_ancestry(>=1)": "Any ancestry\n(≥1)"}
COH_COLOR = {"afr": "#8E44AD", "sas": "#229954", "eas": "#B7950B",
             "oth": "#7F8C8D", "nonnfe": "#2E86C1", "any_ancestry(>=1)": "#16A085"}

fig = plt.figure(figsize=(7.3, 3.6), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.1], wspace=0.34,
                      left=0.085, right=0.975, top=0.88, bottom=0.16)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ── Panel a: effect concordance in the pooled non-European cohort ────────────
pool = perhit[(perhit["cohort"] == "nonnfe") & (perhit["tested"])].copy()
zx, zy = pool["z_nfe"].values, pool["z_rep"].values
conc = np.sign(zx) == np.sign(zy)
lim = np.nanmax(np.abs(np.concatenate([zx, zy]))) * 1.12
ax_a.axhspan(0, lim, xmin=0.5, color="#EAF2FB", zorder=0)            # Q1 concordant
ax_a.axhspan(-lim, 0, xmin=0, xmax=0.5, color="#EAF2FB", zorder=0)  # Q3 concordant
ax_a.axhline(0, color="#AAAAAA", lw=0.7, zorder=1)
ax_a.axvline(0, color="#AAAAAA", lw=0.7, zorder=1)
ax_a.scatter(zx[conc], zy[conc], s=26, color=C_CONC, edgecolor="white",
             linewidth=0.4, zorder=3, label=f"concordant ({int(conc.sum())})")
ax_a.scatter(zx[~conc], zy[~conc], s=26, color=C_DISC, edgecolor="white",
             linewidth=0.4, zorder=3, label=f"discordant ({int((~conc).sum())})")
for _, r in pool.sort_values("logp_nfe", ascending=False).head(6).iterrows():
    ax_a.annotate(r["gene"], (r["z_nfe"], r["z_rep"]), fontsize=6.5,
                  fontstyle="italic", color="#333333",
                  xytext=(3, 3), textcoords="offset points")
if len(zx) > 2:
    r_pear, p_pear = stats.pearsonr(zx, zy)
else:
    r_pear, p_pear = np.nan, np.nan
p_str = (f"P = {p_pear:.1e}" if np.isfinite(p_pear) and p_pear < 1e-3
         else f"P = {p_pear:.3f}" if np.isfinite(p_pear) else "P = n/a")
ax_a.set_xlim(-lim, lim); ax_a.set_ylim(-lim, lim)
ax_a.set_xlabel(r"Discovery (NFE)  $Z$", fontsize=9, fontweight="bold")
ax_a.set_ylabel(r"Replication (non-NFE)  $Z$", fontsize=9, fontweight="bold")
ax_a.set_title("Effect concordance, pooled non-European", fontsize=8.5)
ax_a.text(0.04, 0.97,
          f"{100*conc.mean():.0f}% directional\nr = {r_pear:.2f}, {p_str}  (n = {len(zx)})",
          transform=ax_a.transAxes, fontsize=7.3, va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.5))
ax_a.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.3)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.text(-0.15, 1.07, "a", transform=ax_a.transAxes, fontsize=13,
          fontweight="bold", va="top")

# ── Panel b: replication rate by cohort and rule ─────────────────────────────
RULES = ["sign", "nominal", "bonferroni"]
RULE_LABEL = {"sign": "concordant sign", "nominal": "+ nominal P<0.05", "bonferroni": "+ Bonferroni"}
RULE_ALPHA = {"sign": 0.42, "nominal": 0.70, "bonferroni": 1.0}
order = ["afr", "nonnfe", "any_ancestry(>=1)"]
yb = np.arange(len(order))[::-1]
hh = 0.26
for j, rule in enumerate(RULES):
    for i, coh in enumerate(order):
        row = summary[(summary["cohort"] == coh) & (summary["rule"] == rule)]
        rate = float(row["rate_of_tested"].iloc[0]) if len(row) and pd.notna(row["rate_of_tested"].iloc[0]) else 0.0
        ax_b.barh(yb[i] + (1 - j) * hh, 100 * rate, height=hh,
                  color=COH_COLOR[coh], alpha=RULE_ALPHA[rule],
                  edgecolor="white", linewidth=0.3, zorder=3)
for i, coh in enumerate(order):
    nt = int(summary[(summary["cohort"] == coh) & (summary["rule"] == "sign")]["n_tested"].iloc[0])
    ax_b.text(101, yb[i], f"n={nt}", va="center", ha="left", fontsize=6.5, color="#777777")
ax_b.set_yticks(yb)
ax_b.set_yticklabels([COH_LABEL[c] for c in order], fontsize=7.8)
ax_b.set_xlim(0, 100); ax_b.set_xlabel("Replication rate, % of tested", fontsize=9, fontweight="bold")
ax_b.set_title("Replication by cohort and stringency", fontsize=8.5)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.tick_params(axis="y", length=0)
rule_handles = [Patch(facecolor="#555555", alpha=RULE_ALPHA[r], label=RULE_LABEL[r]) for r in RULES]
ax_b.legend(handles=rule_handles, frameon=False, fontsize=6.8, loc="lower right",
            handlelength=1.1, labelspacing=0.25)
ax_b.text(-0.18, 1.07, "b", transform=ax_b.transAxes, fontsize=13,
          fontweight="bold", va="top")

for fmt in ("png", "pdf"):
    out = OUT_DIR / f"Figure_CrossAncestry_Replication.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=300 if fmt == "png" else None)
    print(f"Saved: {out}")
print(f"\nWrote CSVs:\n  {perhit_out}\n  {summary_out}")
