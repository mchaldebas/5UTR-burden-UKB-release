"""
08_combined_model_comparison.py
-------------------------------
Combined CADD + 5ULTRA weighted models vs the single-source weighted models
(NFE cohort). Combined weights (02_generate_combined_anno.py):

    Combined_Max  w = max(W_PHRED, CADD)
    Combined_Sum  w = W_PHRED + CADD

evaluated only over variants carrying BOTH a 5ULTRA and a CADD weight.

Question: does fusing CADD + 5ULTRA recover associations that neither source
finds alone, and does it preserve the 5ULTRA-unique signal?

Discovery unit = (gene, pheno), best spectrum per model by logp, true_k >= 5.
Counts are reported across a threshold grid so the operating point can be chosen
post-hoc; the set-overlap / gain-loss analysis uses the primary threshold
6.0 + log10(8) = 6.903 (same model-selection-corrected line as the 5ULTRA hits).

Outputs (next to this script):
    combined_model_counts.csv      GWS discovery counts per model x threshold
    combined_model_overlap.csv     gained / retained / lost pairs vs single-source
    combined_model_perpair.csv     per (gene,pheno): logp of each model
    combined_model_comparison.png/.pdf
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

DATA_DIR = Path(os.environ.get("DATA_DIR", "/Volumes/Dropbox Chaldeb/UKB-burden"))
OUT_DIR = Path(__file__).resolve().parent
MASTER = DATA_DIR / "Master_Results_Clean.csv.gz"

TRUE_K_MIN = 5.0
PRIMARY = 6.0 + np.log10(8)  # 6.903
GRID = [6.0, 6.602, 6.903, 7.3]

SINGLE = ["5ULTRA_Weighted", "CADD_Weighted"]
COMBINED = ["Combined_Max", "Combined_Sum"]
MODELS = SINGLE + COMBINED + ["Ablation_Weighted"]


def best_per_pair(df, model):
    """Best (max-logp) row per (gene,pheno) for one model, true_k filtered."""
    sub = df[(df["model"] == model) & (df["true_k"] >= TRUE_K_MIN)]
    return (
        sub.sort_values("logp", ascending=False)
        .drop_duplicates(["gene", "pheno"])
        .set_index(["gene", "pheno"])
    )


print(f"Loading NFE master {MASTER} …")
df = pd.read_csv(MASTER)
print(f"  rows: {len(df):,}")

best = {m: best_per_pair(df, m) for m in MODELS}
for m in MODELS:
    print(f"  {m:18s} pairs with true_k>={TRUE_K_MIN:.0f}: {len(best[m]):,}")

# ── Discovery counts across threshold grid ──────────────────────────────────
count_rows = []
for m in MODELS:
    lp = best[m]["logp"]
    for t in GRID:
        count_rows.append({"model": m, "threshold": round(t, 3), "n_gws": int((lp > t).sum())})
counts = pd.DataFrame(count_rows)
counts_pivot = counts.pivot(index="model", columns="threshold", values="n_gws").reindex(MODELS)
counts.to_csv(OUT_DIR / "combined_model_counts.csv", index=False)
print("\n=== GWS (gene,pheno) discoveries per model x threshold ===")
print(counts_pivot.to_string())

# ── Gain/loss vs single-source union (primary threshold) ────────────────────
def gws_set(model):
    s = best[model]["logp"]
    return set(s[s > PRIMARY].index)


set_5u = gws_set("5ULTRA_Weighted")
set_cd = gws_set("CADD_Weighted")
union_single = set_5u | set_cd
u5_unique = set_5u - set_cd  # 5ULTRA-only signal

overlap_rows = []
for m in COMBINED:
    s_cmb = gws_set(m)
    gained = s_cmb - union_single
    lost = union_single - s_cmb
    retained_5u_unique = u5_unique & s_cmb
    overlap_rows.append(
        {
            "model": m,
            "n_gws": len(s_cmb),
            "n_single_union": len(union_single),
            "gained_vs_union": len(gained),
            "lost_vs_union": len(lost),
            "u5_unique_total": len(u5_unique),
            "u5_unique_retained": len(retained_5u_unique),
            "gained_pairs": ";".join(f"{g}:{p}" for g, p in sorted(gained)),
            "lost_pairs": ";".join(f"{g}:{p}" for g, p in sorted(lost)),
        }
    )
overlap = pd.DataFrame(overlap_rows)
overlap.to_csv(OUT_DIR / "combined_model_overlap.csv", index=False)
print(f"\n=== Gain/loss vs (5ULTRA ∪ CADD) at logp>{PRIMARY:.3f} ===")
print(f"single-source union: {len(union_single)}   (5ULTRA-unique: {len(u5_unique)})")
print(
    overlap[
        ["model", "n_gws", "gained_vs_union", "lost_vs_union", "u5_unique_retained", "u5_unique_total"]
    ].to_string(index=False)
)

# ── Per-pair logp matrix across models (union of any-model GWS pairs) ────────
all_gws_pairs = set()
for m in MODELS:
    s = best[m]["logp"]
    all_gws_pairs |= set(s[s > PRIMARY].index)
perpair = pd.DataFrame(index=pd.MultiIndex.from_tuples(sorted(all_gws_pairs), names=["gene", "pheno"]))
for m in MODELS:
    perpair[m] = best[m]["logp"].reindex(perpair.index)
perpair = perpair.reset_index()
perpair.to_csv(OUT_DIR / "combined_model_perpair.csv", index=False)

# ── FIGURE (house style; answers Aurélie review comment [18]) ────────────────
# Strategy order tells the story left→right: CADD-only, CADD with 5ULTRA-scope
# variants ablated, the two naive CADD+5ULTRA fusions, then 5ULTRA-only.
STRAT = ["CADD_Weighted", "Ablation_Weighted", "Combined_Sum", "Combined_Max", "5ULTRA_Weighted"]
STRAT_LABEL = ["CADD\nonly", "CADD\nablated", "Combined\n(sum)", "Combined\n(max)", "5ULTRA\nonly"]
STRAT_COLOR = {
    "CADD_Weighted":    "#B0B8BD",  # medium grey
    "Ablation_Weighted":"#D5DBDB",  # light grey (ablation control)
    "Combined_Sum":     "#9B59B6",  # purple — fusion
    "Combined_Max":     "#6C3483",  # deep purple — fusion
    "5ULTRA_Weighted":  "#C0392B",  # 5ULTRA red (house C_5U_EX)
}
C_5U_EX = "#C0392B"

n_primary = (counts[counts["threshold"] == round(PRIMARY, 3)]
             .set_index("model")["n_gws"])

fig = plt.figure(figsize=(7.2, 3.5), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.42,
                      left=0.085, right=0.975, top=0.86, bottom=0.18)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ── Panel a: discovery yield by strategy ─────────────────────────────────────
xs = np.arange(len(STRAT))
vals = [int(n_primary[m]) for m in STRAT]
for i, m in enumerate(STRAT):
    ax_a.bar(xs[i], vals[i], width=0.66, color=STRAT_COLOR[m],
             edgecolor="#999999" if "5ULTRA" not in m and "Combined" not in m else "none",
             linewidth=0.3, zorder=3)
    ax_a.text(xs[i], vals[i] + 0.5, str(vals[i]), ha="center", va="bottom",
              fontsize=9.5, fontweight="bold",
              color=C_5U_EX if m == "5ULTRA_Weighted" else "#444444")
ax_a.axhline(len(union_single), ls="--", color="#555555", lw=0.8, zorder=2)
ax_a.text(-0.45, len(union_single) + 0.5,
          r"5ULTRA $\cup$ CADD = %d  (= 5ULTRA alone)" % len(union_single),
          ha="left", va="bottom", fontsize=7, color="#555555")
ax_a.set_xticks(xs)
ax_a.set_xticklabels(STRAT_LABEL, fontsize=7.8)
ax_a.set_ylabel("GWS gene–phenotype associations", fontsize=9, fontweight="bold")
ax_a.set_ylim(0, max(vals) * 1.22)
ax_a.set_xlim(-0.6, len(STRAT) - 0.4)
ax_a.set_title(r"Discovery yield by testing strategy  ($-\log_{10}P$ > %.2f)" % PRIMARY,
               fontsize=8.5)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.tick_params(axis="x", length=0, pad=3)
ax_a.text(-0.13, 1.06, "a", transform=ax_a.transAxes,
          fontsize=13, fontweight="bold", va="top")

# ── Panel b: combining loses signal (per-association dilution) ────────────────
idx = perpair.set_index(["gene", "pheno"]).index
best_single = np.maximum(
    best["5ULTRA_Weighted"]["logp"].reindex(idx).fillna(0).values,
    best["CADD_Weighted"]["logp"].reindex(idx).fillna(0).values,
)
cmb = perpair["Combined_Max"].fillna(0).values
lim = max(np.nanmax(best_single), np.nanmax(cmb)) * 1.06
ax_b.plot([0, lim], [0, lim], ls="--", color="#999999", lw=0.8, zorder=1)
below = cmb < best_single - 0.2
ax_b.scatter(best_single[~below], cmb[~below], s=20, color="#6C3483",
             edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
ax_b.scatter(best_single[below], cmb[below], s=20, color="#6C3483",
             edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
ax_b.axhline(PRIMARY, ls=":", color=C_5U_EX, lw=0.8, zorder=2)
ax_b.axvline(PRIMARY, ls=":", color=C_5U_EX, lw=0.8, zorder=2)
ax_b.set_xlim(0, lim)
ax_b.set_ylim(0, lim)
ax_b.set_xlabel(r"Best single source  max(5ULTRA, CADD)  $-\log_{10}P$",
                fontsize=8.5, fontweight="bold")
ax_b.set_ylabel(r"Combined (max)  $-\log_{10}P$", fontsize=8.5, fontweight="bold")
ax_b.set_title("Fusion dilutes per-association signal", fontsize=8.5)
ax_b.spines[["top", "right"]].set_visible(False)
n_below = int(below.sum())
ax_b.text(0.05, 0.95, f"{n_below}/{len(cmb)} below identity",
          transform=ax_b.transAxes, fontsize=7.5, va="top", color="#555555")
ax_b.text(-0.16, 1.06, "b", transform=ax_b.transAxes,
          fontsize=13, fontweight="bold", va="top")

for fmt in ("png", "pdf"):
    out = OUT_DIR / f"Figure_Combined_Strategy.{fmt}"
    fig.savefig(out, bbox_inches="tight", dpi=300 if fmt == "png" else None)
    print(f"Saved: {out}")
print(f"\nWrote CSVs:\n  {OUT_DIR/'combined_model_counts.csv'}\n  {OUT_DIR/'combined_model_overlap.csv'}\n"
      f"  {OUT_DIR/'combined_model_perpair.csv'}")
