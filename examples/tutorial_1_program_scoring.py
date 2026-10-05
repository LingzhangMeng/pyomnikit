# examples/tutorial_1_program_scoring.py
"""Tutorial 1 — score a gene program, diagnose/remove the depth confound, find its carriers.
Self-contained (synthetic data). Run from the repo root:  python examples/tutorial_1_program_scoring.py
"""
import os
import numpy as np, pandas as pd
import omnikit as ok
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_FIG, OUT_TAB = "docs/figures", "docs/tutorial_outputs"
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
rng = np.random.default_rng(0)

# ---- synthetic single-cell matrix: 3000 cells x 400 genes (sparse, variable depth) ----
n_cells, n_genes = 3000, 400
genes = [f"G{i:03d}" for i in range(n_genes)]
gene_rate = rng.gamma(0.6, 1.0, n_genes) + 0.2             # heavy-tailed baseline rates
depth = rng.integers(300, 4000, n_cells).astype(float)     # variable sequencing depth
myeloid = rng.random(n_cells) < 0.30                       # the carrier population
prog_idx = rng.choice(n_genes, 20, replace=False)          # a 20-gene program
gene_rate[prog_idx] = 0.3                                  # low expression -> detection depends on depth
program = [genes[i] for i in prog_idx]                     # names (ucell maps names -> columns)
X = rng.poisson(gene_rate[None, :] * (depth[:, None] / 1000.0)).astype(float)
# realistic 10x dropout: shallow cells lose more counts (the cause of the depth confound)
drop_prob = np.clip(1 - depth / 4000.0, 0, 1) * 0.5
X[rng.random(X.shape) < drop_prob[:, None]] = 0
# embed the program in the carrier population
X[np.ix_(np.where(myeloid)[0], prog_idx)] += rng.poisson(3.0, (int(myeloid.sum()), 20))

# ---- (1) UCell program score (rank-based, full gene set) ----
score = ok.ucell(X, {"program": program}, genes=genes, chunk=1000)["program"]
lib = X.sum(1)
print("Spearman(score, log counts) =", round(ok.score_diagnostics(score, lib)["spearman_log_lib"], 3))

# ---- (2) depth control (linear regression) + the robust non-parametric alternative ----
dc = ok.depth_control(score, np.log1p(lib))
diag = ok.score_diagnostics(score, lib)
ds = ok.depth_stratified_auc(score, np.log1p(lib), myeloid, nq=5)
pd.DataFrame([{"metric": "spearman_score_logcounts_raw", "value": round(diag["spearman_log_lib"], 3)},
              {"metric": "spearman_after_depth_control", "value": round(dc["r_after"], 3)},
              {"metric": "auc_myeloid_raw", "value": round(ds["auc_raw"], 3)},
              {"metric": "auc_myeloid_depth_stratified", "value": round(ds["auc_matched"], 3)}]
             ).to_csv(f"{OUT_TAB}/tutorial_1_diagnostics.csv", index=False)
print("depth diag:", round(diag["spearman_log_lib"], 3), "-> after control:", round(dc["r_after"], 3))
print("myeloid AUC  raw:", round(ds["auc_raw"], 3), "| depth-stratified:", round(ds["auc_matched"], 3))

# ---- (3) which cells carry the program? (AUC vs rest) ----
celltype = np.where(myeloid, "Myeloid", "Other")
tab = ok.carrier(dc["resid"], celltype)
tab.to_csv(f"{OUT_TAB}/tutorial_1_carriers.csv", index=False)
print(tab.to_string(index=False))

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
col = np.where(myeloid, "#d62728", "#9aa7b3")
ax[0].scatter(np.log1p(lib), score, s=3, c=col, alpha=.4, linewidths=0)
ax[0].set_xlabel("log(1+counts)"); ax[0].set_ylabel("UCell score")
ax[0].set_title("(a) raw score vs depth", fontsize=10)
ax[1].scatter(np.log1p(lib), dc["resid"], s=3, c=col, alpha=.4, linewidths=0)
ax[1].set_xlabel("log(1+counts)"); ax[1].set_ylabel("residual")
ax[1].set_title("(b) depth-controlled", fontsize=10)
auc = tab.set_index("cell_type")["auc_vs_rest"]
ax[2].bar(["Myeloid", "Other"], [auc.get("Myeloid", np.nan), auc.get("Other", np.nan)],
          color=["#d62728", "#9aa7b3"])
ax[2].axhline(0.5, ls=":", c="k", lw=.8); ax[2].set_ylabel("AUC vs rest")
ax[2].set_title("(c) carrier AUC", fontsize=10)
for a in ax:
    a.grid(False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("Tutorial 1 \u2014 program scoring, depth control and carriers", fontsize=11)
fig.savefig(f"{OUT_FIG}/tutorial_1.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT_FIG}/tutorial_1.pdf", bbox_inches="tight")
print("wrote docs/figures/tutorial_1.{png,pdf} + docs/tutorial_outputs/tutorial_1_carriers.csv")
