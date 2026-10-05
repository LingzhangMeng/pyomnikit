# examples/tutorial_3_qc_doublets.py
"""Tutorial 3 — QC-first audit: per-cell metrics, Scrublet-style doublets, batch mixing (LISI).
Self-contained (synthetic). Run from the repo root:  python examples/tutorial_3_qc_doublets.py
"""
import os
import numpy as np, pandas as pd
import omnikit as ok
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_FIG, OUT_TAB = "docs/figures", "docs/tutorial_outputs"
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
rng = np.random.default_rng(2)

# ---- synthetic counts: 2 batches, some mitochondrial genes, injected doublets ----
n, g = 2000, 300
genes = [f"G{i:03d}" for i in range(g)]
genes[0], genes[1] = "MT-CO1", "MT-ND1"                       # mitochondrial
X = rng.poisson(2.0, (n, g)).astype(float)
X[:, :2] += rng.poisson(5, (n, 2))                            # mt signal (for %mt)
for i in range(40):                                           # 40 artificial doublets
    a, b = rng.integers(0, n, 2)
    X[i] = X[a] + X[b]
batch = np.repeat(["batchA", "batchB"], n // 2)

# ---- (1) QC metrics ----
qc = ok.qc_metrics(X, genes=genes)
qc["batch"] = batch
qc.groupby("batch").median(numeric_only=True).round(2).to_csv(f"{OUT_TAB}/tutorial_3_qc_summary.csv")
print("median %mt:", round(qc.pct_mt.median(), 3))

# ---- (2) Scrublet-style doublets ----
# embed on a small PCA of log-normalised data, then kNN in the observed+simulated space
d = ok.doublet_scrublet(X, k=30, n_pcs=20, seed=0)
pd.DataFrame({"score": d["score"], "call": d["call"]}).to_csv(f"{OUT_TAB}/tutorial_3_doublets.csv", index=False)
print(f"doublets called: {int(d['call'].sum())}/{n} ({100*d['call'].mean():.1f}%)  threshold={d['threshold']:.3f}")

# ---- (3) batch mixing (LISI) on a 2-D PCA ----
Xn = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
Xc = Xn - Xn.mean(0)
pcs = np.linalg.svd(Xc, full_matrices=False)[0][:, :10]
mx = ok.mixing(pcs, batch, k=30)
print(f"LISI mean = {mx['mean']:.2f} (max {mx['max']})")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
ax[0].scatter(qc.nCount, qc.pct_mt, s=4, c=np.where(d["call"], "#d62728", "#9aa7b3"), alpha=.5, linewidths=0)
ax[0].set_xlabel("nCount"); ax[0].set_ylabel("% mitochondrial")
ax[0].set_title("(a) QC (red = doublet)", fontsize=10)
ax[1].hist(d["score"], bins=50, color="#6f8fb0")
ax[1].axvline(d["threshold"], color="crimson", ls="--", lw=1, label=f"threshold {d['threshold']:.2f}")
ax[1].set_xlabel("doublet score"); ax[1].set_ylabel("cells"); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_title("(b) doublet score", fontsize=10)
ax[2].hist(mx["lisi"], bins=40, color="#8fae6f")
ax[2].axvline(1, ls=":", c="k", lw=.8); ax[2].axvline(mx["max"], ls=":", c="k", lw=.8)
ax[2].set_xlabel("LISI"); ax[2].set_title(f"(c) mixing (mean {mx['mean']:.2f})", fontsize=10)
for a in ax:
    a.grid(False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("Tutorial 3 \u2014 QC-first audit", fontsize=11)
fig.savefig(f"{OUT_FIG}/tutorial_3.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT_FIG}/tutorial_3.pdf", bbox_inches="tight")
print("wrote docs/figures/tutorial_3.{png,pdf} + docs/tutorial_outputs/tutorial_3_*.csv")
