# examples/tutorial_4_spatial.py
"""Tutorial 4 — spatial niche statistics: Moran's I, LISA, neighbourhood graph, Ripley's L.
Self-contained (synthetic). Run from the repo root:  python examples/tutorial_4_spatial.py
"""
import os
import numpy as np, pandas as pd
import omnikit as ok
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_FIG, OUT_TAB = "docs/figures", "docs/tutorial_outputs"
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
rng = np.random.default_rng(3)

# ---- synthetic 2-D tissue: 1500 cells, a smooth spatial gradient + noise ----
n = 1500
coords = rng.random((n, 2)) * 100
signal = np.sin(coords[:, 0] / 15) * np.cos(coords[:, 1] / 15)      # smooth spatial pattern
value = signal + rng.normal(0, 0.3, n)

# ---- spatial graph + Moran / Geary / LISA ----
W = ok.knn_graph(coords, k=6)
mo = ok.moran(value, W, n_perm=999, seed=0)
ge = ok.geary(value, W, n_perm=999, seed=0)
li = ok.lisa(value, W, n_perm=99, seed=0)
pd.DataFrame([{"stat": "Moran_I", "value": round(mo["I"], 3), "p_perm": round(mo["p_perm"], 4)},
              {"stat": "Geary_C", "value": round(ge["C"], 3), "p_perm": round(ge["p_perm"], 4)}]
             ).to_csv(f"{OUT_TAB}/tutorial_4_global.csv", index=False)
print(f"Moran I = {mo['I']:.3f} (p {mo['p_perm']:.3f}) | Geary C = {ge['C']:.3f}")

# ---- neighbourhood enrichment (two synthetic cell types) ----
labels = np.where(rng.random(n) < 0.5, "A", "B")
ne = ok.nhood_enrichment(labels, W, n_perm=199, seed=0)
pd.DataFrame(ne["zscore"], index=["A", "B"], columns=["A", "B"]).to_csv(
    f"{OUT_TAB}/tutorial_4_nhood.csv", index_label="type")

# ---- Ripley's L ----
rp = pd.DataFrame(ok.ripley(coords))
rp.to_csv(f"{OUT_TAB}/tutorial_4_ripley.csv", index=False)

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
s0 = ax[0].scatter(coords[:, 0], coords[:, 1], c=value, cmap="RdBu_r", s=6, linewidths=0)
ax[0].set_title(f"(a) value  (Moran I={mo['I']:.2f})", fontsize=10); ax[0].set_xticks([]); ax[0].set_yticks([])
fig.colorbar(s0, ax=ax[0], shrink=.7)
s1 = ax[1].scatter(coords[:, 0], coords[:, 1], c=li["Ii"], cmap="RdBu_r", s=6, linewidths=0)
ax[1].set_title("(b) local Moran (LISA)", fontsize=10); ax[1].set_xticks([]); ax[1].set_yticks([])
fig.colorbar(s1, ax=ax[1], shrink=.7)
ax[2].plot(rp["radius"], rp["L"], color="#1f77b4"); ax[2].axhline(0, ls=":", c="k", lw=.8)
ax[2].set_xlabel("radius"); ax[2].set_ylabel("L(r) - r"); ax[2].set_title("(c) Ripley's L", fontsize=10)
for a in ax:
    a.grid(False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("Tutorial 4 \u2014 spatial niche statistics", fontsize=11)
fig.savefig(f"{OUT_FIG}/tutorial_4.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT_FIG}/tutorial_4.pdf", bbox_inches="tight")
print("wrote docs/figures/tutorial_4.{png,pdf} + docs/tutorial_outputs/tutorial_4_*.csv")
