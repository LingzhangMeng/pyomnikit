# examples/tutorial_5_integration.py
"""Tutorial 5 — cross-omics integration: RV, CCA, joint NMF and MOFA-lite over shared samples.
Self-contained (synthetic). Run from the repo root:  python examples/tutorial_5_integration.py
"""
import os
import numpy as np, pandas as pd
import omnikit as ok
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_FIG, OUT_TAB = "docs/figures", "docs/tutorial_outputs"
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
rng = np.random.default_rng(4)

# ---- three "omics" views over the same 3000 samples, driven by 2 shared factors ----
n = 3000
z = rng.normal(0, 1, (n, 2))
v1 = z @ rng.normal(0, 1, (2, 8)) + rng.normal(0, .8, (n, 8))        # transcriptome (8 features)
v2 = z @ rng.normal(0, 1, (2, 4)) + rng.normal(0, .8, (n, 4))        # mouse transcriptome (4)
v3 = (z @ rng.normal(0, 1, (2, 1))).reshape(n, 1) + rng.normal(0, 1, (n, 1))   # genetics (1)

# ---- RV coefficient + CCA ----
rv = {"human~mouse": ok.rv(v1, v2), "human~genetics": ok.rv(v1, v3), "mouse~genetics": ok.rv(v2, v3)}
cc = ok.cca(v1, v3, k=3)
print("RV:", {k: round(x, 3) for k, x in rv.items()})
print("CCA canonical correlations:", np.round(cc["cor"], 3))
pd.DataFrame([{"pair": k, "RV": round(v, 3)} for k, v in rv.items()]).to_csv(
    f"{OUT_TAB}/tutorial_5_rv.csv", index=False)

# ---- joint multi-view NMF + MOFA-lite ----
nmf = ok.joint_nmf([np.abs(v1), np.abs(v2), np.abs(v3)], k=2, n_iter=300, seed=0)
mf = ok.mofa_lite([v1, v2, v3], k=2, n_iter=200, seed=0)
print("MOFA variance explained per view:", np.round(mf["var_explained"], 3))
pd.DataFrame({"view": ["human", "mouse", "genetics"], "var_explained": mf["var_explained"]}).to_csv(
    f"{OUT_TAB}/tutorial_5_mofa.csv", index=False)

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
ax[0].bar(range(3), list(rv.values()), color=["#4c72b0", "#dd8452", "#55a868"])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(list(rv.keys()), rotation=15, fontsize=8)
ax[0].set_ylabel("RV coefficient"); ax[0].set_title("(a) RV similarity", fontsize=10)
xv = v1 @ cc["x_weights"][:, 0]; yv = v3 @ cc["y_weights"][:, 0]
ax[1].scatter(xv, yv, s=4, c="#9aa7b3", alpha=.4, linewidths=0)
ax[1].set_xlabel("human canonical variate 1"); ax[1].set_ylabel("genetics canonical variate 1")
ax[1].set_title(f"(b) CCA  (r = {cc['cor'][0]:.2f})", fontsize=10)
ax[2].bar(["human", "mouse", "genetics"], mf["var_explained"], color="#6f8fb0")
ax[2].set_ylabel("variance explained"); ax[2].set_title("(c) MOFA-lite per view", fontsize=10)
for a in ax:
    a.grid(False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("Tutorial 5 \u2014 cross-omics integration", fontsize=11)
fig.savefig(f"{OUT_FIG}/tutorial_5.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT_FIG}/tutorial_5.pdf", bbox_inches="tight")
print("wrote docs/figures/tutorial_5.{png,pdf} + docs/tutorial_outputs/tutorial_5_*.csv")
