# examples/tutorial_2_conservation.py
"""Tutorial 2 — call a cross-species conserved gene program from per-gene statistics.
Self-contained (synthetic). Run from the repo root:  python examples/tutorial_2_conservation.py
"""
import os
import numpy as np, pandas as pd
import omnikit as ok
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_FIG, OUT_TAB = "docs/figures", "docs/tutorial_outputs"
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
rng = np.random.default_rng(1)

# ---- synthetic per-gene statistics over 2000 genes ----
n = 2000
genes = [f"G{i:04d}" for i in range(n)]
true_prog = rng.random(n) < 0.15                       # a true conserved subset
human_t = rng.normal(0, 1, n)
mouse_rho = rng.normal(0, 0.15, n)
human_t[true_prog] += rng.normal(3, 1, true_prog.sum())     # program genes: real effect
mouse_rho[true_prog] += rng.normal(0.5, 0.1, true_prog.sum())
human_p = 2 * stats.norm.sf(np.abs(human_t))
mouse_p = 2 * stats.t.sf(np.abs(mouse_rho) * np.sqrt(50) / np.sqrt(1 - mouse_rho**2), 50)

# ---- conservation call (Fisher combine + sign consistency + thresholds) ----
call = ok.conservation(human_t, mouse_rho, human_p, mouse_p, genes=genes,
                       t0=2.0, rho0=0.27, alpha=0.05)
df = pd.DataFrame(call)
df.to_csv(f"{OUT_TAB}/tutorial_2_conservation.csv", index=False)

prec = (df.loc[df.conserved, "gene"].isin([g for g, p in zip(genes, true_prog) if p])).mean()
print(f"conserved genes: {int(df.conserved.sum())} | precision vs truth: {prec:.2f}")

# ---- figure: human_t vs mouse_rho, conserved genes in red ----
fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
ax.scatter(df.human_t, df.mouse_rho, s=5, c="#cccccc", linewidths=0, label="not conserved")
ax.scatter(df.human_t[df.conserved], df.mouse_rho[df.conserved], s=6, c="#d62728",
           linewidths=0, label="conserved")
ax.axhline(0, lw=.6, c="k"); ax.axvline(0, lw=.6, c="k")
ax.set_xlabel("human effect (t)"); ax.set_ylabel("mouse trend (\u03c1)")
ax.set_title(f"Tutorial 2 \u2014 conserved program ({int(df.conserved.sum())} genes)", fontsize=10)
ax.legend(fontsize=8, frameon=False); ax.grid(False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(f"{OUT_FIG}/tutorial_2.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUT_FIG}/tutorial_2.pdf", bbox_inches="tight")
print("wrote docs/figures/tutorial_2.{png,pdf} + docs/tutorial_outputs/tutorial_2_conservation.csv")
