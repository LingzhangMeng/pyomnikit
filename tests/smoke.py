"""omnikit (Python mirror) smoke test — run: python tests/smoke.py"""
import numpy as np
import omnikit as ok

rng = np.random.default_rng(0)
fails = []


def check(label, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        fails.append(label)


# ---- G4 UCell bound ----
X = np.repeat((61 - np.arange(1, 61))[None, :], 300, 0).astype(float)
X += rng.random(X.shape) * 0.1
genes = [f"g{i}" for i in range(1, 61)]
uc = ok.ucell(X, {"top": [f"g{i}" for i in range(1, 6)],
                  "bottom": [f"g{i}" for i in range(56, 61)]}, genes=genes, chunk=100)
check("UCell top ~1", (uc["top"] > 0.99).all())
check("UCell bottom ~0", (uc["bottom"] < 0.1).all())
check("UCell in [0,1]", (uc["top"] >= 0).all() and (uc["top"] <= 1).all())

# ---- stats ----
s = np.r_[rng.normal(1, 1, 200), rng.normal(0, 1, 200)]
pos = np.r_[np.ones(200, bool), np.zeros(200, bool)]
d = ok.delong(s, pos)
check("DeLong AUC>0.5 CI brackets", d["auc"] > 0.5 and d["lower"] < d["auc"] < d["upper"])
check("AUC==omni_auc", abs(d["auc"] - ok.auc(s, pos)) < 1e-9)
check("Stouffer 3x2~3.46", abs(ok.stouffer([2, 2, 2])["Z"] - 2 * np.sqrt(3)) < 1e-6)
check("BH monotone", np.all(np.diff(np.sort(ok.bh([0.01, 0.02, 0.5]))) >= -1e-12))
check("BY >= BH", (ok.by([0.01, 0.02, 0.5]) >= ok.bh([0.01, 0.02, 0.5]) - 1e-12).all())

# ---- depth ----
lib = np.exp(rng.normal(8, 1, 500)); tr = rng.normal(0, 1, 500)
dc = ok.depth_control(tr + 0.4 * np.log(lib), np.log(lib))
check("depth control reduces |rho|", abs(dc["r_after"]) < abs(dc["r_before"]))
ds = ok.depth_stratified_auc(tr + 0.4 * np.log(lib), np.log(lib), tr > 0, nq=5)
check("depth-stratified AUC bins", len(ds["per_bin"]) >= 3)

# ---- G3 ----
Xc = rng.poisson(3, (400, 50)).astype(float)
qc = ok.qc_metrics(Xc)
check("QC metrics cols", {"nCount", "nFeature", "pct_mt"} <= set(qc.columns))
check("MAD flags outlier", ok.mad_outlier(np.r_[np.zeros(100), 100])["flag"].any())
Xc[:5] += Xc[5:10]
dbl = ok.doublet_scrublet(Xc, sim_doublets=200, k=15, n_pcs=15)
check("Scrublet in [0,1]", (dbl["score"] >= 0).all() and (dbl["score"] <= 1).all())
mx = ok.mixing(rng.normal(0, 1, (300, 5)), np.repeat(["a", "b", "c"], 100), k=15)
check("LISI in [1,B]", (mx["lisi"] >= 1 - 1e-9).all() and (mx["lisi"] <= mx["max"] + 1e-9).all())

# ---- G2 ----
coords = rng.random((400, 2))
W = ok.knn_graph(coords, k=6)
fld = coords[:, 0] + 0.05 * rng.normal(0, 1, 400)
check("Moran I>0", ok.moran(fld, W, n_perm=99)["I"] > 0)
check("Geary C<1", ok.geary(fld, W, n_perm=99)["C"] < 1)
check("LISA len", ok.lisa(fld, W, n_perm=49)["Ii"].size == 400)
lab = np.where(coords[:, 0] < 0.5, "A", "B")
check("nhood z 2x2", ok.nhood_enrichment(lab, W, n_perm=49)["zscore"].shape == (2, 2))
check("Ripley finite", np.isfinite(ok.ripley(coords)["K"]).all())
E = rng.poisson(2, (400, 3)).astype(float)
slr = ok.spatial_lr(E, [("g0", "g1")], W, genes=["g0", "g1", "g2"], n_perm=99)
check("spatial LR p", np.isfinite(slr["p_perm"][0]))

# ---- G1 ----
n = 200; z = rng.normal(0, 1, (n, 2))
v1 = z @ rng.normal(0, 1, (2, 30)) + rng.normal(0, 1, (n, 30))
v2 = z @ rng.normal(0, 1, (2, 20)) + rng.normal(0, 1, (n, 20))
nmf = ok.joint_nmf([np.abs(v1), np.abs(v2)], k=2, n_iter=60)
check("joint NMF shapes", nmf["W"].shape == (n, 2) and len(nmf["H"]) == 2)
cc = ok.cca(v1, v2, k=3)
check("CCA cor descending", len(cc["cor"]) == 3 and np.all(np.diff(cc["cor"]) <= 1e-9))
check("RV in (0,1)", 0 < ok.rv(v1, v2) < 1)
pr = ok.procrustes(v1, v1 + rng.normal(0, 0.01, v1.shape))
check("Procrustes low residual", pr["residual"] < 0.1)
mf = ok.mofa_lite([v1, v2], k=2, n_iter=60)
check("MOFA shapes", mf["factors"].shape == (n, 2) and len(mf["var_explained"]) == 2)
check("program-GWAS z>0", ok.program_gwas(rng.normal(1, 1, 50), rng.normal(0, 1, 200))["z"] > 0)

print("\nomnikit (python) smoke test:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
raise SystemExit(1 if fails else 0)
