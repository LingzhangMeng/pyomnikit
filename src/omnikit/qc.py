"""QC-first audit (Gap G3; docs/MATH.md §3)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats as _st


def qc_metrics(X, genes=None, mt=None, ribo=None, hb=None):
    """Per-cell QC metrics from integer counts; mt/ribo/hb detected by symbol if not given."""
    X = np.asarray(X, float)
    genes = [f"g{i}" for i in range(X.shape[1])] if genes is None else [str(g) for g in genes]
    up = [g.upper() for g in genes]
    mt = np.array([g.startswith("MT-") or g.startswith("MT.") for g in up]) if mt is None else np.asarray(mt, bool)
    ribo = np.array([g.startswith("RPS") or g.startswith("RPL") for g in up]) if ribo is None else np.asarray(ribo, bool)
    hb = np.array([g.startswith("HB") and len(g) <= 4 for g in up]) if hb is None else np.asarray(hb, bool)
    ncount = X.sum(1)
    pct = lambda m: (100 * X[:, m].sum(1) / np.maximum(ncount, 1)) if m.any() else np.zeros(X.shape[0])
    import pandas as pd
    return pd.DataFrame({"nCount": ncount, "nFeature": (X > 0).sum(1),
                         "pct_mt": pct(mt), "pct_ribo": pct(ribo), "pct_hb": pct(hb)})


def mad_outlier(x, k=3):
    x = np.asarray(x, float)
    med = np.median(x); s = 1.4826 * np.median(np.abs(x - med))
    return {"flag": np.abs(x - med) > k * s, "median": med, "scale": s,
            "lower": med - k * s, "upper": med + k * s}


def gmm2(x, max_iter=200, tol=1e-7):
    """2-component 1-D Gaussian mixture (EM) + density-crossing threshold."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    m = np.array([np.quantile(x, 0.25), np.quantile(x, 0.75)])
    s = np.array([x.std(), x.std()]) / 2 + 1e-9
    w = np.array([0.5, 0.5])
    for _ in range(max_iter):
        d = np.column_stack([w[j] * _st.norm.pdf(x, m[j], s[j]) for j in range(2)])
        r = d / np.maximum(d.sum(1, keepdims=True), 1e-300)
        w0 = w.copy()
        nk = r.sum(0) + 1e-12
        m = (r * x[:, None]).sum(0) / nk
        s = np.sqrt((r * (x[:, None] - m) ** 2).sum(0) / nk) + 1e-12
        w = nk / x.size
        if np.max(np.abs(w - w0)) < tol:
            break
    grid = np.linspace(x.min(), x.max(), 1000)
    dens = np.column_stack([w[j] * _st.norm.pdf(grid, m[j], s[j]) for j in range(2)])
    # threshold = the point where the two components cross (not the min of their sum)
    lo, hi = float(min(m)), float(max(m))
    if hi <= lo:
        hi = lo + 1e-6
    g2 = np.linspace(lo, hi, 1000)
    d0 = w[0] * _st.norm.pdf(g2, m[0], s[0]); d1 = w[1] * _st.norm.pdf(g2, m[1], s[1])
    thr = float(g2[np.argmin(np.abs(d0 - d1))])
    return {"weights": w, "means": m, "sds": s, "threshold": thr}


def doublet_scrublet(X, expected_rate=None, sim_doublets=None, k=30, n_pcs=30, seed=0):
    """Scrublet-style doublet score (Wolock 2019), re-implemented (docs/MATH.md §3.2)."""
    X = np.asarray(X, float); n = X.shape[0]
    if expected_rate is None:
        expected_rate = 0.008 * n / 1000
    if sim_doublets is None:
        sim_doublets = n
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, sim_doublets); j = rng.integers(0, n, sim_doublets)
    sim = X[i] + X[j]
    norm = lambda M: np.log1p(M / np.maximum(M.sum(1, keepdims=True), 1) * 1e4)
    allm = np.vstack([norm(X), norm(sim)])
    v = allm.var(0); hvg = np.argsort(-v)[: min(2000, allm.shape[1])]
    Z = allm[:, hvg]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    npc = min(n_pcs, S.size)
    pcs = U[:, :npc] * S[:npc]
    tree = cKDTree(pcs)
    _, idx = tree.query(pcs[:n], k=k + 1)
    nn = idx[:, 1:]                                   # drop self
    score = (nn >= n).mean(1)
    gm = gmm2(score)
    sep = abs(gm["means"][1] - gm["means"][0]) > 2 * float(np.mean(gm["sds"]))
    # Scrublet's calibrated default: call the top `expected_rate` fraction; the GMM valley is reported
    thr = float(np.quantile(score, 1 - expected_rate))
    return {"call": score >= thr, "score": score, "threshold": thr, "threshold_source": "expected_rate",
            "gmm_threshold": gm["threshold"], "bimodal": sep, "expected_rate": expected_rate, "gmm": gm}


def mixing(embedding, batch, k=30):
    """Local Inverse Simpson Index (LISI): 1 = pure, B = fully mixed."""
    E = np.asarray(embedding, float); batch = np.asarray(batch, object)
    cats = np.unique(batch); code = np.searchsorted(cats, batch)
    tree = cKDTree(E); _, idx = tree.query(E, k=k + 1)
    nn = idx[:, 1:]
    lisi = np.empty(E.shape[0])
    for r in range(E.shape[0]):
        p = np.bincount(code[nn[r]], minlength=cats.size) / nn.shape[1]
        lisi[r] = 1.0 / (p ** 2).sum()
    return {"lisi": lisi, "mean": float(lisi.mean()), "max": cats.size}
