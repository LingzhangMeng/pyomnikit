"""Spatial niche statistics (Gap G2; docs/MATH.md §4)."""
import numpy as np
from scipy.spatial import cKDTree, distance as _sd
from scipy.sparse import coo_matrix, diags
from .stats import perm_p


def knn_graph(coords, k=6, radius=None, row_normalize=True):
    """Symmetric kNN (or fixed-radius) spatial graph; row-normalised by default (scipy CSR)."""
    coords = np.asarray(coords, float); n = coords.shape[0]
    tree = cKDTree(coords)
    if radius is None:
        _, idx = tree.query(coords, k=k + 1)
        i = np.repeat(np.arange(n), k); j = idx[:, 1:].ravel()
    else:
        pairs = list(tree.query_pairs(radius))
        i = np.array([p[0] for p in pairs], int); j = np.array([p[1] for p in pairs], int)
    W = coo_matrix((np.ones(i.size), (i, j)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    if row_normalize:
        rs = np.asarray(W.sum(1)).ravel(); rs[rs == 0] = 1
        W = diags(1.0 / rs) @ W
    return W


def moran(x, W, n_perm=999, seed=0):
    x = np.asarray(x, float); n = x.size; S0 = W.sum()
    stat = lambda v: (n / S0) * float((v - v.mean()) @ (W @ (v - v.mean()))) / float(((v - v.mean()) ** 2).sum())
    I = stat(x)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(x)) for _ in range(n_perm)])
    return {"I": I, "EI": -1 / (n - 1), "p_perm": perm_p(I, null, "two.sided"), "null_sd": float(null.std())}


def geary(x, W, n_perm=999, seed=0):
    x = np.asarray(x, float); n = x.size; S0 = W.sum()
    rs = np.asarray(W.sum(1)).ravel(); cs = np.asarray(W.sum(0)).ravel()
    def stat(v):
        z = v - v.mean()
        num = (v ** 2 * (rs + cs)).sum() - 2 * float(v @ (W @ v))
        return ((n - 1) / (2 * S0)) * num / (z ** 2).sum()
    C = stat(x)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(x)) for _ in range(n_perm)])
    return {"C": C, "EC": 1.0, "p_perm": perm_p(C, null, "two.sided")}


def lisa(x, W, n_perm=999, seed=0):
    x = np.asarray(x, float); z = x - x.mean()
    lag = W @ z; Ii = z * lag
    rng = np.random.default_rng(seed)
    null = np.array([(lambda zs: zs * (W @ zs))(rng.permutation(z)) for _ in range(n_perm)])
    p = np.array([perm_p(Ii[i], null[:, i], "two.sided") for i in range(x.size)])
    return {"Ii": Ii, "lag": lag, "p_perm": p}


def nhood_enrichment(labels, W, n_perm=999, seed=0):
    labels = np.asarray(labels, object); lv = np.unique(labels); K = lv.size
    idx = {c: i for i, c in enumerate(lv)}
    Wc = W.tocoo(); m = Wc.row < Wc.col
    ii, jj = Wc.row[m], Wc.col[m]
    a = np.array([idx[c] for c in labels[ii]]); b = np.array([idx[c] for c in labels[jj]])
    obs = np.zeros((K, K)); np.add.at(obs, (a, b), 1); np.add.at(obs, (b, a), 1)
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, K, K))
    for p in range(n_perm):
        lb = rng.permutation(labels)
        aa = np.array([idx[c] for c in lb[ii]]); bb = np.array([idx[c] for c in lb[jj]])
        np.add.at(null[p], (aa, bb), 1); np.add.at(null[p], (bb, aa), 1)
    mu, sd = null.mean(0), null.std(0)
    return {"observed": obs, "zscore": (obs - mu) / np.maximum(sd, 1e-9)}


def ripley(coords, radii=None, mark=None):
    coords = np.asarray(coords, float)
    if mark is not None:
        coords = coords[np.asarray(mark, bool)]
    n = coords.shape[0]
    rng = coords.min(0), coords.max(0)
    A = np.prod(rng[1] - rng[0])
    if radii is None:
        radii = np.linspace(0, 0.25 * np.sqrt(A), 30)[1:]
    D = None
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    inside = ((coords[:, 0][:, None] >= rng[0][0] + radii) & (coords[:, 0][:, None] <= rng[1][0] - radii) &
              (coords[:, 1][:, None] >= rng[0][1] + radii) & (coords[:, 1][:, None] <= rng[1][1] - radii))
    Ks = []
    for k, r in enumerate(radii):
        cnt = tree.count_neighbors(tree, r) - n          # ordered pairs i != j (no O(n^2) matrix)
        frac = inside[:, k].sum() / n if n else 1
        Ks.append((A / (n * (n - 1))) * (cnt / max(frac, 1e-9)))
    Ks = np.array(Ks)
    return {"radius": radii, "K": Ks, "L": np.sqrt(Ks / np.pi) - radii}


def spatial_lr(expr, lr_pairs, W, genes=None, n_perm=999, seed=0):
    """Spatial ligand-receptor co-expression on the graph (permutation null)."""
    import pandas as pd
    expr = np.asarray(expr, float)
    genes = [f"g{i}" for i in range(expr.shape[1])] if genes is None else list(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    if isinstance(lr_pairs, dict):
        lr_pairs = list(lr_pairs.items())
    S0 = W.sum()
    gamma = lambda s: float(s @ (W @ s)) / S0
    rng = np.random.default_rng(seed)
    rows = []
    for pair in lr_pairs:
        L, R = (pair if not isinstance(pair, tuple) else pair)
        if L not in gidx or R not in gidx:
            rows.append({"ligand": L, "receptor": R, "gamma": np.nan, "p_perm": np.nan}); continue
        s = ((expr[:, gidx[L]] > 0) & (expr[:, gidx[R]] > 0)).astype(float)
        g = gamma(s)
        null = np.array([gamma(rng.permutation(s)) for _ in range(n_perm)])
        rows.append({"ligand": L, "receptor": R, "gamma": g, "p_perm": perm_p(g, null, "greater")})
    df = pd.DataFrame(rows)
    return df.sort_values("gamma", ascending=False).reset_index(drop=True)
