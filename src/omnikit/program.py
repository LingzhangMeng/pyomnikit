"""Program scoring & conservation (Gap G4; docs/MATH.md §2)."""
import numpy as np
from .stats import auc, mwu, bh, spearman
from scipy import stats as _st


def ucell(X, gene_sets, genes=None, max_rank=1500, seed=0, chunk=2000):
    """UCell (rank-based), re-implemented. Bound maxU = k*N, N = min(max_rank, G).
    gene_sets: dict name -> gene names (with `genes`) or integer indices. Returns dict name -> scores."""
    X = np.asarray(X, float)
    n, G = X.shape
    N = min(max_rank, G)
    if genes is None:
        gs = {k: np.asarray(v, int) for k, v in gene_sets.items()}
    else:
        gidx = {g: i for i, g in enumerate(genes)}
        gs = {k: np.array([gidx[g] for g in v if g in gidx], int) for k, v in gene_sets.items()}
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n, np.float32) for k in gs}
    for s in range(0, n, chunk):
        blk = X[s:s + chunk].copy()
        blk += rng.random(blk.shape, dtype=np.float32) * 1e-4
        order = np.argsort(-blk, axis=1, kind="quicksort")
        ranks = np.empty(blk.shape, np.float32)
        ranks[np.arange(blk.shape[0])[:, None], order] = np.arange(1, G + 1, dtype=np.float32)
        np.minimum(ranks, max_rank, out=ranks)
        for k, idx in gs.items():
            kk = idx.size
            if kk == 0:
                out[k][s:s + chunk] = np.nan
                continue
            U = ranks[:, idx].sum(1)
            out[k][s:s + chunk] = (kk * N - U) / (kk * N - kk * (kk + 1) / 2.0)
    return out


def ucell_signed(X, up, down, **kw):
    s = ucell(X, {"up": up, "down": down}, **kw)
    return s["up"] - s["down"]


def score_genes(X, gene_set, genes=None, ctrl_size=50, n_bins=25, seed=0, zscore=False):
    """Control-bin (expression-matched) score, scanpy/Pagès style (docs/MATH.md §2.2)."""
    X = np.asarray(X, float)
    n, G = X.shape
    genes = np.arange(G) if genes is None else np.asarray(genes, object)
    gmean = X.mean(0)
    edges = np.unique(np.quantile(gmean, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(gmean, edges[1:-1]), 0, len(edges) - 2)
    rng = np.random.default_rng(seed)
    sig = np.array([i for i, g in enumerate(genes) if g in set(gene_set)], int)
    cols = []
    for b in np.unique(bins):
        inb = np.where(bins == b)[0]
        sig_b = np.intersect1d(sig, inb)
        if sig_b.size == 0:
            continue
        pool = np.setdiff1d(inb, sig_b)
        nb = min(ctrl_size, pool.size)
        if nb == 0:
            continue
        ctrl = rng.choice(pool, nb, replace=False)
        cols.append(X[:, sig_b].mean(1) - X[:, ctrl].mean(1))
    if not cols:
        return np.full(n, np.nan)
    M = np.column_stack(cols)
    score = M.mean(1)
    if zscore:
        score = score / np.maximum(M.std(1), 1e-9)
    return score


def score_diagnostics(score, lib):
    return {"spearman_log_lib": spearman(np.asarray(score, float), np.log1p(np.asarray(lib, float)))}


def conservation(human_t, mouse_rho, human_p, mouse_p, sign_consistent=None,
                 t0=2.0, rho0=0.27, alpha=0.05, genes=None):
    """Cross-species conservation call (Fisher combine + sign + thresholds)."""
    t = np.asarray(human_t, float); rho = np.asarray(mouse_rho, float)
    X2 = -2 * (np.log(np.asarray(human_p, float)) + np.log(np.asarray(mouse_p, float)))
    p = _st.chi2.sf(X2, 4)
    q = bh(p)
    sign_ok = np.sign(t) == np.sign(rho) if sign_consistent is None else np.asarray(sign_consistent, bool)
    conserved = sign_ok & (np.abs(t) >= t0) & (np.abs(rho) >= rho0) & (q < alpha)
    df = {"combined_p": p, "combined_q": q, "sign_consistent": sign_ok, "conserved": conserved}
    if genes is not None:
        df["gene"] = np.asarray(genes)
    return df


def carrier(scores, celltype, donor=None, min_cells=10):
    """Per-cell-type AUC vs rest (+BH) and an optional donor-aware within-donor contrast."""
    import pandas as pd
    scores = np.asarray(scores, float); celltype = np.asarray(celltype, object)
    rows = []
    for ct in np.unique(celltype):
        m = celltype == ct
        if m.sum() < min_cells or m.sum() == len(m):
            continue
        r = {"cell_type": ct, "n_cells": int(m.sum()), "mean_score": float(scores[m].mean()),
             "auc_vs_rest": auc(scores, m), "p_mwu": mwu(scores[m], scores[~m])["p"],
             "donor_diff_median": np.nan, "donor_p": np.nan}
        if donor is not None:
            donor = np.asarray(donor, object)
            diffs = []
            for d in np.unique(donor):
                ix = np.where(donor == d)[0]
                own = ix[celltype[ix] == ct]; oth = ix[celltype[ix] != ct]
                if own.size and oth.size:
                    diffs.append(scores[own].mean() - scores[oth].mean())
            diffs = np.array([x for x in diffs if np.isfinite(x)])
            if diffs.size:
                r["donor_diff_median"] = float(np.median(diffs))
                if diffs.size >= 3:
                    r["donor_p"] = float(_st.wilcoxon(diffs).pvalue)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["q_BH"] = bh(df["p_mwu"].values)
    return df.sort_values("auc_vs_rest", ascending=False).reset_index(drop=True)
