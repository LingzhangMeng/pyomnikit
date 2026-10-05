"""Depth control and ortholog handling (docs/MATH.md §1)."""
import numpy as np
from .stats import auc, spearman


def depth_control(score, log_lib, group=None):
    """Regress score on log library size (per group) and return residuals + diagnostics."""
    score = np.asarray(score, float); log_lib = np.asarray(log_lib, float)
    n = score.size
    group = np.array(["all"] * n) if group is None else np.asarray(group)
    resid = np.full(n, np.nan); fit = np.full(n, np.nan)
    for g in np.unique(group):
        idx = np.where((group == g) & np.isfinite(score) & np.isfinite(log_lib))[0]
        if idx.size < 3:
            continue
        b = np.polyfit(log_lib[idx], score[idx], 1)
        fit[idx] = np.polyval(b, log_lib[idx])
        resid[idx] = score[idx] - fit[idx]
    keep = np.isfinite(score) & np.isfinite(log_lib)
    k2 = keep & np.isfinite(resid)
    return {"resid": resid, "fit": fit,
            "r_before": spearman(score[keep], log_lib[keep]) if keep.sum() > 2 else np.nan,
            "r_after": spearman(resid[k2], log_lib[k2]) if k2.sum() > 2 else np.nan}


def depth_stratified_auc(score, log_lib, pos, nq=5):
    """Depth-stratified (matched) AUC: per-quantile-bin AUC + size-weighted mean."""
    score = np.asarray(score, float); log_lib = np.asarray(log_lib, float); pos = np.asarray(pos, bool)
    keep = np.isfinite(score) & np.isfinite(log_lib)
    score, log_lib, pos = score[keep], log_lib[keep], pos[keep]
    edges = np.unique(np.quantile(log_lib, np.linspace(0, 1, nq + 1)))
    bins = np.clip(np.digitize(log_lib, edges[1:-1], right=False), 0, len(edges) - 2)
    rows, aucs, ns = [], [], []
    for b in np.unique(bins):
        m = bins == b
        a = auc(score[m], pos[m]); rows.append((int(b), int(m.sum()), a))
        aucs.append(a); ns.append(int(m.sum()))
    aucs, ns = np.array(aucs, float), np.array(ns, float)
    return {"per_bin": rows, "auc_matched": np.average(aucs, weights=ns),
            "auc_min": np.nanmin(aucs), "auc_raw": auc(score, pos)}


def casefold_map(symbols):
    """fold -> representative original symbol (drops case collisions)."""
    m, amb = {}, set()
    for g in symbols:
        k = g.upper()
        if k in m and m[k] != g:
            amb.add(k)
        else:
            m[k] = g
    for k in amb:
        m.pop(k, None)
    return {"map": m, "collisions": sorted(amb)}


def collapse_orthologs(pairs_human, pairs_mouse, score=None):
    """Collapse many-to-many ortholog pairs to one representative mouse gene per human gene
    (case-fold-identical preferred, then highest optional score)."""
    h = np.asarray(pairs_human); mo = np.asarray(pairs_mouse)
    sc = np.zeros(h.size) if score is None else np.asarray(score, float)
    ident = np.array([a.upper() == b.upper() for a, b in zip(h, mo)])
    order = np.lexsort((-sc, -ident, h))
    seen, out = set(), []
    for i in order:
        if h[i] not in seen:
            seen.add(h[i]); out.append((h[i], mo[i]))
    return out
