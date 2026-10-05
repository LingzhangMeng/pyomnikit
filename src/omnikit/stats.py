"""Shared estimators (docs/MATH.md §0)."""
import numpy as np
from scipy import stats as _st


def auc(x, pos):
    """AUC of x against a boolean ``pos`` mask via the normalised Mann-Whitney U."""
    x = np.asarray(x, float); pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _st.rankdata(x)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def mwu(x, y, alternative="two.sided"):
    """Mann-Whitney U with a tie-corrected normal approximation."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n1, n0 = len(x), len(y)
    r = _st.rankdata(np.concatenate([x, y]))
    U = r[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n0 / 2
    _, cnt = np.unique(np.concatenate([x, y]), return_counts=True)
    tie = np.sum(cnt ** 3 - cnt)
    N = n1 + n0
    sig2 = (n1 * n0 / 12) * ((N + 1) - tie / (N * (N - 1)))
    z = (U - mu) / np.sqrt(sig2)
    p = {"two.sided": 2 * _st.norm.sf(abs(z)), "greater": _st.norm.sf(z), "less": _st.norm.cdf(z)}[alternative]
    return {"statistic": U, "z": z, "p": p}


def delong(scores, pos, conf=0.95):
    """DeLong AUC with variance/CI (fast midrank form)."""
    scores = np.asarray(scores, float); pos = np.asarray(pos, bool)
    x, y = scores[pos], scores[~pos]
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return {"auc": auc(scores, pos), "se": np.nan, "lower": np.nan, "upper": np.nan}
    R = _st.rankdata(np.concatenate([x, y]))
    Rx, Ry = R[:m], R[m:]
    rxx, ryy = _st.rankdata(x), _st.rankdata(y)
    V10 = (Rx - rxx) / n
    V01 = (Ry - ryy) / m
    a = V10.mean()
    se = np.sqrt(V10.var(ddof=1) / m + V01.var(ddof=1) / n)
    z = _st.norm.ppf(1 - (1 - conf) / 2)
    return {"auc": a, "se": se, "lower": a - z * se, "upper": a + z * se}


def stouffer(z, w=None):
    """Weighted Stouffer meta-analysis."""
    z = np.asarray([v for v in np.asarray(z, float) if np.isfinite(v)])
    if z.size == 0:
        return {"Z": np.nan, "p": np.nan, "k": 0}
    w = np.ones_like(z) if w is None else np.asarray(w, float)[: z.size]
    Z = float((w * z).sum() / np.sqrt((w ** 2).sum()))
    return {"Z": Z, "p": 2 * _st.norm.sf(abs(Z)), "k": z.size}


def eb_shrink(beta, se):
    """Empirical-Bayes (method-of-moments) shrinkage toward the grand mean."""
    beta = np.asarray(beta, float); se = np.asarray(se, float)
    ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    b, s = beta[ok], se[ok]
    bbar = np.median(b)
    tau2 = max(0.0, np.var(b, ddof=1) - np.mean(s ** 2))
    w = tau2 / (tau2 + s ** 2) if tau2 > 0 else np.zeros_like(s)
    out = np.full(beta.shape, np.nan)
    out[ok] = bbar + w * (b - bbar)
    return out


def fisher(p, R=None):
    """Fisher combine of p-values (optional Brown correction given correlation R)."""
    p = np.asarray([v for v in np.asarray(p, float) if np.isfinite(v) and 0 < v <= 1])
    k = p.size
    if k == 0:
        return {"X2": np.nan, "p": np.nan, "df": 0, "k": 0}
    X2 = float((-2 * np.log(p)).sum())
    out = {"X2": X2, "p": float(_st.chi2.sf(X2, 2 * k)), "df": 2 * k, "k": k}
    if R is not None and k > 1:
        R = np.asarray(R, float)
        V = 4 * k + 8 * np.triu(R, 1).sum()
        X2b = (X2 - 2 * k) * np.sqrt(2 * k) / np.sqrt(V) + 2 * k
        out["X2_brown"] = X2b
        out["p_brown"] = float(_st.chi2.sf(X2b, 2 * k))
    return out


def cohen_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n0 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n0 - 1) * y.var(ddof=1)) / (n1 + n0 - 2))
    return (x.mean() - y.mean()) / sp


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(_st.spearmanr(x[ok], y[ok]).statistic)


def perm_p(obs, null, alternative="greater"):
    null = np.asarray([v for v in np.asarray(null, float) if np.isfinite(v)])
    B = null.size
    if B == 0:
        return np.nan
    if alternative == "greater":
        return (1 + (null >= obs).sum()) / (B + 1)
    m = null.mean()
    return (1 + (np.abs(null - m) >= abs(obs - m)).sum()) / (B + 1)


def bh(p):
    return _bh_manual(np.asarray(p, float))


def by(p):
    return _by_manual(np.asarray(p, float))


def _bh_manual(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def _by_manual(p):
    p = np.asarray(p, float)
    n = p.size
    c = np.sum(1.0 / np.arange(1, n + 1))
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n * c / k)
        q[i] = prev
    return q
