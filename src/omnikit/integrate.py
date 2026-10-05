"""Cross-omics integration (Gap G1; docs/MATH.md §5)."""
import numpy as np
from scipy import stats as _st


def joint_nmf(views, k=5, n_iter=200, tol=1e-5, seed=0):
    """Multi-view NMF with a shared sample-factor W (X^(v) ~ W H^(v))."""
    Xs = [np.maximum(np.asarray(V, float), 0) for V in views]
    n, V = Xs[0].shape[0], len(Xs)
    rng = np.random.default_rng(seed)
    W = rng.random((n, k))
    Hs = [rng.random((k, X.shape[1])) for X in Xs]
    obj = []
    for _ in range(n_iter):
        numW = np.zeros((n, k)); denW = np.zeros((n, k))
        for X, H in zip(Xs, Hs):
            numW += X @ H.T; denW += W @ (H @ H.T)
        W *= numW / np.maximum(denW, 1e-10)
        for v in range(V):
            Hs[v] *= (W.T @ Xs[v]) / np.maximum(W.T @ W @ Hs[v], 1e-10)
        o = sum(((X - W @ H) ** 2).sum() for X, H in zip(Xs, Hs))
        obj.append(o)
        if len(obj) > 1 and abs(obj[-2] - obj[-1]) / max(obj[-2], 1e-10) < tol:
            break
    return {"W": W, "H": Hs, "objective": np.array(obj), "k": k}


def cca(X, Y, k=None):
    """Canonical correlation analysis (canonical correlations + weights)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    n1 = X.shape[0] - 1
    Sxx = X.T @ X / n1; Syy = Y.T @ Y / n1; Sxy = X.T @ Y / n1
    def wh(S):
        w, V = np.linalg.eigh(S)
        return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    Wx, Wy = wh(Sxx), wh(Syy)
    U, s, Vt = np.linalg.svd(Wx @ Sxy @ Wy)
    kk = s.size if k is None else min(k, s.size)
    return {"cor": s[:kk], "x_weights": Wx @ U[:, :kk], "y_weights": Wy @ Vt[:kk].T}


def rv(A, B):
    """RV coefficient (0..1)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A - A.mean(0); B = B - B.mean(0)
    Sxy = A.T @ B; Sxx = A.T @ A; Syy = B.T @ B
    return float((Sxy ** 2).sum()) / np.sqrt((Sxx ** 2).sum() * (Syy ** 2).sum())


def procrustes(X, Y):
    """Procrustes rotation/scale aligning X to Y."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    return {"rotation": R, "scale": float(s.sum() / (X ** 2).sum()),
            "residual": float(1 - (s.sum() ** 2) / ((X ** 2).sum() * (Y ** 2).sum()))}


def mofa_lite(views, k=5, n_iter=200, tol=1e-6, seed=0):
    """MOFA-style multi-view factor analysis (EM)."""
    Xs = [np.asarray(V, float) for V in views]
    n, V = Xs[0].shape[0], len(Xs)
    rng = np.random.default_rng(seed)
    Lam = [rng.normal(0, 0.1, (k, X.shape[1])) for X in Xs]
    sig2 = np.array([0.5 * X.var() for X in Xs])
    ll = []
    Ez = np.zeros((n, k))
    for _ in range(n_iter):
        M = np.eye(k)
        for v in range(V):
            M += Lam[v] @ Lam[v].T / sig2[v]
        Minv = np.linalg.inv(M)
        Ez = np.zeros((n, k))
        for v in range(V):
            Ez += Xs[v] @ Lam[v].T / sig2[v]
        Ez = Ez @ Minv
        Ezz = n * Minv + Ez.T @ Ez
        for v in range(V):
            Lam[v] = np.linalg.solve(Ezz, Ez.T @ Xs[v])
            sig2[v] = ((Xs[v] - Ez @ Lam[v]) ** 2).sum() / (n * Xs[v].shape[1])
        ll.append(sum(-n * X.shape[1] / 2 * np.log(sig2[i]) -
                      ((X - Ez @ Lam[i]) ** 2).sum() / (2 * sig2[i]) for i, X in enumerate(Xs)))
        if len(ll) > 1 and abs(ll[-2] - ll[-1]) / max(abs(ll[-2]), 1e-9) < tol:
            break
    ve = np.array([1 - sig2[i] / Xs[i].var() for i in range(V)])
    return {"factors": Ez, "loadings": Lam, "sigma2": sig2, "var_explained": ve,
            "loglik": np.array(ll), "k": k}


def program_gwas(program_score, control_score=None):
    """scDRS-flavoured program-GWAS association."""
    obs = float(np.nanmean(program_score))
    if control_score is None:
        return {"stat": obs, "z": np.nan, "p": np.nan, "n_control": 0}
    z = (obs - np.mean(control_score)) / np.std(control_score)
    return {"stat": obs, "z": z, "p": 2 * _st.norm.sf(abs(z)), "n_control": len(control_score)}
