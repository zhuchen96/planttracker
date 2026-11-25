from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import wtrack as W


class MLS:
    """B (um) -> A (um): locally weighted affine through anchors, ridge-pulled to the global affine"""
    def __init__(self, XB, XA, sigma, lam=0.3, k=40):
        self.XB, self.XA, self.sigma, self.lam = XB, XA, sigma, lam
        self.k = min(k, len(XB)); self.tree = cKDTree(XB)
        sol, *_ = np.linalg.lstsq(np.c_[XB, np.ones(len(XB))], XA, rcond=None)
        self.G = sol                                            # 4x3 global affine
        self.M = sol[:3].T; self.t = sol[3]
    def __call__(self, X, chunk=20000):
        out = np.empty_like(X, dtype=float)
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk]
            d, i = self.tree.query(x, k=self.k)
            sig = np.maximum(self.sigma, d[:, min(6, self.k - 1)])[:, None]
            w = np.exp(-0.5 * (d / sig) ** 2)
            P = self.XB[i] - x[:, None]                          # local coordinates
            Ph = np.concatenate([P, np.ones(P.shape[:2] + (1,))], -1)
            Y = self.XA[i]
            # prior: global affine expressed in local coordinates
            G0 = np.broadcast_to(self.G[:3], (len(x), 3, 3)); g0 = x @ self.G[:3] + self.G[3]
            Gl = np.concatenate([G0, g0[:, None]], 1)            # (n,4,3)
            lam = self.lam * np.array([self.sigma ** 2] * 3 + [1e-3])
            N = np.einsum('nki,nk,nkj->nij', Ph, w, Ph) + np.diag(lam)
            Rhs = np.einsum('nki,nk,nkj->nij', Ph, w, Y) + lam[None, :, None] * Gl
            S = np.linalg.solve(N, Rhs)
            out[s:s + chunk] = S[:, 3]                           # value at the query (local origin)
        return out


def jac(T, X, eps=1.0):
    """local volume factor |det dT/dx| at points X"""
    cols = [(T(X + e) - T(X - e)) / (2 * eps) for e in np.eye(3) * eps]
    return np.abs(np.linalg.det(np.stack(cols, -1)))


def families(tab, volB, fmin=0.5):
    """each B cell -> max-overlap A cell; returns {a: (daughters, overlap voxels, daughter voxels)} in B-grid voxels"""
    t = tab[tab[:, 0] > 0]
    cov = np.bincount(t[:, 1], weights=t[:, 2], minlength=len(volB))
    order = np.argsort(-t[:, 2]); seen = set(); fam = {}
    for a, b, c in t[order]:
        if b in seen: continue
        seen.add(b)
        if c / cov[b] >= fmin: fam.setdefault(int(a), []).append(int(b))
    ov = {(int(a), int(b)): c for a, b, c in t}
    return {a: (ds, sum(ov[(a, d)] for d in ds), sum(volB[d] for d in ds)) for a, ds in fam.items()}


def step(seq, a, T, A, B, thr):
    tab, _, volB, volA = W.overlap(seq, a, a + 1, T)
    fam = families(tab, volB)
    ia = {int(l): i for i, l in enumerate(A['labs'])}; ib = {int(l): i for i, l in enumerate(B['labs'])}
    ms = [m for m, (ds, i, vb) in fam.items() if m in ia and all(d in ib for d in ds)]
    XB = np.array([(B['cen'][[ib[d] for d in fam[m][0]]] * B['vol'][[ib[d] for d in fam[m][0]], None]).sum(0)
                   / B['vol'][[ib[d] for d in fam[m][0]]].sum() for m in ms]).reshape(-1, 3)
    XA = np.array([A['cen'][ia[m]] for m in ms]).reshape(-1, 3)
    if len(ms) == 0: return tab, volB, volA, {}, XB, XA, 0.0
    J = jac(T, XB)
    inter = np.array([fam[m][1] for m in ms]) * J; vb = np.array([fam[m][2] for m in ms]) * J
    va = np.array([volA[m] for m in ms])
    iou = inter / (va + vb - inter)
    ok = iou >= thr
    famq = {m: (fam[m][0], float(q)) for m, q in zip(ms, iou)}
    return tab, volB, volA, famq, XB[ok], XA[ok], float(iou.sum())


class Affine:
    def __init__(self, XB, XA):
        sol, *_ = np.linalg.lstsq(np.c_[XB, np.ones(len(XB))], XA, rcond=None)
        self.M = sol[:3].T; self.t = sol[3]
    def __call__(self, X): return X @ self.M.T + self.t
