from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import tifffile
from scipy import ndimage
SP=np.array([1.2,.6,.6])
R=None


def descriptors(cen, vol, k=6):
    tr = cKDTree(cen)
    d, i = tr.query(cen, k=k + 1)
    lv = np.log(vol)
    return np.concatenate([lv[:, None], d[:, 1:] / 3.0, lv[i[:, 1:]]], 1)


def kabsch(X, Y):
    """rigid R,t with Y ~ X@R.T + t; batched over leading dims"""
    mx = X.mean(-2, keepdims=True); my = Y.mean(-2, keepdims=True)
    H = np.swapaxes(X - mx, -1, -2) @ (Y - my)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros(H.shape); D[..., 0, 0] = 1; D[..., 1, 1] = 1; D[..., 2, 2] = d
    Rm = np.swapaxes(U @ D @ Vt, -1, -2)
    t = my[..., 0, :] - (Rm @ mx[..., 0, :, None])[..., 0]
    return Rm, t


def vscore(Y, vB, XA, vA, trA, thr=1.6):
    """number of warped B cells with an A cell within thr um and a similar volume"""
    d, j = trA.query(Y, distance_upper_bound=thr)
    ok = np.isfinite(d)
    return int((np.abs(np.log(vB[ok] / vA[j[ok]])) < 0.35).sum())


def ransac_rigid(XB, XA, dB, dA, vB, vA, rng, n_iter=100000, top=3, thr=2.0, keep=24):
    """rigid hypotheses B->A from descriptor candidates; returns list of (score, R, t), best first"""
    trd = cKDTree(dA)
    dd, ii = trd.query(dB, k=top)
    cb = np.repeat(np.arange(len(XB)), top); ca = ii.ravel()
    hyp = []
    trA = cKDTree(XA)
    for _ in range(max(1, n_iter // 10000)):
        s = rng.integers(0, len(cb), (10000, 3))
        Pb = XB[cb[s]]; Pa = XA[ca[s]]
        ok = np.ones(len(s), bool)
        for u, v in ((0, 1), (0, 2), (1, 2)):
            lb = np.linalg.norm(Pb[:, u] - Pb[:, v], axis=1); la = np.linalg.norm(Pa[:, u] - Pa[:, v], axis=1)
            ok &= (lb > 8) & (np.abs(lb - la) < 0.12 * lb + 1.0)
        Pb, Pa = Pb[ok], Pa[ok]
        if len(Pb) == 0: continue
        Rm, t = kabsch(Pb, Pa)
        sel = rng.choice(len(XB), min(len(XB), 150), replace=False)
        Y = np.einsum('nij,mj->nmi', Rm, XB[sel]) + t[:, None]
        d, _ = trA.query(Y.reshape(-1, 3), distance_upper_bound=thr)
        sc = np.isfinite(d).reshape(len(Rm), -1).sum(1)
        for j in np.argsort(-sc)[:8]:
            hyp.append((vscore(XB @ Rm[j].T + t[j], vB, XA, vA, trA, thr), Rm[j], t[j]))
    hyp.sort(key=lambda h: -h[0])
    return hyp[:keep] or [(0, np.eye(3), XA.mean(0) - XB.mean(0))]


def mutual(Y, XA, trA, thr, vB=None, vA=None):
    d, j = trA.query(Y, distance_upper_bound=thr)
    ok = np.isfinite(d)
    ib = np.where(ok)[0]; ja = j[ok]
    # mutual: A's nearest warped B is this one
    trY = cKDTree(Y)
    _, back = trY.query(XA[ja])
    m = back == ib
    ib, ja = ib[m], ja[m]
    if vB is not None:
        r = np.log(vB[ib] / vA[ja]); m = np.abs(r - np.median(r)) < 0.7
        ib, ja = ib[m], ja[m]
    return ib, ja


class Transform:
    """B (um) -> A (um): affine + Gaussian-kernel smoothed residual field"""
    def __init__(self, M, t):
        self.M, self.t = M, t; self.ctrl = None
    def affine(self, X): return X @ self.M.T + self.t
    def set_residual(self, ctrl, res, sigma):
        self.ctrl, self.res, self.sigma = ctrl, res, sigma; self.tree = cKDTree(ctrl)
    def resid(self, X, chunk=20000):
        out = np.zeros_like(X)
        if self.ctrl is None or len(self.ctrl) < 8: return out
        k = min(24, len(self.ctrl))
        for s in range(0, len(X), chunk):
            d, i = self.tree.query(X[s:s + chunk], k=k)
            w = np.exp(-0.5 * (d / self.sigma) ** 2) + 1e-9
            # shrink towards zero far from control points
            out[s:s + chunk] = (w[..., None] * self.res[i]).sum(1) / (w.sum(1)[:, None] + 0.05)
        return out
    def __call__(self, X): return self.affine(X) + self.resid(X)


def affine_icp(T, XA, XB, vA, vB, trA, thrs=(3, 2.5, 2.5, 2.5)):
    for thr in thrs:
        ib, ja = mutual(T.affine(XB), XA, trA, thr, vB, vA)
        if len(ib) < 12: break
        Xh = np.c_[XB[ib], np.ones(len(ib))]
        sol, *_ = np.linalg.lstsq(Xh, XA[ja], rcond=None)
        M = sol[:3].T
        sv = np.linalg.svd(M, compute_uv=False)
        if sv.max() > 1.5 or sv.min() < 0.6 or np.linalg.det(M) < 0: break     # implausible growth
        T.M, T.t = M, sol[3]
    return T


def register1(A, B, seed=0, keep=40):
    rng = np.random.default_rng(seed)
    sa = A['lay'] <= 3; sb = B['lay'] <= 3
    hyps = ransac_rigid(B['cen'][sb], A['cen'][sa], descriptors(B['cen'][sb], B['vol'][sb]),
                        descriptors(A['cen'][sa], A['vol'][sa]), B['vol'][sb], A['vol'][sa], rng, thr=3.0, keep=keep)
    XA, XB = A['cen'], B['cen']; vA, vB = A['vol'], B['vol']
    trA = cKDTree(XA)
    best = None; scores = []
    for n0, Rm, t in hyps:
        T = Transform(Rm, t)
        for thr in (4, 3, 3):
            ib, ja = mutual(T.affine(XB), XA, trA, thr, vB, vA)
            if len(ib) >= 4: T.M, T.t = kabsch(XB[ib], XA[ja])
        T = affine_icp(T, XA, XB, vA, vB, trA)
        sc = vscore(T.affine(XB), vB, XA, vA, trA)
        scores.append(sc)
        if best is None or sc > best[0]: best = (sc, T, n0)
    sc, T, n0 = best
    n_rigid = sc
    srt = sorted(scores)[::-1]
    for thr, sig in ((2.5, 15), (2.2, 10), (2.0, 8), (2.0, 8)):
        ib, ja = mutual(T(XB), XA, trA, thr, vB, vA)
        if len(ib) < 12: break
        T.set_residual(XB[ib], XA[ja] - T.affine(XB[ib]), sig)
    ib, ja = mutual(T(XB), XA, trA, 2.0, vB, vA)
    info = dict(ransac_inliers=int(n0), n_rigid=int(n_rigid), rigid_scores=srt, n_final=int(len(ib)), nA=len(XA), nB=len(XB),
                anchors=np.c_[B['labs'][ib], A['labs'][ja]])
    return T, info


def overlap(pl, a, b, T):
    segA = tifffile.imread(R / pl / f'T{a}/seg.tif'); segB = tifffile.imread(R / pl / f'T{b}/seg.tif')
    st = int(round(SP[0] / SP[1])); iso = float(SP[0])
    sb = segB[:, ::st, ::st]                    # isotropic grid
    shp = sb.shape
    g = 8                                       # coarse grid for the displacement field (4 um)
    cz, cy, cx = [np.arange(0, n + g, g) for n in shp]
    C = np.stack(np.meshgrid(cz, cy, cx, indexing='ij'), -1).reshape(-1, 3) * iso
    D = (T(C) - C).reshape(len(cz), len(cy), len(cx), 3)
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in shp], indexing='ij')
    q = np.stack([zz / g, yy / g, xx / g])
    pos = np.empty((3,) + shp, np.float32)
    for k, base in enumerate((zz, yy, xx)):
        pos[k] = base * iso + ndimage.map_coordinates(D[..., k], q, order=1)
    idx = [np.rint(pos[k] / SP[k]).astype(np.int64) for k in range(3)]
    ok = np.ones(shp, bool)
    for k in range(3): ok &= (idx[k] >= 0) & (idx[k] < segA.shape[k])
    la = np.zeros(shp, np.int64)
    la[ok] = segA[idx[0][ok], idx[1][ok], idx[2][ok]]
    inside = ok
    m = sb > 0
    key = la[m] * 1000000 + sb[m].astype(np.int64)
    u, c = np.unique(key, return_counts=True)
    tab = np.c_[u // 1000000, u % 1000000, c]     # A label (0 = background/outside), B label, voxels (0.125 um3)
    # voxels of each B cell falling outside A's volume
    out = np.bincount(sb[m & ~inside].astype(np.int64), minlength=int(sb.max()) + 1)
    volB = np.bincount(sb[m].astype(np.int64), minlength=int(sb.max()) + 1)
    # A volumes on a comparable grid
    volA = np.bincount(segA[:, ::st, ::st].ravel().astype(np.int64))
    return tab, out, volB, volA


def refine_from_overlap(T, tab, A, B, sigma=6.0):
    """second pass: every A cell with a clear set of overlapping B cells becomes a control point
    (A centroid  <->  volume-weighted centroid of its B cells), which also covers dividing cells"""
    t = tab[tab[:, 0] > 0]
    nb = int(max(t[:, 1].max(), B['labs'].max())) + 1
    cov = np.bincount(t[:, 1], weights=t[:, 2], minlength=nb)
    order = np.argsort(-t[:, 2]); seen = set(); fam = {}
    for a, b, c in t[order]:
        if b in seen: continue
        seen.add(b)
        if c / cov[b] >= 0.6: fam.setdefault(int(a), []).append(int(b))
    ia = {int(l): i for i, l in enumerate(A['labs'])}; ib = {int(l): i for i, l in enumerate(B['labs'])}
    XB = []; XA = []
    for a, ds in fam.items():
        if a not in ia or any(d not in ib for d in ds): continue
        j = [ib[d] for d in ds]; w = B['vol'][j]
        r = w.sum() / A['vol'][ia[a]]
        XB.append((B['cen'][j] * w[:, None]).sum(0) / w.sum()); XA.append(A['cen'][ia[a]])
    XB = np.array(XB); XA = np.array(XA)
    if len(XB) < 20: return T, 0
    # drop control points that disagree with their neighbourhood
    res = XA - T(XB)
    tr = cKDTree(XB); _, nn = tr.query(XB, k=min(9, len(XB)))
    dev = np.linalg.norm(res - np.median(res[nn], 1), axis=1)
    ok = dev < 2.0
    XB, XA = XB[ok], XA[ok]
    Xh = np.c_[XB, np.ones(len(XB))]
    sol, *_ = np.linalg.lstsq(Xh, XA, rcond=None)
    T2 = Transform(sol[:3].T, sol[3])
    T2.set_residual(XB, XA - T2.affine(XB), sigma)
    return T2, len(XB)
