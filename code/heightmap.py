"""Height-map initialization for training registrations."""
import numpy as np
from scipy import ndimage
from scipy.fft import rfft2, irfft2, next_fast_len

DS = 4

def heightmap(seg, ds=DS):
    tis = seg > 0; has = tis.any(0); z = np.where(has, tis.argmax(0), 0).astype(float)
    h = z[::ds, ::ds]; m = has[::ds, ::ds].astype(float)
    m = ndimage.binary_erosion(m, iterations=1).astype(float)   # drop ragged edges
    return h, m

def corr(a, b, s):
    """cross-correlation of fixed a with moving b for all shifts (same convention as masked_ncc)"""
    return np.round(irfft2(rfft2(a, s) * rfft2(b[::-1, ::-1], s), s))

def masked_ncc(f, fm, g, gm, min_overlap):
    """Padfield masked NCC of fixed f (mask fm) and moving g (mask gm) for all integer shifts. returns (ncc map, overlap map, shape)"""
    s = [next_fast_len(f.shape[i] + g.shape[i] - 1) for i in range(2)]
    F = lambda x: rfft2(x, s)
    rot = lambda x: x[::-1, ::-1]
    f = f * fm; g = g * gm
    Fm, Gm = F(fm), F(rot(gm))
    ov = np.round(irfft2(Fm * Gm, s)); ov = np.maximum(ov, 1)
    Ff, Fg = F(f), F(rot(g))
    fg = irfft2(Ff * Fg, s)
    fs = irfft2(Ff * Gm, s); gs = irfft2(Fm * Fg, s)
    f2 = irfft2(F(f * f) * Gm, s); g2 = irfft2(Fm * F(rot(g * g)), s)
    num = fg - fs * gs / ov
    den = np.sqrt(np.maximum(f2 - fs ** 2 / ov, 0) * np.maximum(g2 - gs ** 2 / ov, 0))
    ncc = np.where((ov >= min_overlap) & (den > 1e-6), num / np.maximum(den, 1e-6), -1)
    return ncc, ov

def warp2d(img, A, t, out_shape, order=1):
    """sample img at A @ p + t for every output pixel p (A maps output->input coords)"""
    return ndimage.affine_transform(img, A, offset=t, output_shape=out_shape, order=order, cval=0)

def surface_labels(seg):
    """2D map of the first (outermost, low-z) cell in each column, and its depth"""
    tis = seg > 0; has = tis.any(0); z = tis.argmax(0)
    lab = np.take_along_axis(seg, z[None], 0)[0] * has
    return lab, np.where(has, z, 0).astype(float), has

def search_topk(hA, mA, hB, mB, scales, angles, k=30, min_frac=0.25, wn=1.0, wd=1.0):
    """like search, but return the k best (s, ang) candidates with their best shift (full-res L, t)"""
    out = []
    nA = mA.sum()
    for s in scales:
        for ang in angles:
            th = np.radians(ang); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); L = s * R
            corners = np.array([[0, 0], [hA.shape[0], 0], [0, hA.shape[1]], hA.shape]) @ L.T
            lo = corners.min(0); hi = corners.max(0); shp = tuple(np.ceil(hi - lo).astype(int) + 1)
            Ainv = np.linalg.inv(L)
            wh = warp2d(hA, Ainv, Ainv @ lo, shp); wm = warp2d(mA, Ainv, Ainv @ lo, shp, order=0)
            wf = warp2d(np.ones_like(mA), Ainv, Ainv @ lo, shp, order=0)
            ncc, ov = masked_ncc(hB, mB, wh, wm, min_overlap=min_frac * nA * s * s)
            sz = [next_fast_len(hB.shape[i] + wh.shape[i] - 1) for i in range(2)]
            dice = 2 * ov / np.maximum(corr(np.ones_like(mB), wm, sz) + corr(mB, wf, sz), 1)
            J = np.where(ncc > -1, wn * ncc + wd * dice, -2)
            # keep the 3 best separated shifts per (s, ang)
            Jc = J.copy()
            for _ in range(3):
                i = np.unravel_index(Jc.argmax(), Jc.shape)
                if Jc[i] <= -1: break
                d = np.array(i) - (np.array(wh.shape) - 1)
                out.append((Jc[i], s, ang, L, (d - lo) * DS))
                Jc[max(0, i[0]-6):i[0]+7, max(0, i[1]-6):i[1]+7] = -2
    out.sort(key=lambda x: -x[0])
    return out[:k]

def lift3d_full(depthA, hasA, depthB, hasB, L, t):
    """z mapping from full-res depth maps: zB = a zA + cy yB + cx xB + b on overlapping pixels"""
    yy, xx = np.nonzero(hasA)
    q = np.c_[yy, xx] @ L.T + t
    ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < hasB.shape[0] - 1) & (q[:, 1] < hasB.shape[1] - 1)
    qi = np.round(q[ok]).astype(int); inB = hasB[qi[:, 0], qi[:, 1]]
    zA = depthA[yy[ok][inB], xx[ok][inB]]; zB = depthB[qi[inB, 0], qi[inB, 1]]
    X = np.c_[zA, qi[inB, 0], qi[inB, 1], np.ones(inB.sum())]
    coef, *_ = np.linalg.lstsq(X, zB, rcond=None)
    for _ in range(2):
        r = np.abs(X @ coef - zB); k = r < np.quantile(r, 0.8)
        coef, *_ = np.linalg.lstsq(X[k], zB[k], rcond=None)
    a, cy, cx, b = coef
    def warp(P):
        P = np.asarray(P, float); yx = P[:, 1:] @ L.T + t
        return np.c_[a * P[:, 0] + cy * yx[:, 0] + cx * yx[:, 1] + b, yx]
    return warp, coef

def search_topk_aniso(hA, mA, hB, mB, scales, angles, ratios=(1.0, 1.25, 1.5), dirs=(0, 45, 90, 135), k=60, **kw):
    """search_topk over anisotropic 2D linear maps L = R(ang) R(dir) diag(s*sqrt(q), s/sqrt(q)) R(-dir)"""
    out = []
    import itertools
    combos = [(1.0, 0)] + [(q, d) for q in ratios if q != 1.0 for d in dirs]
    for q, d in combos:
        ph = np.radians(d); Rd = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
        S = Rd @ np.diag([np.sqrt(q), 1 / np.sqrt(q)]) @ Rd.T
        res = _search_with_shape(hA, mA, hB, mB, scales, angles, S, k=k, **kw)
        out += [(J, s, ang, L, t, q, d) for J, s, ang, L, t in res]
    out.sort(key=lambda x: -x[0])
    return [o[:5] for o in out[:k]], out[:k]

def _search_with_shape(hA, mA, hB, mB, scales, angles, S, k=30, min_frac=0.25, wn=1.0, wd=1.0):
    out = []
    nA = mA.sum()
    for s in scales:
        for ang in angles:
            th = np.radians(ang); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); L = s * R @ S
            corners = np.array([[0, 0], [hA.shape[0], 0], [0, hA.shape[1]], hA.shape]) @ L.T
            lo = corners.min(0); hi = corners.max(0); shp = tuple(np.ceil(hi - lo).astype(int) + 1)
            Ainv = np.linalg.inv(L)
            wh = warp2d(hA, Ainv, Ainv @ lo, shp); wm = warp2d(mA, Ainv, Ainv @ lo, shp, order=0)
            wf = warp2d(np.ones_like(mA), Ainv, Ainv @ lo, shp, order=0)
            ncc, ov = masked_ncc(hB, mB, wh, wm, min_overlap=min_frac * nA * s * s)
            sz = [next_fast_len(hB.shape[i] + wh.shape[i] - 1) for i in range(2)]
            dice = 2 * ov / np.maximum(corr(np.ones_like(mB), wm, sz) + corr(mB, wf, sz), 1)
            J = np.where(ncc > -1, wn * ncc + wd * dice, -2)
            Jc = J.copy()
            for _ in range(2):
                i = np.unravel_index(Jc.argmax(), Jc.shape)
                if Jc[i] <= -1: break
                d = np.array(i) - (np.array(wh.shape) - 1)
                out.append((Jc[i], s, ang, L, (d - lo) * DS))
                Jc[max(0, i[0]-6):i[0]+7, max(0, i[1]-6):i[1]+7] = -2
    out.sort(key=lambda x: -x[0])
    return out[:k]
