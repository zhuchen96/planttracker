"""Raw-image block matching used to generate training candidates."""
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from skimage.feature import match_template
from bm import Warp, ZS, bwd_coords

def affine_from_fn(fn, shape):
    g = np.stack(np.meshgrid(*[np.linspace(0, s - 1, 6) for s in shape], indexing='ij'), -1).reshape(-1, 3)
    M, *_ = np.linalg.lstsq(np.c_[g, np.ones(len(g))], fn(g), rcond=None)
    return M

def inv_affine(M):
    A = np.eye(4); A[:3, :3] = M[:3].T; A[:3, 3] = M[3]
    Ai = np.linalg.inv(A); return np.c_[Ai[:3, :3], Ai[:3, 3]].T

def resample_A_into_B(rawA, W, shapeB, ds):
    """warped A sampled on the downsampled B grid"""
    return ndimage.map_coordinates(rawA, bwd_coords(W, shapeB, ds), order=1, cval=0)

def prep(raw):
    r = np.log1p(raw.astype(np.float32)); lo, hi = np.percentile(r, [1, 99.8])
    return np.clip((r - lo) / (hi - lo), 0, 1)

def block_match(Aw, Bd, maskB, block, win, stride, min_ncc=0.2):
    """Aw, Bd: images on the same (downsampled) B grid. template = B block, search in Aw. returns centres, displacement (B->Aw), ncc"""
    cz, cy, cx = [np.arange(block[i] // 2 + win[i], Bd.shape[i] - block[i] // 2 - win[i], stride[i]) for i in range(3)]
    C, Dd, S = [], [], []
    for z in cz:
        for y in cy:
            for x in cx:
                if not maskB[z, y, x]: continue
                sl = tuple(slice(c - b // 2, c - b // 2 + b) for c, b in zip((z, y, x), block))
                T = Bd[sl]
                if T.std() < 1e-3: continue
                sr = tuple(slice(c - b // 2 - w, c - b // 2 + b + w) for c, b, w in zip((z, y, x), block, win))
                Rg = Aw[sr]
                if Rg.std() < 1e-3: continue
                ncc = match_template(Rg, T)
                i = np.unravel_index(ncc.argmax(), ncc.shape)
                if ncc[i] < min_ncc: continue
                C.append((z, y, x)); Dd.append(np.array(i) - np.array(win)); S.append(ncc[i])
    return np.array(C, float), np.array(Dd, float), np.array(S)

def robust_filter(C, Dd, k=8, thr=2.0):
    """drop displacements that disagree with their neighbours' median"""
    from scipy.spatial import cKDTree
    if len(C) < k + 1: return np.ones(len(C), bool)
    _, nn = cKDTree(C * [ZS, 1, 1]).query(C * [ZS, 1, 1], k=k + 1)
    med = np.median(Dd[nn[:, 1:]], axis=1)
    dev = np.linalg.norm(Dd - med, axis=1)
    return dev <= max(thr, np.median(dev) * 2.5)

LEVELS = [  # (downsample, block (z,y,x) in downsampled voxels, search window, stride, rbf smoothing)
    (4, (6, 10, 10), (3, 6, 6), (3, 5, 5), 3e3),
    (2, (8, 14, 14), (3, 5, 5), (4, 7, 7), 1e3),
    (2, (8, 14, 14), (2, 3, 3), (4, 7, 7), 3e2),
]

def boundary_image(seg, sigma=1.0):
    """smoothed cell-wall map from a segmentation (1 on walls between cells and at the tissue surface)"""
    b = np.zeros(seg.shape, np.float32)
    for ax in range(3):
        d = np.diff(seg, axis=ax) != 0
        sl = [slice(None)] * 3; sl[ax] = slice(1, None); b[tuple(sl)] = np.maximum(b[tuple(sl)], d)
        sl[ax] = slice(None, -1); b[tuple(sl)] = np.maximum(b[tuple(sl)], d)
    b = ndimage.gaussian_filter(b, sigma)
    return b / max(b.max(), 1e-6)

def make_image(raw, seg, mode):
    if mode == 'raw': return prep(raw)
    if mode == 'seg': return boundary_image(seg)
    if mode == 'mix': return 0.5 * prep(raw) + 0.5 * boundary_image(seg)
    raise ValueError(mode)

def refine(rawA, rawB, segB, W0, levels=LEVELS, verbose=False, mode='raw', segA=None):
    """iterative block matching; returns Warp and per-level median NCC"""
    W = W0; scores = []
    A = make_image(rawA, segA, mode); Bp = make_image(rawB, segB, mode)
    for ds, block, win, stride, smooth in levels:
        Bd = Bp[::ds, ::ds, ::ds]
        maskB = ndimage.binary_dilation(segB[::ds, ::ds, ::ds] > 0, iterations=1)
        Aw = resample_A_into_B(A, W, segB.shape, ds)
        C, Dd, S = block_match(Aw, Bd, maskB, block, win, stride)
        if len(C) < 30:
            scores.append((ds, 0.0, len(C))); continue
        keep = robust_filter(C, Dd)
        C, Dd, S = C[keep], Dd[keep], S[keep]
        pB = C * ds                                 # block centre in full-res B voxels
        pA = W.bwd((C + Dd) * ds)                   # matched location in A
        scores.append((ds, float(np.median(S)), len(C)))
        # refit: affine + TPS residual, both directions
        fa, *_ = np.linalg.lstsq(np.c_[pA, np.ones(len(pA))], pB, rcond=None)
        ba, *_ = np.linalg.lstsq(np.c_[pB, np.ones(len(pB))], pA, rcond=None)
        fr = RBFInterpolator(pA * [ZS, 1, 1], pB - np.c_[pA, np.ones(len(pA))] @ fa, kernel='thin_plate_spline', smoothing=smooth)
        br = RBFInterpolator(pB * [ZS, 1, 1], pA - np.c_[pB, np.ones(len(pB))] @ ba, kernel='thin_plate_spline', smoothing=smooth)
        W = Warp(fa, ba, fr, br)
        if verbose: print(f'   level ds={ds}: {len(C)} blocks, median ncc {np.median(S):.3f}', flush=True)
    return W, scores
