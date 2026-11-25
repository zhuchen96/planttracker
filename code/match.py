import numpy as np
from scipy import ndimage
from bm import bwd_coords
MAXLAYER=2


def overlaps(segA, segB, W, ds=2):
    """map every ds-th voxel of B back into A; count (B cell, A cell) pairs.
    returns dict: cnt[(c, m)] in B-grid voxels, and per-B-cell totals."""
    P = bwd_coords(W, segB.shape, ds)                    # (3, *grid) A coordinates
    labB = segB[::ds, ::ds, ::ds]
    inside = np.all([(P[k] >= 0) & (P[k] <= segA.shape[k] - 1) for k in range(3)], axis=0)
    labA = np.zeros(labB.shape, segA.dtype)
    labA[inside] = segA[tuple(np.round(P[k][inside]).astype(int) for k in range(3))]
    m = labB > 0
    code = labB[m].astype(np.int64) * 100000 + labA[m]
    k, c = np.unique(code, return_counts=True)
    cnt = {}
    for kk, cc in zip(k.tolist(), c.tolist()):
        cnt[divmod(kk, 100000)] = cc
    # how much of each A cell's warped volume lands in B tissue / B background / outside B field of view:
    # forward: map A voxels into B
    return cnt


def build_pairs(FA, FB, cnt, W, topk=3):
    """candidate (c, m) pairs: for every B cell, its top-k A cells by overlap. returns rows of features"""
    byc = {}
    msum = {}
    for (c, m), n in cnt.items():
        byc.setdefault(c, []).append((n, m))
        if m: msum[m] = msum.get(m, 0) + n
    rows = []
    for c, lst in byc.items():
        tot = sum(n for n, _ in lst)
        lst = sorted([x for x in lst if x[1] > 0], reverse=True)[:topk]
        for rank, (n, m) in enumerate(lst):
            rows.append((c, m, rank, n / tot, n / max(msum.get(m, 1), 1), tot))
    if not rows: return np.zeros((0, 2), int), np.zeros((0, 0))
    rows = np.array(rows, float)
    C = rows[:, 0].astype(int); M = rows[:, 1].astype(int)
    # geometry
    detJ = abs(np.linalg.det(W.fa[:3]))
    wc = W.fwd(FA['cen'][M])
    rB = (3 * FB['vol'][C] / (4 * np.pi)) ** (1 / 3)
    rM = (3 * FA['vol'][M] * detJ / (4 * np.pi)) ** (1 / 3)
    dist = np.linalg.norm((wc - FB['cen'][C]) * [2, 1, 1], axis=1) / rM
    vr = FB['vol'][C] / (FA['vol'][M] * detJ)
    layA = np.array([FA['lay'].get(m, 9) for m in M]); layB = np.array([FB['lay'].get(c, 9) for c in C])
    # second-best margin for the B cell
    margin = np.zeros(len(rows))
    for i in range(len(rows)):
        pass
    order = np.lexsort((-rows[:, 3], C))
    first = {}
    for i in order:
        c = C[i]
        if c not in first: first[c] = rows[i, 3]
    best = np.array([first[c] for c in C])
    second = {}
    for i in order:
        c = C[i]
        if rows[i, 3] < first[c] and c not in second: second[c] = rows[i, 3]
    sec = np.array([second.get(c, 0.0) for c in C])
    border = np.array([c in FB['border'] for c in C], float)
    X = np.c_[rows[:, 2], rows[:, 3], rows[:, 4], dist, np.log(vr), layA, layB, best - sec, rows[:, 3] - best,
              np.log(FB['vol'][C]), np.log(rows[:, 5] / np.maximum(FB['vol'][C] / 8, 1)), border]
    return np.c_[C, M], X
