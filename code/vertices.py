from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import wtrack as W, wgen
from wallage_data import l1seg


def vertices(seq, t, link=3.0):
    f = wgen.OUT / '_vert' / f'{seq}_T{t}.pkl'
    if f.exists(): return pickle.load(open(f, 'rb'))
    seg, D = l1seg(seq, t); iso = float(W.SP[0])
    blk = np.stack([seg[dz:seg.shape[0] - 1 + dz, dy:seg.shape[1] - 1 + dy, dx:seg.shape[2] - 1 + dx] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)], -1)
    cand = blk.max(-1) != blk.min(-1); idx = np.argwhere(cand); v = np.sort(blk[cand], axis=1); del blk
    nd = (np.diff(v, axis=1) != 0).sum(1) + 1; has0 = v[:, 0] == 0; sel = (nd - has0) == 3; idx = idx[sel]; v = v[sel]
    tri = np.array([np.unique(r[r > 0]) for r in v]).reshape(-1, 3).astype(np.int64); key = tri[:, 0] * 10 ** 12 + tri[:, 1] * 10 ** 6 + tri[:, 2]
    o = np.argsort(key, kind='stable'); key, idx, tri = key[o], idx[o], tri[o]; uk, s = np.unique(key, return_index=True); e = np.r_[s[1:], len(key)]
    pos, cells = [], []
    for s_, e_ in zip(s, e):
        P = (idx[s_:e_] + 0.5) * iso
        if e_ - s_ == 1 or np.ptp(P, 0).max() <= link: groups = [np.arange(e_ - s_)]
        else:                                        # split into connected groups
            tr = cKDTree(P); lab = -np.ones(len(P), int); g = 0
            for i in range(len(P)):
                if lab[i] >= 0: continue
                st = [i]; lab[i] = g
                while st:
                    for j in tr.query_ball_point(P[st.pop()], link):
                        if lab[j] < 0: lab[j] = g; st.append(j)
                g += 1
            groups = [np.flatnonzero(lab == q) for q in range(g)]
        for gidx in groups: pos.append(P[gidx].mean(0)); cells.append(tri[s_])
    pos = np.array(pos); cells = np.array(cells); of = {}
    for i, c in enumerate(cells):
        for x in c: of.setdefault(int(x), []).append(i)
    out = dict(pos=pos, cells=cells, of=of); f.parent.mkdir(exist_ok=True); pickle.dump(out, open(f, 'wb')); return out
