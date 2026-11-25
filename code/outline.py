from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import wgen
from wallpts import wall_points


def plane(n):
    e1 = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1); return e1, np.cross(n, e1)


class Walls:
    def __init__(self, seq, t):
        wp = wall_points(seq, t); self.P = wp['P'].astype(np.float64); self.a = wp['a'].astype(np.int64); self.b = wp['b'].astype(np.int64); self.idx = {}
        for lab_arr in (self.a, self.b):
            o = np.argsort(lab_arr, kind='stable'); u, s = np.unique(lab_arr[o], return_index=True); e = np.r_[s[1:], len(o)]
            for l, s_, e_ in zip(u, s, e): self.idx.setdefault(int(l), []).append(o[s_:e_])
        self.idx = {l: np.concatenate(v) for l, v in self.idx.items()}
        D = wgen.load(seq, t); self.C = D['cen'][D['lay'] == 1]; self.tr = cKDTree(self.C); self.cen = dict(zip(D['labs'].tolist(), D['cen'])); self.nb = {}
        for l, j in self.idx.items(): self.nb[l] = set(np.unique(np.r_[self.a[j], self.b[j]]).tolist()) - {l}
    def outline(self, cells):
        """outer wall points of a set of cells (walls to layer-1 cells outside the set) and the neighbour label of every point"""
        cs = set(cells); j = np.unique(np.concatenate([self.idx[c] for c in cells if c in self.idx] or [np.zeros(0, np.int64)]))
        ina = np.isin(self.a[j], list(cs)); inb = np.isin(self.b[j], list(cs)); k = j[ina ^ inb]
        return self.P[k], np.where(np.isin(self.a[k], list(cs)), self.b[k], self.a[k])
