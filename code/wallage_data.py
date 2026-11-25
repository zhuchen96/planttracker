from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import tifffile
import wtrack as W, wgen


def l1seg(seq, t):
    A = wgen.load(seq, t); st = int(round(W.SP[0] / W.SP[1]))
    seg = tifffile.imread(wgen.G / seq / f'T{t}/seg.tif')[:, ::st, ::st].astype(np.int32)
    lut = np.zeros(int(max(seg.max(), A['labs'].max())) + 1, bool); lut[A['labs'][A['lay'] == 1]] = True
    return np.where(lut[seg], seg, 0), A


def normal_at(tr, C, p, r=40.0):
    j = tr.query_ball_point(p, r)
    if len(j) < 5: j = tr.query(p, 6)[1]
    X = C[j] - C[j].mean(0); return np.linalg.svd(X, full_matrices=False)[2][-1]
