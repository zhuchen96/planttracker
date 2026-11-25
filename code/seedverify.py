from pathlib import Path
import pickle
import numpy as np
from scipy import ndimage
import wtrack as W, wgen
from wallage_data import l1seg
ISO=float(W.SP[0])


def snapped(seq, t, R=8.0):
    f = wgen.OUT / '_snapB' / f'{seq}_T{t}_R{R:g}.npy'
    if f.exists(): return np.load(f)
    seg, D = l1seg(seq, t); d, ind = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    L = seg[ind[0], ind[1], ind[2]]; L[d * ISO > R] = 0; f.parent.mkdir(exist_ok=True); np.save(f, L.astype(np.int32)); return L
