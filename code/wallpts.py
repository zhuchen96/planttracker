from pathlib import Path
import pickle
import numpy as np
import tifffile
import wtrack as W, wgen


def wall_points(seq, t, layers=(1,)):
    f = wgen.OUT / '_wallpts' / f'{seq}_T{t}.pkl'
    if f.exists(): return pickle.load(open(f, 'rb'))
    A = wgen.load(seq, t); st = int(round(W.SP[0] / W.SP[1])); iso = float(W.SP[0])
    seg = tifffile.imread(wgen.G / seq / f'T{t}/seg.tif')[:, ::st, ::st].astype(np.int32)
    lut = np.zeros(int(max(seg.max(), A['labs'].max())) + 1, bool); lut[A['labs'][np.isin(A['lay'], layers)]] = True
    seg = np.where(lut[seg], seg, 0); P, LA, LB = [], [], []
    for ax in range(3):
        sa = [slice(None)] * 3; sb = [slice(None)] * 3; sa[ax] = slice(0, -1); sb[ax] = slice(1, None)
        a = seg[tuple(sa)]; b = seg[tuple(sb)]; m = (a != b) & (a > 0) & (b > 0); c = np.argwhere(m).astype(np.float32); c[:, ax] += 0.5
        P.append(c * iso); LA.append(a[m]); LB.append(b[m])
    out = dict(P=np.concatenate(P), a=np.concatenate(LA), b=np.concatenate(LB))
    # voxel points per cell (subsampled) for moments
    f.parent.mkdir(exist_ok=True); pickle.dump(out, open(f, 'wb')); return out
