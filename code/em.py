import numpy as np
from scipy.interpolate import RBFInterpolator
from bm import Warp, ZS
from match import MAXLAYER


def confident_families(CM, X, FA, FB, frac_d=0.7, cover_m=0.6, maxlayer=MAXLAYER):
    """rule-based (no labels): each B cell -> best mother if it lies mostly inside it; keep mothers whose daughters
    are connected and together cover most of the mother's warped volume"""
    fam = {}; cover = {}
    for (c, m), x in zip(CM, X):
        if x[0] == 0 and x[1] >= frac_d:            # rank 0 and mostly inside m
            fam.setdefault(m, []).append(c); cover[m] = cover.get(m, 0) + x[2]
    out = {}
    for m, ds in fam.items():
        if FA['lay'].get(m, 99) > maxlayer or cover[m] < cover_m: continue
        if len(ds) > 1:
            seen = {ds[0]}; st = [ds[0]]
            while st:
                x = st.pop()
                for y in FB['adj'].get(x, ()):
                    if y in ds and y not in seen: seen.add(y); st.append(y)
            if len(seen) != len(ds): continue
        out[m] = ds
    return out


def fit_warp(FA, FB, fams, smoothing=100.0):
    ms = sorted(fams)
    P = FA['cen'][ms]; D = np.array([np.average(FB['cen'][fams[m]], axis=0, weights=FB['vol'][fams[m]]) for m in ms])
    fa, *_ = np.linalg.lstsq(np.c_[P, np.ones(len(P))], D, rcond=None); ba, *_ = np.linalg.lstsq(np.c_[D, np.ones(len(D))], P, rcond=None)
    fr = RBFInterpolator(P * [ZS, 1, 1], D - np.c_[P, np.ones(len(P))] @ fa, kernel='thin_plate_spline', smoothing=smoothing)
    br = RBFInterpolator(D * [ZS, 1, 1], P - np.c_[D, np.ones(len(D))] @ ba, kernel='thin_plate_spline', smoothing=smoothing)
    return Warp(fa, ba, fr, br)
