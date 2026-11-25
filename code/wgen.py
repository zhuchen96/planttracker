from pathlib import Path
import pickle
import numpy as np
import wtrack as W
G=OUT=None
ISO=(1.,.9,.8,.7,.6,.5,.42)
ZS=(1.,.8,1.25)


def load(seq, t, unit=None):
    P = pickle.load(open(G / seq / f'T{t}/props.pkl', 'rb'))
    labs = np.asarray(P['labels']).astype(int)
    cen = np.asarray(P['cen'])[labs] * W.SP; vol = np.asarray(P['vol'])[labs] * W.SP.prod()
    lay = np.array([P['lay'].get(int(l), 99) for l in labs])
    return dict(labs=labs, cen=cen, vol=vol, lay=lay)


def register(A, B, mirror=False):
    u = np.cbrt(np.median(A['vol'])) / 5.5
    An = dict(A, cen=A['cen'] / u, vol=A['vol'] / u ** 3)
    best = None; tried = {}
    for s in ISO:
        for zs in ZS:
            sc = np.array([s * zs, s, s * (-1 if mirror else 1)])
            Bn = dict(B, cen=B['cen'] / u * sc, vol=B['vol'] / u ** 3 * s ** 3 * zs)
            T, info = W.register1(An, Bn)
            tried[(s, zs)] = info['n_rigid']
            if best is None or info['n_rigid'] > best[2]['n_rigid']: best = (sc, T, info, (s, zs))
    sc, T, info, key = best
    # fold normalisation and scale into one transform acting on real um
    T2 = W.Transform(T.M @ np.diag(sc), T.t * u)
    if T.ctrl is not None: T2.set_residual(T.ctrl / sc * u, T.res * u, T.sigma * u)
    info.update(scale=key, tried=tried, unit=float(u), mirror=mirror)
    return T2, info
