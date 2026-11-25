import numpy as np


def predict(d, fmin=0.5, amin=0.15):
    """parent of a B cell = A cell with the largest overlap; fraction is taken over the part of the
    B cell that lands on any A cell (background / outside is ignored), with a floor on the absolute fraction"""
    tab = d['tab']; volB = d['volB']
    pred = {}
    t = tab[tab[:, 0] > 0]
    cov = np.bincount(t[:, 1], weights=t[:, 2], minlength=len(volB))
    order = np.argsort(-t[:, 2])
    seen = set()
    for a, b, c in t[order]:
        if b in seen: continue
        seen.add(b)
        if c / cov[b] >= fmin and c / volB[b] >= amin: pred.setdefault(int(a), []).append(int(b))
    return pred
