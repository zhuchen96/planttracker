"""LambdaRank set ranker (per-mother groups) with per-mother softmax probabilities."""
import os, numpy as np, lightgbm as lgb
NJ = max(1, int(os.environ.get("OMP_NUM_THREADS", 4)))
from setmatch import competition_features

def groups(rows, y):
    ms = np.array([m for m, _ in rows]); keep = np.nonzero(y >= 0)[0]
    keep = keep[np.argsort(ms[keep], kind='stable')]
    _, cnt = np.unique(ms[keep], return_counts=True)
    return keep, cnt

def softmax_by_mother(rows, s):
    ms = np.array([m for m, _ in rows]); p = np.zeros_like(s, dtype=float)
    order = np.argsort(ms, kind='stable'); b = np.r_[0, np.nonzero(np.diff(ms[order]))[0] + 1, len(order)]
    for lo, hi in zip(b[:-1], b[1:]):
        i = order[lo:hi]; e = np.exp(s[i] - s[i].max()); p[i] = e / e.sum()
    return p

def fit_rank(sets_list, Xkey='X'):
    Xl, yl, gl = [], [], []
    for E in sets_list:
        k, cnt = groups(E['rows'], E['y']); Xl.append(E[Xkey][k]); yl.append(E['y'][k]); gl.append(cnt)
    m = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                       lambdarank_truncation_level=10, verbose=-1, random_state=0, n_jobs=NJ)
    return m.fit(np.concatenate(Xl), np.concatenate(yl), group=np.concatenate(gl))

class TwoStage:
    """stage 1 on set features; stage 2 on set + competition features (from cross-fitted stage-1 probabilities)"""
    def __init__(self, train_sets, FA_FB):
        """train_sets: list of (sets dict, transition index); FA_FB: dict index -> (FA, FB)"""
        self.m1 = fit_rank([E for E, _ in train_sets])
        idx = sorted(set(a for _, a in train_sets)); s2 = []
        # All augmentation sources of a transition have the same held-out model.
        # Fit it once per transition, retaining exactly the original training order.
        inner_models = {a: fit_rank([E for E, b in train_sets if b != a]) for a in idx}
        for E, a in train_sets:
            p = softmax_by_mother(E['rows'], inner_models[a].predict(E['X']))
            FA, FB = FA_FB[a]
            s2.append(dict(rows=E['rows'], y=E['y'], X2=competition_features(E['rows'], E['X'], p, FA, FB)))
        self.m2 = fit_rank(s2, Xkey='X2')
    def p1(self, rows, X): return softmax_by_mother(rows, self.m1.predict(X))
    def p2(self, rows, X, p1, FA, FB): return softmax_by_mother(rows, self.m2.predict(competition_features(rows, X, p1, FA, FB)))
