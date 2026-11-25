from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
import itertools, time
import wtrack as W, wgen
from outline import plane
from prog import MLS
from wallage_data import normal_at
from vertices import vertices
from grow import clone_cen
_G={}
ISO=float(W.SP[0])


class Warp:
    def __init__(self, PA, PB, NA, NB, sigma, lam=0.05, h=5.0, k=40):
        XA = np.r_[PA, PA + h * NA, PA - h * NA]; XB = np.r_[PB, PB + h * NB, PB - h * NB]
        self.f = MLS(XA, XB, sigma, lam, k=k); self.b = MLS(XB, XA, sigma, lam, k=k)
    def fwd(self, X): return self.f(X)
    def __call__(self, X): return self.b(X)


def lock_cell(job):
    """returns (m, n inliers, k, [(va, vb)], rms) or None"""
    m, R, tol, strict, cmin = job; VA, VB, trVB, F, cenA, trA, CA, trB, CB = (_G[x] for x in ('VA', 'VB', 'trVB', 'F', 'cenA', 'trA', 'CA', 'trB', 'CB')); lock = _G['lock']
    vs = VA['of'][m]; k = len(vs)
    if k < 4: return (('fail', 'k<4', None) if _G.get('debug') else None)
    PA = VA['pos'][vs]; cm = cenA[m]; q = F(np.r_[cm[None], PA]); c = q[0]; q = q[1:]
    EA = np.stack(plane(normal_at(trA, CA, cm, 35.0)), 1); EB = np.stack(plane(normal_at(trB, CB, c, 35.0)), 1)
    u = (PA - cm) @ EA; w0 = (q - c) @ EB
    J0 = np.linalg.lstsq(np.c_[u, np.ones(k)], w0, rcond=None)[0][:2].T
    if abs(np.linalg.det(J0)) < 1e-6: return (('fail', 'degenerate', None) if _G.get('debug') else None)
    J0i = np.linalg.inv(J0); cand = []
    for i in range(k):
        j = trVB.query_ball_point(q[i], R); j = sorted(j, key=lambda x: np.linalg.norm(VB['pos'][x] - q[i]))[:5]
        if vs[i] in lock: j = [lock[vs[i]]]                     # a vertex locked by the neighbours is fixed
        cand.append((np.array(j, int), (VB['pos'][j] - c) @ EB if j else np.zeros((0, 2))))
    has = [i for i in range(k) if len(cand[i][0])]
    if len(has) < 4: return (('fail', 'few candidates', None) if _G.get('debug') else None)
    need = (k if k <= 5 else int(np.ceil(0.8 * k))) if strict else max(4, int(np.ceil(_G.get('needfrac', 0.7) * k)))
    def area(t): a, b, c_ = u[list(t)]; return abs((b[0] - a[0]) * (c_[1] - a[1]) - (b[1] - a[1]) * (c_[0] - a[0]))
    triples = sorted(itertools.combinations(has, 3), key=lambda t: -area(t))[:3]; best = None
    for tr3 in triples:
        Um = np.c_[u[list(tr3)], np.ones(3)]
        if abs(np.linalg.det(Um)) < 1e-6: continue
        Ui = np.linalg.inv(Um)
        for combo in itertools.product(*[range(len(cand[i][0])) for i in tr3]):
            Wm = np.array([cand[i][1][j] for i, j in zip(tr3, combo)]); sol = Ui @ Wm; M = sol[:2].T
            sv = np.linalg.svd(M @ J0i, compute_uv=False)
            if sv[0] > 1.5 or sv[1] < 0.67: continue
            for _ in range(2):                                  # count, refit on inliers, recount
                pred = np.c_[u, np.ones(k)] @ sol; pairs = []; err = []
                for i in has:
                    d = np.linalg.norm(cand[i][1] - pred[i], axis=1); j = int(np.argmin(d))
                    if d[j] < tol: pairs.append((i, j)); err.append(d[j])
                if len(pairs) < 3: break
                sol = np.linalg.lstsq(np.c_[u[[p[0] for p in pairs]], np.ones(len(pairs))], np.array([cand[i][1][j] for i, j in pairs]), rcond=None)[0]
            if len(pairs) >= 3 and (best is None or (len(pairs), -np.mean(err)) > (best[0], -best[2])): best = (len(pairs), pairs, float(np.mean(err)), sol)
    if best is None or best[0] < need: return (('fail', 'few inliers', (k, len(has), best[0] if best else 0, need)) if _G.get('debug') else None)
    # clone test: T_b cells whose centroid maps back into m form the clone; its outline must coincide with the mapped outline of m
    WA, WB, SA, labB = _G['WA'], _G['WB'], _G['SA'], _G['labB']; sol = best[3]; M = sol[:2].T; t = sol[2]
    try:
        Mi = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return (('fail', 'degenerate', None) if _G.get('debug') else None)
    jm = WA.idx.get(m)
    if jm is None: return (('fail', 'no walls', None) if _G.get('debug') else None)
    P = WA.P[jm[:: max(1, len(jm) // 250)]]; Uo = (P - cm) @ EA; Y = c + (Uo @ M.T + t) @ EB.T
    rad = np.linalg.norm(Uo @ M.T + t, axis=1).max() + 6.0; jb = trB.query_ball_point(c, rad)
    if not jb: return (('fail', 'no B cells', None) if _G.get('debug') else None)
    wb = (CB[jb] - c) @ EB; xa = cm + ((wb - t) @ Mi.T) @ EA.T; I = np.rint(xa / ISO).astype(np.int64); ok = np.ones(len(I), bool)
    for q_ in range(3): ok &= (I[:, q_] >= 0) & (I[:, q_] < SA.shape[q_]); I[:, q_] = np.clip(I[:, q_], 0, SA.shape[q_] - 1)
    inside = (SA[I[:, 0], I[:, 1], I[:, 2]] == m) & ok; clone = [int(labB[x]) for x, s_ in zip(jb, inside) if s_]
    if not clone: return (('fail', 'empty clone', None) if _G.get('debug') else None)
    PC, _ = WB.outline(clone)
    if len(PC) < 10: return (('fail', 'short outline', None) if _G.get('debug') else None)
    Y2 = Uo @ M.T + t; PC2 = (PC - c) @ EB                      # compare in the sheet plane (wall points span the wall height)
    f1 = float((cKDTree(PC2).query(Y2, distance_upper_bound=3.0)[0] < 1.5).mean()); f2 = float((cKDTree(Y2).query(PC2, distance_upper_bound=3.0)[0] < 1.5).mean())
    if min(f1, f2) < cmin: return (('fail', 'outline', (round(f1, 2), round(f2, 2), len(clone), best[0], k)) if _G.get('debug') else None)
    cs = set(clone); pairs = [(i_, j_) for i_, j_ in best[1] if sum(int(x) in cs for x in VB['cells'][cand[i_][0][j_]]) == 1]
    if len(pairs) < 3: return (('fail', 'few outer vertices', None) if _G.get('debug') else None)
    return m, len(pairs), k, [(int(vs[i_]), int(cand[i_][0][j_])) for i_, j_ in pairs], best[2], clone, min(f1, f2)


def orient(N, ref_pts, ref_N, pts):
    j = cKDTree(ref_pts).query(pts)[1]; return N * np.where((N * ref_N[j]).sum(1, keepdims=True) < 0, -1.0, 1.0)


def run(seq, a, seedA, seedB, R=4.0, tol=1.5, sg=1.0, iters=120, log=print, Gt=None, workers=40, reach=2.5, minsup=1, R0=6.0, strict=False, cmin=0.85, keep=True, seed_fam=None, cfloor=0.7, cstep=0.05):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    b = a + 1; A, B = wgen.load(seq, a), wgen.load(seq, b); u = np.cbrt(np.median(A['vol'][A['lay'] == 1]))
    VA, VB = vertices(seq, a), vertices(seq, b); CA = A['cen'][A['lay'] == 1]; CB = B['cen'][B['lay'] == 1]; trA, trB = cKDTree(CA), cKDTree(CB)
    cenA = dict(zip(A['labs'].tolist(), A['cen'])); cells = sorted(VA['of']); ccen = np.array([cenA[m] for m in cells]); trC = cKDTree(ccen)
    nA = lambda X: np.array([normal_at(trA, CA, x, 35.0) for x in X]).reshape(-1, 3); nB = lambda X: np.array([normal_at(trB, CB, x, 35.0) for x in X]).reshape(-1, 3)
    NAs = nA(seedA); NAs *= np.where(NAs[:, :1] < 0, -1.0, 1.0); NBs = nB(seedB)
    J = np.linalg.lstsq(seedA - seedA.mean(0), seedB - seedB.mean(0), rcond=None)[0].T
    for i in range(len(NBs)):
        e1, e2 = plane(NAs[i]); s = np.sign(np.linalg.det(np.stack([J @ e1, J @ e2, NBs[i]], 1)) * np.linalg.det(np.stack([e1, e2, NAs[i]], 1))); NBs[i] *= s if s != 0 else 1.0
    par = {d: m for m, ds in Gt.items() for d in ds} if Gt else None
    def precision(lk):
        ok = n = 0
        for va, vb in lk.items():
            ca = VA['cells'][va]; cb_ = VB['cells'][vb]
            if all(int(x) in Gt for x in ca) and all(int(x) in par for x in cb_): n += 1; ok += sorted(par[int(x)] for x in cb_) == sorted(int(x) for x in ca)
        return ok / max(n, 1), n
    from outline import Walls
    from wallpts import wall_points
    from seedverify import snapped
    WA, WB = Walls(seq, a), Walls(seq, b); SA = snapped(seq, a, 6.0); labB = B['labs'][B['lay'] == 1]; clones = {}
    lock = {}; hist = []; stall = 0; PA, PB, NA_, NB_ = seedA, seedB, NAs, NBs
    if seed_fam:                                               # exact locks: T_a vertex of three seed mothers <-> the T_b vertex where their three clones meet
        sp = {d: m for m, ds in seed_fam.items() for d in ds}; tb = {}
        for j, cb_ in enumerate(VB['cells']):
            if all(int(x) in sp for x in cb_):
                kk = tuple(sorted(sp[int(x)] for x in cb_))
                if len(set(kk)) == 3: tb.setdefault(kk, []).append(j)
        for i, ca in enumerate(VA['cells']):
            kk = tuple(int(x) for x in ca)
            if kk in tb and len(tb[kk]) == 1: lock[i] = tb[kk][0]
        log(f'seed families {len(seed_fam)} -> {len(lock)} exact vertex locks')
        if len(lock) >= 6:
            va = np.array(sorted(lock)); PA = np.r_[VA['pos'][va], seedA]; PB = np.r_[VB['pos'][[lock[v] for v in va]], seedB]
            NA_ = orient(nA(PA), seedA, NAs, PA); NB_ = orient(nB(PB), seedB, NBs, PB)
    for it in range(iters):
        t0 = time.time(); Wp = Warp(PA, PB, NA_, NB_, sg * u)
        act = set();
        for j in trC.query_ball_point(PA, reach * u): act.update(j)
        act = [cells[j] for j in sorted(act)]
        _G.update(VA=VA, VB=VB, trVB=cKDTree(VB['pos']), F=Wp.fwd, cenA=cenA, trA=trA, CA=CA, trB=trB, CB=CB, lock=lock, WA=WA, WB=WB, SA=SA, labB=labB)
        Rit = R0 if len(lock) < 6 else R; cmin_it = cmin
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as ex: raw = list(ex.map(lock_cell, [(m, Rit, tol, strict, cmin_it) for m in act], chunksize=8)); _G['fails'] = [r for r in raw if r and r[0] == 'fail']; res = [r for r in raw if r and r[0] != 'fail']
        votes = {}
        for m, ninl, k, pairs, err, clone, cv in res:
            if m not in clones or cv >= clones[m][1]: clones[m] = (clone, cv)
            for p in pairs: votes[p] = votes.get(p, 0) + 1
        ms = minsup if len(lock) >= 6 else 1
        bya = {};
        for (va, vb), s in votes.items():
            if s >= ms and (va not in bya or s > bya[va][1]): bya[va] = (vb, s)
        byb = {}
        for va, (vb, s) in bya.items():
            if vb not in byb or s > byb[vb][1]: byb[vb] = (va, s)
        new = {va: vb for vb, (va, s) in byb.items()}
        if keep:                                                # earlier locks stay unless a new vote claims the same vertex
            usedb = set(new.values())
            for va, vb in lock.items():
                if va not in new and vb not in usedb: new[va] = vb
        msg = f'iter {it}: active cells {len(act)}, cells with an affine vertex fit {len(res)}, locked vertex pairs {len(new)} (was {len(lock)})'
        if Gt is not None: pr, n = precision(new); msg += f' | diag: lock precision {pr:.3f} (n {n})'
        log(msg + f'  [{time.time() - t0:.0f}s]')
        grew = len(new) > len(lock); stall = 0 if grew else stall + 1; lock = new; hist.append((len(act), len(res), len(new)))
        if len(lock) >= 6:
            va = np.array(sorted(lock)); PAo, PBo, NAo, NBo = PA, PB, NA_, NB_
            PA = VA['pos'][va]; PB = VB['pos'][[lock[v] for v in va]]
            if len(lock) < 3 * len(seedA): PA = np.r_[PA, seedA]; PB = np.r_[PB, seedB]
            NA_ = orient(nA(PA), PAo, NAo, PA); NB_ = orient(nB(PB), PBo, NBo, PB)
        if stall >= 4:
            if cmin - cstep < cfloor - 1e-9: break
            cmin -= cstep; stall = 0; log(f'   -- stalled: outline threshold lowered to {cmin:.2f}')
    Wp = Warp(PA, PB, NA_, NB_, sg * u); cells_ok = {m for m in cells if sum(v in lock for v in VA['of'][m]) >= 3}
    return dict(lock=lock, PA=PA, PB=PB, NA=NA_, NB=NB_, sigma=sg * u, cells=cells_ok, hist=hist, clones=clones), Wp
