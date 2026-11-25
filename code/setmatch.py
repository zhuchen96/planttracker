"""Set matching: for each mother, enumerate connected candidate daughter sets and describe each with shape,
position, volume and neighbour-contact features. A ranker trained on other transitions picks the set."""
import numpy as np
from scipy import ndimage
from em import confident_families
from match import overlaps, build_pairs, MAXLAYER

# v4 boundary-member features: helped T2→T3 (0.756→0.799) but hurt T1→T2 (0.738→0.590) in LOTO -> off by default
MEMBER_FEATS = False

def cell_points(seg, labels, bboxes, n=300, ds=1):
    """subsampled voxel coordinates for each label"""
    pts = {}
    for l in labels:
        sl = bboxes[l - 1] if l - 1 < len(bboxes) else None
        if sl is None: continue
        P = np.argwhere(seg[sl] == l) + [s.start for s in sl]
        if len(P) > n: P = P[np.linspace(0, len(P) - 1, n).astype(int)]
        pts[l] = P.astype(float)
    return pts

def shape_desc(P):
    """aspect ratios and principal axis of a point cloud (z scaled x2)"""
    X = (P - P.mean(0)) * [2, 1, 1]
    # Tiny labels can survive nearest-neighbour resampling as one voxel. Their
    # spread is zero; sample covariance would divide by zero and produce NaNs.
    covariance = np.cov(X.T) if len(X) > 1 else np.zeros((3, 3))
    ev, evec = np.linalg.eigh(covariance + 1e-6 * np.eye(3))
    ev = np.sqrt(np.maximum(ev[::-1], 1e-6)); ax = evec[:, ::-1][:, 0]
    return ev, ax

def connected_subsets(cands, adj, maxsize):
    """all connected subsets (as frozensets) of the candidate cells up to maxsize"""
    cands = set(cands); out = set()
    frontier = [frozenset([c]) for c in cands]
    out.update(frontier)
    for size in range(2, maxsize + 1):
        new = set()
        for S in frontier:
            nb = set().union(*(adj.get(c, set()) for c in S)) & cands
            for c in nb - S: new.add(S | {c})
        if len(new) > 4000: new = set(list(new)[:4000])
        out.update(new); frontier = list(new)
    return out

def build_sets(a, segA, segB, FA, FB, W, gt_fam=None, maxsize=6, rad=1.6, anchors=None, growth=None):
    """returns rows: (m, tuple(S)), feature matrix, labels (if gt_fam), plus per-mother info"""
    mothers = [m for m in FA['labels'] if FA['lay'].get(m, 9) <= MAXLAYER]
    # anchors (unsupervised) for contact features + growth estimate
    if anchors is None:
        cnt = overlaps(segA, segB, W); CM, X = build_pairs(FA, FB, cnt, W)
        anchors = confident_families(CM, X, FA, FB, maxlayer=99)
    if growth is None:
        ratios = [FB['vol'][ds].sum() / FA['vol'][m] for m, ds in anchors.items()]
        growth = np.median(ratios) if ratios else abs(np.linalg.det(W.fa[:3]))
    detJ = abs(np.linalg.det(W.fa[:3]))
    Jinv = np.linalg.inv(W.fa[:3].T)
    ptsA = cell_points(segA, mothers, FA['bbox'], n=500)
    from scipy.spatial import cKDTree
    labsB = FB['labels']; treeB = cKDTree(FB['cen'][labsB] * [2, 1, 1])
    ptsB_cache = {}
    def ptsB(c):
        if c not in ptsB_cache:
            sl = FB['bbox'][c - 1]; P = np.argwhere(segB[sl] == c) + [s.start for s in sl]
            if len(P) > 300: P = P[np.linspace(0, len(P) - 1, 300).astype(int)]
            ptsB_cache[c] = (P.astype(float), W.bwd(P.astype(float)))
        return ptsB_cache[c]
    shpB = np.array(segB.shape) - 1; shpA = np.array(segA.shape) - 1
    rows, feats, labels, info = [], [], [], {}
    for m in mothers:
        if m not in ptsA: continue
        Q = W.fwd(ptsA[m]); q0 = Q.mean(0)
        if np.any(q0 < 0) or np.any(q0 > shpB): continue
        vexp = FA['vol'][m] * growth
        rM = (3 * FA['vol'][m] * detJ / (4 * np.pi)) ** (1 / 3)
        idx = treeB.query_ball_point(q0 * [2, 1, 1], rad * rM * 1.0 + 1e-6)
        cands = [labsB[i] for i in idx]
        # also any cell the warped mother lands in
        qi = np.clip(np.round(Q).astype(int), 0, shpB); hit = set(np.unique(segB[qi[:, 0], qi[:, 1], qi[:, 2]]).tolist()) - {0}
        cands = list(set(cands) | hit)
        if not cands: continue
        if len(cands) > 14:
            cands = sorted(cands, key=lambda c: np.linalg.norm((FB['cen'][c] - q0) * [2, 1, 1]))[:14]
        evM, axM = shape_desc(Q)
        # neighbour anchors with big contact
        nbrs = [n for n in FA['adj'].get(m, ()) if n in anchors and n != m and FA['area'].get(min(m, n) * 100000 + max(m, n), 0) > 50]
        subsets = connected_subsets(cands, FB['adj'], maxsize)
        info[m] = dict(n_cands=len(cands), n_sets=len(subsets))
        Qlab = segB[qi[:, 0], qi[:, 1], qi[:, 2]]
        # per-member evidence: fraction of each candidate cell lying inside the warped mother, and share of the
        # mother's points landing in each candidate
        pin = {}
        for c in cands:
            pa = np.clip(np.round(ptsB(c)[1]).astype(int), 0, shpA)
            pin[c] = (segA[pa[:, 0], pa[:, 1], pa[:, 2]] == m).mean()
        qshare = {c: (Qlab == c).mean() for c in cands}
        for S in subsets:
            Sl = list(S); vS = FB['vol'][Sl].sum()
            if not (0.3 * vexp <= vS <= 3.0 * vexp): continue
            cS = np.average(FB['cen'][Sl], axis=0, weights=FB['vol'][Sl])
            dc = np.linalg.norm((cS - q0) * [2, 1, 1]) / rM
            if dc > 1.5: continue
            inS = np.isin(Qlab, Sl)
            rec0 = inS.mean()                                         # mother inside S (no alignment)
            Pb = np.concatenate([ptsB(c)[0] for c in Sl]); Pa = np.concatenate([ptsB(c)[1] for c in Sl])
            pai = np.clip(np.round(Pa).astype(int), 0, shpA)
            prec0 = (segA[pai[:, 0], pai[:, 1], pai[:, 2]] == m).mean()   # S inside mother (no alignment)
            # centroid-aligned shape agreement
            d = cS - q0
            qa = np.clip(np.round(Q + d).astype(int), 0, shpB)
            rec1 = np.isin(segB[qa[:, 0], qa[:, 1], qa[:, 2]], Sl).mean()
            pa = np.clip(np.round(Pa - d @ Jinv.T).astype(int), 0, shpA)
            prec1 = (segA[pa[:, 0], pa[:, 1], pa[:, 2]] == m).mean()
            evS, axS = shape_desc(Pb)
            ang = abs(axS @ axM)
            cp = (np.mean([any(FB['adj'].get(x, set()) & set(anchors[n]) for x in Sl) for n in nbrs]) if nbrs else -1)
            # does S steal cells that belong to a neighbour anchor?
            steal = sum(any(x in anchors[n] for n in nbrs) for x in Sl) / len(Sl) if nbrs else -1
            # division-wall signature: internal walls (between members of S) vs the outer walls of S
            iA = iI = eA = eI = 0.0; imax = 0.0
            for x in Sl:
                for nb in FB['adj'].get(x, ()):
                    k = min(x, nb) * 100000 + max(x, nb); ar = FB['area'].get(k, 0); wi = FB['wall'].get(k, 0)
                    if nb in S:
                        if x < nb: iA += ar; iI += ar * wi; imax = max(imax, wi)
                    else: eA += ar; eI += ar * wi
            eMean = eI / max(eA, 1); iMean = iI / max(iA, 1) if iA else eMean
            border = (set().union(*(FB['adj'].get(x, set()) for x in Sl)) - S) & set(cands)
            pins = [pin[x] for x in Sl]
            out_pin = max([pin[x] for x in border], default=0.0)
            out_share = sum(qshare[x] for x in border)
            rows.append((m, tuple(sorted(Sl))))
            feats.append([len(Sl), np.log(vS / vexp), dc, rec0, prec0, rec1, prec1, 2 * rec1 * prec1 / max(rec1 + prec1, 1e-6),
                          np.log(evS[0] / evS[1]) - np.log(evM[0] / evM[1]), np.log(evS[1] / evS[2]) - np.log(evM[1] / evM[2]),
                          ang, cp, steal, len(nbrs), FA['lay'].get(m, 9), info[m]['n_cands'],
                          np.log(max(iMean, 1e-6) / max(eMean, 1e-6)), iA / max(iA + eA, 1), np.log(max(imax, 1e-6) / max(eMean, 1e-6)) if iA else 0.0]
                         + ([min(pins), float(np.mean(pins)), out_pin, out_share, min(pins) - out_pin] if MEMBER_FEATS else []))
            if gt_fam is not None:
                labels.append(int(m in gt_fam and sorted(gt_fam[m]) == sorted(Sl)) if m in gt_fam else -1)
    return rows, np.array(feats), np.array(labels), info, growth

SET_FEATS = ['size', 'log_vol_vs_expected', 'centroid_offset_r', 'mother_in_S', 'S_in_mother', 'mother_in_S_aligned',
             'S_in_mother_aligned', 'shape_F1_aligned', 'aspect12_diff', 'aspect23_diff', 'axis_alignment',
             'contact_preservation', 'steals_anchor_cells', 'n_anchor_nbrs', 'layer', 'n_cands', 'int_ext_wall_intensity', 'int_wall_area_frac', 'max_int_wall_vs_ext'] + (['min_member_in_mother', 'mean_member_in_mother', 'max_excluded_nbr_in_mother', 'mother_share_on_excluded_nbrs', 'member_vs_excluded_margin'] if MEMBER_FEATS else [])

def decode_sets(rows, score, mothers_all, kappa=0.0):
    """greedy: highest-scoring (mother, set) first; each B cell used once; each mother once"""
    order = np.argsort(-score)
    used_c, done = set(), {}
    for i in order:
        m, S = rows[i]
        if m in done or score[i] < kappa: continue
        if any(c in used_c for c in S): continue
        done[m] = (list(S), float(score[i])); used_c.update(S)
    pred = {}
    for m in mothers_all:
        if m in done:
            S, s = done[m]; pred[m] = dict(cls='persists' if len(S) == 1 else 'divided', daughters=sorted(S), conf=s, outfov=0.0)
        else:
            pred[m] = dict(cls='not_trackable', daughters=[], conf=0.0, outfov=0.0)
    return pred


def competition_features(rows, X, s, FA, FB):
    """stage-2 features from stage-1 scores s: margin to the mother's best set, contest by other mothers, overlap
    with neighbours' top sets, contact preservation with neighbours' top sets"""
    best_m, second_m = {}, {}
    for i, (m, S) in enumerate(rows):
        if m not in best_m or s[i] > s[best_m[m]]:
            if m in best_m: second_m[m] = s[best_m[m]]
            best_m[m] = i
        elif s[i] > second_m.get(m, 0): second_m[m] = s[i]
    cell_best = {}
    for i, (m, S) in enumerate(rows):
        for c in S:
            d = cell_best.setdefault(c, {})
            if s[i] > d.get(m, 0): d[m] = s[i]
    top = {m: set(rows[i][1]) for m, i in best_m.items()}
    out = np.zeros((len(rows), 6))
    for i, (m, S) in enumerate(rows):
        bm = s[best_m[m]]
        other = [max([v for mm, v in cell_best[c].items() if mm != m], default=0) for c in S]
        nbrs = [n for n in FA['adj'].get(m, ()) if n in top]
        overl = sum(bool(top[n] & set(S)) for n in nbrs)
        cp = np.mean([any(FB['adj'].get(x, set()) & top[n] for x in S) for n in nbrs if FA['area'].get(min(m, n) * 100000 + max(m, n), 0) > 50] or [-1])
        out[i] = [s[i], s[i] - bm, bm - second_m.get(m, 0), max(other), np.mean(other), overl + 0.1 * cp]
    return np.c_[X, out]
