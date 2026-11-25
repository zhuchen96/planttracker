"""Original source-registration routes, with explicit frames and no reference labels."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from bm import Warp, ZS
from em import fit_warp, confident_families
from match import overlaps, build_pairs
from blockmatch import affine_from_fn, inv_affine, refine, LEVELS
from heightmap import heightmap, search_topk, search_topk_aniso, lift3d_full, surface_labels

def consistent(P, D, k=8, thr=0.35):
    """drop anchors whose displacement deviates from the median of their k nearest anchors by > thr x local spacing"""
    if len(P) <= k: return np.ones(len(P), bool)
    tree = cKDTree(P * [ZS, 1, 1]); dist, nn = tree.query(P * [ZS, 1, 1], k=k + 1)
    disp = D - P; med = np.median(disp[nn[:, 1:]], axis=1)
    return np.linalg.norm((disp - med) * [ZS, 1, 1], axis=1) <= thr * np.median(dist[:, 1:]) * 2.5

def fit(P, D, smoothing):
    fa, *_ = np.linalg.lstsq(np.c_[P, np.ones(len(P))], D, rcond=None); ba, *_ = np.linalg.lstsq(np.c_[D, np.ones(len(D))], P, rcond=None)
    if len(P) > 2500:
        sel = np.random.default_rng(0).choice(len(P), 2500, replace=False); P, D = P[sel], D[sel]
    fr = RBFInterpolator(P * [ZS, 1, 1], D - np.c_[P, np.ones(len(P))] @ fa, kernel='thin_plate_spline', smoothing=smoothing)
    br = RBFInterpolator(D * [ZS, 1, 1], P - np.c_[D, np.ones(len(D))] @ ba, kernel='thin_plate_spline', smoothing=smoothing)
    return Warp(fa, ba, fr, br)

def refine_families(warp, masks, props, annealed):
    A, B = props
    schedule = [(0.7, 0.6)] * 10
    if annealed:
        schedule = [(0.7, 0.6), (0.65, 0.55), (0.6, 0.5), (0.55, 0.45)] + [(0.5, 0.4)] * 3
    for daughter_share, mother_share in schedule:
        pairs, x = build_pairs(A, B, overlaps(*masks, warp), warp)
        families = confident_families(pairs, x, A, B, daughter_share, mother_share,
                                       maxlayer=2 if annealed else 99)
        if annealed:
            mothers = sorted(families)
            if len(mothers) < 4:
                break
            centers = A['cen'][mothers]
            targets = np.array([np.average(B['cen'][families[m]], axis=0,
                                weights=B['vol'][families[m]]) for m in mothers])
            keep = consistent(centers, targets)
            if keep.sum() < 4:
                break
            warp = fit(centers[keep], targets[keep], 100.)
        elif len(families) >= 30:
            warp = fit_warp(A, B, families, 100.)
        else:
            break
    return warp


def estimate(masks, images, props, recipe=None):
    """Replay a selected height-map candidate, or screen and refine candidates for a new pair."""
    hA, mA = heightmap(masks[0])
    hB, mB = heightmap(masks[1])
    search_args = (hA, mA, hB, mB, np.arange(.9, 1.95, .05), np.arange(-45, 46, 3))
    surface = [surface_labels(s)[1:] for s in masks]

    def initialize(candidate):
        transform, _ = lift3d_full(*surface[0], *surface[1], *candidate[3:5])
        affine = affine_from_fn(transform, masks[0].shape)
        return Warp(affine, inv_affine(affine))

    def candidates(route):
        if route == 'similarity':
            return search_topk(*search_args, k=60)
        if route == 'anisotropic':
            return search_topk_aniso(*search_args, k=60)[0]
        raise ValueError(f'Unknown registration route {route}')

    levels = [LEVELS[0]] * 3 + [LEVELS[1]] * 2 + [LEVELS[2]] * 2
    if recipe:
        route, index = recipe['route'], recipe['candidate']
        selected = candidates(route)
        if index >= len(selected):
            raise ValueError(f'Candidate {index} unavailable among {len(selected)} height-map hypotheses')
        start = initialize(selected[index])
        warp, _ = refine(*images, masks[1], start, levels=levels)
        return refine_families(warp, masks, props, route == 'similarity')

    best = None
    for route in ('similarity', 'anisotropic'):
        screened = []
        for candidate in candidates(route):
            try:
                start = initialize(candidate)
            except (np.linalg.LinAlgError, IndexError):
                continue
            coarse, _ = refine(*images, masks[1], start, levels=LEVELS[:1])
            _, scores = refine(*images, masks[1], coarse, levels=LEVELS[:1])
            screened.append((scores[0][1], start))
        for _, start in sorted(screened, key=lambda item: -item[0])[:3 if route == 'similarity' else 10]:
            warp, _ = refine(*images, masks[1], start, levels=levels)
            warp = refine_families(warp, masks, props, route == 'similarity')
            pairs, x = build_pairs(*props, overlaps(*masks, warp), warp)
            count = len(confident_families(pairs, x, *props, maxlayer=99))
            if best is None or count > best[0]:
                best = (count, warp)
    if best is None:
        raise ValueError('Training registration found no usable surface alignment')
    return best[1]
