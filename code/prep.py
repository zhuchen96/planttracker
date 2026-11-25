from collections import deque
import numpy as np, tifffile
from scipy import ndimage


def frame_props(t, seg_path=None, raw_path=None):
    seg = tifffile.imread(seg_path)
    raw = tifffile.imread(raw_path)
    n = int(seg.max()) + 1
    vol = np.bincount(seg.ravel(), minlength=n).astype(float)
    with np.errstate(invalid='ignore'):
        cen = np.array(ndimage.center_of_mass(np.ones_like(seg, dtype=np.uint8), seg, np.arange(n)))
    labels = np.nonzero(vol)[0]; labels = labels[labels > 0]
    # contact areas (face counts) and boundary membrane intensity
    area, wall = {}, {}
    rawf = raw.astype(np.float32)
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3; s1[ax] = slice(1, None); s2[ax] = slice(None, -1)
        u, v = seg[tuple(s1)].ravel(), seg[tuple(s2)].ravel()
        m = (u != v) & (u > 0) & (v > 0)
        code = np.minimum(u[m], v[m]).astype(np.int64) * 100000 + np.maximum(u[m], v[m])
        inten = (rawf[tuple(s1)].ravel()[m] + rawf[tuple(s2)].ravel()[m]) / 2
        k, inv, c = np.unique(code, return_inverse=True, return_counts=True)
        s = np.bincount(inv, weights=inten)
        for kk, cc, ss in zip(k.tolist(), c.tolist(), s.tolist()):
            area[kk] = area.get(kk, 0) + cc; wall[kk] = wall.get(kk, 0) + ss
    wall = {k: wall[k] / area[k] for k in area}
    adj = {}
    for k in area:
        i, j = divmod(k, 100000); adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
    # layers from the outer (low-z) surface
    tis = seg > 0; has = tis.any(0); zt = tis.argmax(0)
    first = set(np.unique(np.take_along_axis(seg, zt[None], 0)[0][has]).tolist()) - {0}
    lay = {c: 1 for c in first}; q = deque(first)
    while q:
        c = q.popleft()
        for nb in adj.get(c, ()):
            if nb not in lay: lay[nb] = lay[c] + 1; q.append(nb)
    # mean interior intensity per cell, and boundary-to-background contact
    inner = np.bincount(seg.ravel(), weights=rawf.ravel(), minlength=n) / np.maximum(vol, 1)
    bb = ndimage.find_objects(seg)
    # cells touching the image border (partially out of field of view)
    border = set()
    for ax in range(3):
        for i in (0, -1): border |= set(np.unique(np.take(seg, i, axis=ax)).tolist())
    border.discard(0)
    # principal axes for shape features
    return dict(shape=seg.shape, vol=vol, cen=cen, labels=labels, area=area, wall=wall, adj=adj, lay=lay,
                inner=inner, bbox=bb, border=border)
