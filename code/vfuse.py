from pathlib import Path
import pickle
import numpy as np


def vfuse(vg, fam, ov, mt, tau=0.7, cthr=0.85):
    out = {}; used = set()
    for m, (ds, cv) in sorted(vg.items(), key=lambda x: -x[1][1]):
        if cv >= cthr and not (set(ds) & used): out[m] = list(ds); used |= set(ds)
    for src in ({m: ds for m, (ds, q) in sorted(fam.items(), key=lambda x: -x[1][1]) if q >= tau}, mt, ov):
        for m, ds in src.items():
            if m in out: continue
            ds = [d for d in ds if d not in used]
            if ds: out[m] = ds; used |= set(ds)
    return out
