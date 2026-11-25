from pathlib import Path
import pickle
import numpy as np


def clone_cen(B, ib, ds):
    j = [ib[d] for d in ds]; w = B['vol'][j]; return (B['cen'][j] * w[:, None]).sum(0) / w.sum()
