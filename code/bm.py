import numpy as np
from scipy import ndimage
ZS=2.


class Warp:
    """forward A->B and backward B->A maps: affine + optional TPS correction (fitted on isotropic coords)"""
    def __init__(self, fwd_aff, bwd_aff, fwd_rbf=None, bwd_rbf=None):
        self.fa, self.ba, self.fr, self.br = fwd_aff, bwd_aff, fwd_rbf, bwd_rbf
    @staticmethod
    def _app(M, P): return np.c_[P, np.ones(len(P))] @ M
    def fwd(self, P):
        P = np.asarray(P, float); Q = self._app(self.fa, P)
        if self.fr is not None: Q = Q + self.fr(P * [ZS, 1, 1])
        return Q
    def bwd(self, Q):
        Q = np.asarray(Q, float); P = self._app(self.ba, Q)
        if self.br is not None: P = P + self.br(Q * [ZS, 1, 1])
        return P


def bwd_coords(W, shapeB, ds, step=8):
    """B->A coordinates for every voxel of the ds-downsampled B grid, via a coarse grid + linear interpolation"""
    cg = [np.arange(0, s + step, step) for s in shapeB]
    G = np.stack(np.meshgrid(*cg, indexing='ij'), -1)
    P = W.bwd(G.reshape(-1, 3).astype(float)).reshape(G.shape)
    fine = [np.arange(0, s, ds) / step for s in shapeB]
    FG = np.stack(np.meshgrid(*fine, indexing='ij'), 0)
    return np.stack([ndimage.map_coordinates(P[..., k], FG, order=1) for k in range(3)], 0)
