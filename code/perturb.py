"""Perturbed warps for training-data augmentation: smooth random displacement added on top of a warp."""
import numpy as np
from scipy.interpolate import RBFInterpolator
ZS = np.array([2.0, 1.0, 1.0])

class PerturbedWarp:
    def __init__(self, W, shapeB, amp_xy=3.0, amp_z=1.5, n=40, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.fa = W, W.fa
        P = rng.uniform(0, 1, (n, 3)) * (np.array(shapeB) - 1)
        D = rng.normal(0, 1, (n, 3)) * [amp_z, amp_xy, amp_xy]
        self.d = RBFInterpolator(P * ZS, D, kernel='thin_plate_spline', smoothing=n * 10.0, degree=0)
    def fwd(self, P):
        Q = self.W.fwd(P); return Q + self.d(Q * ZS)
    def bwd(self, Q):
        Q = np.asarray(Q, float); return self.W.bwd(Q - self.d(Q * ZS))
