import numpy as np
from spatialmath.base import trlog

def B(eps):
    norm_eps = np.linalg.norm(eps)
    if norm_eps == 0:
        return np.eye(3, dtype=np.float64)
    skew_eps = np.array([[0, -eps[2], eps[1]], [eps[2], 0, -eps[0]], [-eps[1], eps[0], 0]], np.float64)
    alpha = (norm_eps/2)/np.tan(norm_eps/2)
    return np.eye(3, dtype=np.float64) + skew_eps/2 + ((1-alpha)/(norm_eps**2)) * (skew_eps@skew_eps)

def hat(w):
    assert w.shape[0] == 3, "It is not a 3D vector."
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0]], np.float64)

def vee(W):
    assert W.shape == (3,3), "It is not a 3x3 matrix."
    return np.array([W[2,1], W[0,2], W[1,0]], np.float64)

def trLog(R, check=True, twist=False):
    if np.trace(R) >= 3:
        return np.zeros(3, np.float64)
    else:
        return trlog(R, check, twist)