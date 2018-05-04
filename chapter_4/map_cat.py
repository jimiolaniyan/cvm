from __future__ import division
import numpy as np


def map_cat(x, alpha, K):
    x_array = np.array(x)
    alpha_array = np.array(alpha)
    nk = np.bincount(x_array)

    theta = (nk[1:] - 1 + alpha_array) / (len(x) - K + sum(alpha))
    return theta
