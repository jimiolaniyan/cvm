from __future__ import division

import numpy as np


def bayesian_cat(x,alpha):
    x_array = np.array(x)
    counts = np.bincount(x_array)

    # exclude 0
    alpha_prime = alpha + counts[1:]
    x_pred = alpha_prime / sum(alpha_prime)
    return x_pred