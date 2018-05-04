from __future__ import division

import numpy as np


def mle_cat(x):
    x_array = np.array(x)
    counts = np.bincount(x_array)
    theta = counts / len(x)

    # Ignore 0
    return theta[1:]
