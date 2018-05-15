import numpy as np


def mog(x, K):
    if type(x) != np.ndarray:
        x = np.array(x)

    intial_lambda = [1/K for i in range(0, K)]
    mu = x[np.random.choice(x.shape[0], K, replace=False)]

    # # find initial covariance
    mean = np.sum(x, axis=0) / x.shape[0]
    variance = var_sum(x, mean) / x.shape[0]
    covariance = [variance for i in range(0, K)]
    covariance = np.array(covariance)

    # while True:
        # Expectation Step


def var_sum(x, mu):
    cov = 0
    for i in range(0, len(x)):
            mu_diff = x[i] - mu
            mu_diff_col = np.array([mu_diff])
            diff_by_trans = mu_diff_col.T * mu_diff
            cov += diff_by_trans
    return cov


x = [[23,22,123],
     [213, 22, 31],
     [23, 145, 123],
     [46, 22, 50],
     [46, 22, 50],[36, 212, 50],[72, 12, 70],[46, 11, 8],[146, 292, 20],[49, 252, 150],
     [23, 78, 123],]

mog(x,2)

