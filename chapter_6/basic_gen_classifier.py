from __future__ import division
from scipy.stats import multivariate_normal
import numpy as np


# A basic generative classifier
def gen_classifier(x, x_test, K):
    # Ensure we have a numpy array for both data set and test set
    x = np.array(x)
    x_test = np.array(x_test)

    # Get the training test; all columns except the last one
    x_train = x[:, :-1]

    # get the training set labels
    labels = x[:, [-1]]

    mu_vector = []
    var = []

    # Use to get prior probabilities over the world state
    lammda = np.zeros((K,1))

    # the counts of each class
    counts = np.bincount(np.squeeze(labels))

    # Get params (theta) by class
    for k in range(0, K):
        # Get the numerator for the mean of the class k
        k_mean_sum = mean_sum(k, x_train, labels)

        # Get the mean of the class k (Dx1 matrix)
        k_mean = k_mean_sum / counts[k]
        mu_vector.append(k_mean)

        # Get the numerator for the covariance of the class k
        k_var_sum = var_sum(k, x_train, k_mean, labels)

        # Get the covariance of the class k (DxD matrix)
        k_var = k_var_sum / counts[k]

        # Prior probability of class k
        lammda[k] = counts[k] / x_train.shape[0]
        var.append(k_var)

    likelihood = np.zeros((x_test.shape[0], K))

    # Compute likelihoods for each class for data points in test set
    for k in range(0, K):
        cov_mat = var[k] * np.eye(var[0].shape[0])
        l_val = multivariate_normal.pdf(x_test, cov=cov_mat, mean=mu_vector[k])
        likelihood[:, k] = l_val

    # Classify new data point using Bayes rule
    # Get the denominator
    denom = 1/np.dot(likelihood, lammda)

    # Calculate the posterior probability of each data point in test set
    posterior_prob = likelihood * lammda.T * denom
    return posterior_prob


# helper method to get the sum by class for mean calculation
def mean_sum(k, x, labels):
    mu_sum = 0
    for i in range(0, len(x)):
        if labels[i][0] == k:
            mu_sum += x[i]
    return mu_sum


# helper method to get the sum by class for covariance calculation
def var_sum(k, x, mu, labels):
    cov = 0
    for i in range(0, len(x)):
        if labels[i][0] == k:
            mu_diff = x[i] - mu
            mu_diff_col = np.array([mu_diff])
            diff_by_trans = mu_diff_col.T * mu_diff
            cov += diff_by_trans
    return cov
