from math import gamma, sqrt, pi
import numpy as np


def bayesian_univ_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test):
    # Compute normal inverse gamma posterior over normal parameters
    alpha_prime = alpha_prior + (len(x) / 2)

    beta_prime_left = (sum(x ** 2) / 2) + beta_prior + ((gamma_prior * (delta_prior ** 2)) / 2)
    beta_prime_right = (((gamma_prior * delta_prior) + sum(x)) ** 2) / (2 * (gamma_prior + len(x)))
    beta_prime = beta_prime_left - beta_prime_right

    gamma_prime = gamma_prior + len(x)
    delta_prime = ((gamma_prior * delta_prior) + sum(x)[0]) / (gamma_prior + len(x))

    # Compute intermediate parameters
    alpha_cap = alpha_prime + 0.5
    gamma_cap = gamma_prime + 1
    beta_cap = (((x_test ** 2) / 2) + beta_prime + (gamma_prime * (delta_prime ** 2)) / 2) - (
                (((gamma_prime * delta_prime) + x_test) ** 2) / (2 * gamma_cap))

    x_pred_num = sqrt(gamma_prime) * (beta_prime ** alpha_prime) * gamma(alpha_cap)
    x_pred_denum = (sqrt(2 * pi) * sqrt(gamma_cap) * (beta_cap ** alpha_cap) * gamma(alpha_prime))
    x_pred = x_pred_num/x_pred_denum
    return x_pred
