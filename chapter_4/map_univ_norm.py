def map_univ_norm(x, alpha, beta, gamma, delta):
    mean = (sum(x) * 1.0 + (gamma * delta))/(len(x) + gamma)
    var = (sum((x - mean) ** 2) + 2 * beta + gamma * ((delta - mean) ** 2))/(len(x) + 3 + 2 * alpha)
    return mean[0], var[0]
