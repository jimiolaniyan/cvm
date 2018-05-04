# Maximum likelihood estimator
def ml(obs):
    mean = sum(obs) * 1.0 / len(obs)
    var = sum([(x - mean) ** 2 for x in obs]) * 1.0 / len(obs)
    # sum returns an array
    return mean[0], var[0]


