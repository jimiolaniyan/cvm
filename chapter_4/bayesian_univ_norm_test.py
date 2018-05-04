from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

import map_univ_norm
import bayesian_univ_norm

original_mu = 5
original_sigma = 8

gen_data = original_mu + original_sigma * np.random.randn(100, 1)

x_test = np.linspace(-20, 30, 51)

x_pred = bayesian_univ_norm.bayesian_univ_norm(gen_data, 1, 1, 1, 0, x_test)

estimated_mu, estimated_var = map_univ_norm.map_univ_norm(gen_data, 4, 1, 1, 0)
estimated_sigma = sqrt(estimated_var)

estimated_map = mlab.normpdf(x_test, estimated_mu, estimated_sigma)
original = mlab.normpdf(x_test, original_mu, original_sigma)

# estimated_mu_bayes = map_univ_norm.map_univ_norm(x_pred, 1, 1, 1, 0)
print(len(x_pred))

plt.plot(x_test, x_pred, 'r', label="Bayesian")
plt.plot(x_test, estimated_map, 'g', label="MAP")
plt.plot(x_test, original, 'b', label="Original")
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.show()
