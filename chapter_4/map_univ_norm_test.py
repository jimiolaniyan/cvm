from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

import map_univ_norm
import mle_univ_norm

original_mu = 5
original_sigma = 10

# Generate a distribution with mean and std
gen_data = original_mu + original_sigma * np.random.randn(10000, 1)

estimated_mu, estimated_var = map_univ_norm.map_univ_norm(gen_data, 1, 2, 1, 0)
estimated_sigma = sqrt(estimated_var)

mle_mu, mle_var = mle_univ_norm.ml(gen_data)
mle_sigma = sqrt(mle_var)

map_mu_error = abs(original_mu - estimated_mu)
map_sig_error = abs(original_sigma - estimated_sigma)

mle_mu_error = abs(original_mu - mle_mu)
mle_sig_error = abs(original_sigma - mle_sigma)

# print estimated_mu, mle_mu, '\n', estimated_sigma, mle_sigma, '\n'
# print map_mu_error, mle_mu_error, '\n', map_sig_error, mle_sig_error,

x = np.linspace(-30, 40, 100)
original = mlab.normpdf(x, original_mu, original_sigma)
estimated = mlab.normpdf(x, estimated_mu, estimated_sigma)
mle = mlab.normpdf(x, mle_mu, mle_sigma)

plt.plot(x, original, 'b', label="Original")
plt.plot(x, estimated, 'r', label="MAP")
plt.plot(x, mle, 'g', label="MLE")
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.show()

