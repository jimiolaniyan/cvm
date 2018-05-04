import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from math import sqrt

import cv_algo.chapter_4.mle_univ_norm

original_mu = 5
original_sigma = 10

# Generate a distribution with mean and std
gen_data = original_mu + original_sigma * np.random.randn(100, 1)

estimated_mu, estimated_var = cv_algo.chapter_4.mle_univ_norm.ml(gen_data)
estimated_sigma = sqrt(estimated_var)

muError = abs(original_mu - estimated_mu)
sigError = abs(original_sigma - estimated_sigma)

print(muError, sigError)

x = np.linspace(-30, 40, 100)
original = mlab.normpdf(x, original_mu, original_sigma)
estimated = mlab.normpdf(x, estimated_mu, estimated_sigma)

plt.plot(x, original, 'b', label="original")
plt.plot(x, estimated, 'r', label="estimated")
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.show()
