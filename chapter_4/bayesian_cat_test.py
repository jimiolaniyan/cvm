import numpy as np
import map_cat
import bayesian_cat
import matplotlib.pyplot as plt

original_probabilities = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25]

r1 = np.random.choice(range(1, 7), 2000, p=original_probabilities)

alpha = [1,1,1,1,1,1]

prediction = bayesian_cat.bayesian_cat(r1, alpha)
map_estimated_probabilities = map_cat.map_cat(r1, alpha, 6)

ax1 = plt.subplot(131)
ax1.bar(range(1, 7), original_probabilities)

ax2 = plt.subplot(132)
ax2.bar(range(1, 7), map_estimated_probabilities, color='red')

ax3 = plt.subplot(133)
ax3.bar(range(1, 7), prediction, color='green')

plt.show()
