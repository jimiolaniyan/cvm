import numpy as np
import map_cat
import mle_cat
import matplotlib.pyplot as plt

original_probabilities = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25]

r1 = np.random.choice(range(1, 7), 500, p=original_probabilities)
alpha = [5,5,5,5,5,5]

estimated_probabilities = map_cat.map_cat(r1, alpha, 6)
mle_estimated_probabilities = mle_cat.mle_cat(r1)

ax1 = plt.subplot(131)
ax1.bar(range(1, 7), original_probabilities)

ax2 = plt.subplot(132)
ax2.bar(range(1, 7), estimated_probabilities, color='red')

ax3 = plt.subplot(133)
ax3.bar(range(1, 7), mle_estimated_probabilities, color='green')

plt.show()
