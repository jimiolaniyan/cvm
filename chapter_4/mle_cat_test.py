import numpy as np
import mle_cat
import matplotlib.pyplot as plt

original_probabilities = [0.25,0.15,0.1,0.1,0.15,0.25]

r1 = np.random.choice(range(1,7), 500, p=original_probabilities)
estimated_probabilities = mle_cat.mle_cat(r1)

ax1 = plt.subplot(121)
ax1.bar(range(1,7), original_probabilities)

ax2 = plt.subplot(122)
ax2.bar(range(1, 7), estimated_probabilities, color='red')

plt.show()