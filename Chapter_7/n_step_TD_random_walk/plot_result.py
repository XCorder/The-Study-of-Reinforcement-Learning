#python3 script
#plot the result of n-step TD Methods on the Random Walk(Example 7.1)
#author: Xiang Chao

import numpy as np
import matplotlib.pyplot as plt

full_error = np.load('full_error_state_9_left_minus1_rep_1000.npy')
alpha = np.arange(1, 21) / 20
plt.ylim(0.24, 0.6)
for i in range(10):
    plt.plot(alpha, full_error[i, :], label='n = '+str(2**i))
plt.xlabel('$\\alpha$')
plt.ylabel('Average RMS error over 19 states and first 10 episodes')
plt.legend()
plt.show()