#python3 script
#plot the results of Shortcut Maze (Example 8.1, Exercise 8.4)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

start_time = time.time()

cumulative_reward_0 = np.load('cumulative_reward_Dyna_Q_n_200.npy')
cumulative_reward_1 = np.load('cumulative_reward_Dyna_Q_Plus_n_200_kappa_0.0009.npy')
cumulative_reward_2 = np.load('cumulative_reward_Dyna_Q_Plus_variation_n_200_kappa_0.0009.npy')
cumulative_reward_3 = np.load('cumulative_reward_Dyna_Q_Plus_epsilon_0_n_200_kappa_0.0009.npy')
cumulative_reward_4 = np.load('cumulative_reward_Dyna_Q_epsilon_0_n_200.npy')

steps = np.arange(1, 6000+1)
plt.plot(steps, cumulative_reward_0[1:], label='Dyna_Q')
plt.plot(steps, cumulative_reward_1[1:], label='Dyna_Q_Plus')
plt.plot(steps, cumulative_reward_2[1:], label='Dyna_Q_Plus_variation(epsilon = 0)')
plt.plot(steps, cumulative_reward_3[1:], label='Dyna_Q_Plus_epsilon_0')
plt.plot(steps, cumulative_reward_4[1:], label='Dyna_Q_epsilon_0')
plt.xlabel('Time steps')
plt.ylabel('Cumulative reward')
plt.title('only compare the slopes')
plt.legend()
plt.show()


end_time = time.time()
print('time cost: '+str((end_time-start_time)//3600)+'h' \
      +str(((end_time-start_time)%3600)//60)+'m' \
      +str(((end_time-start_time)%3600)%60)+'s')