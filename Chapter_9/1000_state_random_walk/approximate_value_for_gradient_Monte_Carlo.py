#python3 script
#value function for 1000-state Random Walk using the gradient Monte Carlo algorithm(Example 9.1)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

start_time = time.time()

MAX_EPISODE = int(1e7)
alpha = 2e-5
Theta = np.zeros(10)
for i in range(MAX_EPISODE):
    state_list = [499]
    while True:
        if random.random()<0.5:
            state_list.append(state_list[-1] - random.randint(1,100))
        else:
            state_list.append(state_list[-1] + random.randint(1,100))
        if state_list[-1]<0 or state_list[-1]>999:
            break
    terminal_state = state_list.pop()
    if terminal_state<0:
        reward = -1
    else:
        reward = 1
    while state_list:
        state = state_list.pop()
        Theta[state//100] = Theta[state//100] + alpha*(reward - Theta[state//100])
V = np.zeros(1000)
for i in range(10):
    V[i*100:i*100+100] = Theta[i]
TrueValue = np.load('TrueValue.npy')

plt.plot(range(1,1001), TrueValue, label='True value')
plt.plot(range(1,1001), V, label='Approximate MC value')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('approximate_value_for_gradient_Monte_Carlo')
plt.legend()
plt.show()

end_time = time.time()
print('time cost: '+str((end_time-start_time)//3600)+'h' \
      +str(((end_time-start_time)%3600)//60)+'m' \
      +str(((end_time-start_time)%3600)%60)+'s')