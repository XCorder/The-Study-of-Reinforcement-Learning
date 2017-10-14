#python3 script
#value function for 1000-state Random Walk using the Semi-gradient TD(0) algorithm(Example 9.1)
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
    state = 499
    while True:
        if random.random()<0.5:
            next_state = state - random.randint(1,100)#math.ceil(random.random()*100)
        else:
            next_state = state + random.randint(1,100)#math.ceil(random.random()*100)
        if next_state<0:
            reward = -1
            V_next = 0
        elif next_state>999:
            reward = 1
            V_next = 0
        else:
            reward = 0
            V_next = Theta[next_state//100]
        Theta[state//100] = Theta[state//100] + alpha*(reward + V_next - Theta[state//100])
        state = copy.deepcopy(next_state)
        if state<0 or state>999:
            break

V = np.zeros(1000)
for i in range(10):
    V[i*100:i*100+100] = Theta[i]
TrueValue = np.load('TrueValue.npy')

plt.plot(range(1,1001), TrueValue, label='True value')
plt.plot(range(1,1001), V, label='Approximate TD value')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('approximate_value_for_semi_gradient_TD')
plt.legend()
plt.show()

end_time = time.time()
print('time cost: '+str((end_time-start_time)//3600)+'h' \
      +str(((end_time-start_time)%3600)//60)+'m' \
      +str(((end_time-start_time)%3600)%60)+'s')