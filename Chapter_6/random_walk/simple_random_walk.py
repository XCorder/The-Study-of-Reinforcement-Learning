#python3 script
#Simple Random Walk: only one non-terminal state
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
#import copy

start = time.time()

V = np.zeros(3)# V[0] is of the left terminal state, V[2] is of the right terminal state, 
# V[1] if of the state A
V[2] = 1
V_true = np.array([0, 1/2, 1])
MaxEpisode = 20
alpha = 0.17
Repeat = 10000000
RMS_error = np.zeros(MaxEpisode)
for r in range(Repeat):
    V = np.array([0.,0.65,1.])
    for i in range(MaxEpisode):
        RMS_error[i] = RMS_error[i] + np.linalg.norm(V - V_true)
        state = 1
        if random.random() < 0.5:
            state_1 = state - 1
        else:
            state_1 = state + 1
        V[state] = V[state] + alpha*(V[state_1]-V[state])
        
RMS_error = RMS_error/Repeat
print('RMS_error[min] = '+str(np.min(RMS_error)))

episode = range(MaxEpisode)
plt.ylim(0.11, 0.16)
plt.plot(episode, RMS_error, label='$\\alpha$ = '+str(alpha))
plt.xlabel('Episodes')
plt.ylabel('Empirical RMS error')
plt.legend()
plt.show()

end = time.time()
print('time cost: '+str((end-start)//3600)+'h'+str(((end-start)%3600)//60)+'m'+str(((end-start)%3600)%60)+'s')