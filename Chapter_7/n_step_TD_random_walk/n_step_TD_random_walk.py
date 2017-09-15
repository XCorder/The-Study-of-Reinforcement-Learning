#python3 script
#n-step TD Methods on the Random Walk(Example 7.1)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
#import copy

start = time.time()

REPETITION = 1000
NUM_S = 9 #number of not terminal states
NUM_EPISODE = 10 #average RMS error on first NUM_EPISODE episodes
left_reward = -1
right_reward = 1
Value = left_reward + (right_reward-left_reward) * np.arange(1, NUM_S+1) / (NUM_S+1) #the true value of the state 0 to state 18
V = np.zeros(NUM_S)
full_error = np.zeros((10, 20))
for h in range(REPETITION):
    for i in range(10):
        n = 2**i
        for j in range(20):
            alpha = 0.05*(j+1)
            s = np.zeros(n+1, dtype=np.int)
            r = np.zeros(n)
            V = V * 0
            error = 0
            for k in range(NUM_EPISODE):
                s[0] = NUM_S // 2
                T = float('inf')
                t = 0
                while True:
                    if t < T:
                        if random.random() < 0.5:
                            s[(t+1)%(n+1)] = s[t%(n+1)] - 1
                        else:
                            s[(t+1)%(n+1)] = s[t%(n+1)] + 1
                        if s[(t+1)%(n+1)] == -1:
                            r[(t+1)%n] = left_reward
                            T = t + 1
                        elif s[(t+1)%(n+1)] == NUM_S:
                            r[(t+1)%n] = right_reward
                            T = t + 1
                        else:
                            r[(t+1)%n] = 0
                    tau = t - n + 1
                    if tau >= 0:
                        G = 0
                        for ii in range(tau+1, min(tau+n, T)+1):
                            G = G + r[ii%n]
                        if tau+n < T:
                            G = G + V[s[(tau+n)%(n+1)]]
                        V[s[tau%(n+1)]] = V[s[tau%(n+1)]] + alpha*(G - V[s[tau%(n+1)]])
                    if tau == T-1:
                        break
                    t = t + 1
                error = error + np.linalg.norm(V-Value)/math.sqrt(NUM_S)

            full_error[i,j] = full_error[i,j] + error/NUM_EPISODE
full_error = full_error / REPETITION

np.save('full_error_state_9_left_minus1_rep_1000.npy', full_error)


end = time.time()
print('time cost: '+str((end-start)//3600)+'h'+str(((end-start)%3600)//60)+'m'+str(((end-start)%3600)%60)+'s')