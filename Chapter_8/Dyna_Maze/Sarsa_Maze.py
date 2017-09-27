#python3 script
#Sarsa Maze: Example 8.1 using on-policy  n-step Sarsa rather than Dyna-Q
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

start_time = time.time()
# set the maze, 1 means the obstacles
hight = 6
width = 9
Maze = np.zeros((hight,width), dtype=np.int)
Maze[1:4, 2] = 1
Maze[4, 5] = 1
Maze[0:3, 7] = 1
start = np.array([2,0])
goal = np.array([0,8])

def one_step(state, action, Maze, goal):
    (hight, width) = np.shape(Maze)
    next_state = np.zeros(2, dtype=np.int)
    if np.all(state == goal):
        print('error! episode can not stop')
        exit(1)
    if action == 0:
        next_state[0] = state[0] - 1
        next_state[1] = state[1]
    elif action == 1:
        next_state[0] = state[0] 
        next_state[1] = state[1] + 1
    elif action == 2:
        next_state[0] = state[0] + 1
        next_state[1] = state[1]
    elif action == 3:
        next_state[0] = state[0]
        next_state[1] = state[1] - 1
    else:
        print('illegal action')
        exit(1)
    if not(next_state[0]>=0 and next_state[0]<hight and next_state[1]>=0 and next_state[1]<width) \
       or Maze[next_state[0], next_state[1]]==1:
        next_state[0] = state[0]
        next_state[1] = state[1]
    if np.all(next_state == goal):
        reward = 1
    else:
        reward = 0
    return (next_state, reward)
def epsilon_greedy(epsilon, Q, state):
    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        Q_max = np.max(Q[state[0], state[1], :])
        candidate = np.where(Q[state[0], state[1], :] == Q_max)[0]
        # action = candidate[random.randint(0, np.shape(candidate)[0]-1)]
        # action = random.sample(list(candidate), 1)[0]
        action = random.choice(list(candidate))
    return action


n = 50# n-step sarsa
epsilon = 0.1# essilon greedy
gamma = 0.95# discount
alpha = 0.1# step size
REP = 300# repeat times of the whole experiment
EPI = 50# number of episodes
step_per_epi = np.zeros(EPI, dtype=np.int)

for i in range(REP):
    Q = np.zeros((hight,width,4))# "4" means the 4 actions: up, right, down, left
    for i_1 in range(EPI):
        s_a_memory = np.zeros((n+1, 3), dtype=np.int)
        r_memory = np.zeros(n, dtype=np.int)
        s_a_memory[0, 0:2] = start
        s_a_memory[0, 2] = epsilon_greedy(epsilon, Q, s_a_memory[0, 0:2])
        T = float('inf')
        t = 0
        while True:
            if t < T:
                s_a_memory[(t+1)%(n+1), 0:2], r_memory[(t+1)%n] = one_step(s_a_memory[t%(n+1), 0:2], s_a_memory[t%(n+1), 2], Maze, goal)
                if np.all(s_a_memory[(t+1)%(n+1), 0:2]==goal):
                    T = t + 1
                else:
                    s_a_memory[(t+1)%(n+1), 2] = epsilon_greedy(epsilon, Q, s_a_memory[(t+1)%(n+1), 0:2])
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i_2 in range(tau+1, min(tau+n,T)+1):
                    G = G + gamma**(i_2-tau-1)*r_memory[i_2%n]
                if tau+n < T:
                    G = G + gamma**n*Q[s_a_memory[(t+1)%(n+1), 0], s_a_memory[(t+1)%(n+1), 1], s_a_memory[(t+1)%(n+1), 2]]
                Q[s_a_memory[tau%(n+1), 0], s_a_memory[tau%(n+1), 1], s_a_memory[tau%(n+1), 2]] \
                = Q[s_a_memory[tau%(n+1), 0], s_a_memory[tau%(n+1), 1], s_a_memory[tau%(n+1), 2]] + \
                  alpha*(G - Q[s_a_memory[tau%(n+1), 0], s_a_memory[tau%(n+1), 1], s_a_memory[tau%(n+1), 2]])
            t = t + 1
            if tau == T-1:
                break
        step_per_epi[i_1] = step_per_epi[i_1] + T
step_per_epi = step_per_epi / REP

episodes = np.arange(1, EPI+1)
plt.ylim(14, 400)
plt.plot(episodes, step_per_epi, label=str(n)+'-step Sarsa')
plt.xlabel('Episodes')
plt.ylabel('Steps per episodes')
plt.legend()
plt.show()

end_time = time.time()
print('time cost: '+str((end_time-start_time)//3600)+'h' \
      +str(((end_time-start_time)%3600)//60)+'m' \
      +str(((end_time-start_time)%3600)%60)+'s')