#python3 script
#Shortcut Maze (Example 8.1, using Dyna-Q)
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
Maze[3, 1:] = 1
start = np.array([5,3])
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
        action = random.choice(list(candidate))
    return action
def planning(Model, observed_state, Q, alpha, gamma):
    state = random.choice(observed_state)
    observed_action = np.where(Model[state[0], state[1], :, 0] != -1)[0]
    action = random.choice(list(observed_action))
    next_state = Model[state[0], state[1], action, 0:2]
    reward = Model[state[0], state[1], action, 2]
    Q[state[0], state[1], action] \
    = Q[state[0], state[1], action] + \
      alpha*(reward + gamma*np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])

n = 200# planning steps
epsilon = 0# essilon greedy
gamma = 0.95# discount
alpha = 0.1# step size
#REP = 30# repeat times of the whole experiment
STEP = 6000# number of steps
cumulative_reward = np.zeros(STEP+1, dtype=np.int)
Q = np.zeros((hight,width,4))# "4" means the 4 actions: up, right, down, left
Model = np.full((hight,width,4, 3), -1, dtype=np.int)# the model learnt; "3" means the latest state and reward
observed_state = []

state = copy.deepcopy(start)
for i in range(1, STEP+1):
    if i==(STEP//2+1):
        Maze[3, 8] = 0
    action = epsilon_greedy(epsilon, Q, state)
    next_state, reward = one_step(state, action, Maze, goal)
    Q[state[0], state[1], action] \
    = Q[state[0], state[1], action] + \
      alpha*(reward + gamma*np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])
    Model[state[0], state[1], action, :] = np.array([next_state[0], next_state[1], reward])
    if not(list(state) in observed_state):
        observed_state.append(list(state))
    if np.all(next_state==goal):
        state = copy.deepcopy(start)
    else:
        state = next_state
    for i_1 in range(n):
        planning(Model, observed_state, Q, alpha, gamma)
    cumulative_reward[i] = cumulative_reward[i-1] + reward 

np.save('cumulative_reward_Dyna_Q_epsilon_0_n_200.npy', cumulative_reward)

steps = np.arange(1, STEP+1)
plt.plot(steps, cumulative_reward[1:], label=str(n)+' planning steps')
plt.xlabel('Time steps')
plt.ylabel('Cumulative reward')
plt.legend()
plt.show()

end_time = time.time()
print('time cost: '+str((end_time-start_time)//3600)+'h' \
      +str(((end_time-start_time)%3600)//60)+'m' \
      +str(((end_time-start_time)%3600)%60)+'s')