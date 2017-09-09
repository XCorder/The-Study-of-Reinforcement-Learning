#python3 script
#Stochastic Wind(Exercise 6.8)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

start = time.time()

def epsilon_greedy(epsilon, s, Q):
    if random.random() < epsilon:
        a = random.randint(0,7)
    else:
        a = np.argmax(Q[s[0],s[1],:])
    return a

def one_step(s, a, wind, h, w):
    s_1 = np.array([0,0])
    r = -1
    if a == 0:
        s_1[0] = s[0] - 1
        s_1[1] = s[1] - 1
    elif a == 1:
        s_1[0] = s[0] - 1
        s_1[1] = s[1]
    elif a == 2:
        s_1[0] = s[0] - 1
        s_1[1] = s[1] + 1
    elif a == 3:
        s_1[0] = s[0]
        s_1[1] = s[1] + 1
    elif a == 4:
        s_1[0] = s[0] + 1
        s_1[1] = s[1] + 1
    elif a == 5:
        s_1[0] = s[0] + 1
        s_1[1] = s[1]
    elif a == 6:
        s_1[0] = s[0] + 1
        s_1[1] = s[1] - 1
    elif a == 7:
        s_1[0] = s[0]
        s_1[1] = s[1] - 1
    else:
        print('appear illegal action')
        exit(-1)
    win = wind[s[1]]
    x = random.random()
    if x < 1/3:
        win = win - 1
    elif x > 2/3:
        win = win + 1
    s_1[0] = s_1[0] - win
    if s_1[0] < 0:
        s_1[0] = 0
    elif s_1[0] >= h:
        s_1[0] = h - 1
    if s_1[1] < 0:
        s_1[1] = 0
    elif s_1[1] >= w:
        s_1[1] = w - 1
    return (r, s_1)

def one_step_definite_wind(s, a, wind, h, w):
    s_1 = np.array([0,0])
    r = -1
    if a == 0:
        s_1[0] = s[0] - 1
        s_1[1] = s[1] - 1
    elif a == 1:
        s_1[0] = s[0] - 1
        s_1[1] = s[1]
    elif a == 2:
        s_1[0] = s[0] - 1
        s_1[1] = s[1] + 1
    elif a == 3:
        s_1[0] = s[0]
        s_1[1] = s[1] + 1
    elif a == 4:
        s_1[0] = s[0] + 1
        s_1[1] = s[1] + 1
    elif a == 5:
        s_1[0] = s[0] + 1
        s_1[1] = s[1]
    elif a == 6:
        s_1[0] = s[0] + 1
        s_1[1] = s[1] - 1
    elif a == 7:
        s_1[0] = s[0]
        s_1[1] = s[1] - 1
    else:
        print('appear illegal action')
        exit(-1)
    win = wind[s[1]]
    # x = random.random()
    # if x < 1/3:
    #     win = win - 1
    # elif x > 2/3:
    #     win = win + 1
    s_1[0] = s_1[0] - win
    if s_1[0] < 0:
        s_1[0] = 0
    elif s_1[0] >= h:
        s_1[0] = h - 1
    if s_1[1] < 0:
        s_1[1] = 0
    elif s_1[1] >= w:
        s_1[1] = w - 1
    return (r, s_1)

h = 7
w = 10
gridworld = np.zeros((h, w), dtype=np.int8)
gridworld[3,0] = 1 #start state
gridworld[3,7] = 2 #goal state
wind = np.array([0,0,0,1,1,1,2,2,1,0])
Q = np.zeros(gridworld.shape+(8,))
epsilon = 0.1
alpha = 0.5
MaxStep = 100000
total_cnt = 0
step_episode = [] #the number of time steps of each episode
while total_cnt < MaxStep:
    s = np.array([3, 0])
    a = epsilon_greedy(epsilon, s, Q)
    step_cnt = 0
    while not np.all(s == np.array([3,7])):
        step_cnt = step_cnt + 1
        total_cnt = total_cnt + 1
        r, s_1 = one_step(s, a, wind, h, w)
        #r, s_1 = one_step_definite_wind(s, a, wind, h, w) # with definite wind
        a_1 = epsilon_greedy(epsilon, s_1, Q)
        Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha*(r + Q[s_1[0],s_1[1],a_1] - Q[s[0],s[1],a])
        s = copy.deepcopy(s_1)
        a = copy.deepcopy(a_1)
    step_episode.append(step_cnt)
print('min(step_episode) = '+str(min(step_episode)))

episode = range(1,len(step_episode)+1)
plt.ylim(0, 1000)
plt.plot(episode, step_episode, label='$\\alpha$ = '+str(alpha)+', $\epsilon$ = '+str(epsilon))
plt.xlabel('Episodes')
plt.ylabel('Time Steps')
plt.legend()
plt.show()

# run with definite wind
s = np.array([3, 0])
a = epsilon_greedy(epsilon, s, Q)
step_cnt = 0
while not np.all(s == np.array([3,7])):
    step_cnt = step_cnt + 1
    r, s_1 = one_step_definite_wind(s, a, wind, h, w)
    a_1 = epsilon_greedy(epsilon, s_1, Q)
    s = copy.deepcopy(s_1)
    a = copy.deepcopy(a_1)
print('the number of time steps with definite wind is '+str(step_cnt))

end = time.time()
print('time cost: '+str((end-start)//3600)+'h'+str(((end-start)%3600)//60)+'m'+str(((end-start)%3600)%60)+'s')