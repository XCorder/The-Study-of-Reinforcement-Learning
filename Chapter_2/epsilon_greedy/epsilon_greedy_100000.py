#python3 script
#epsilon greedy method timestep 100000
#author: Xiang Chao

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random

def OneRun(arm_num, eps, step_num):
    value = np.random.normal(0, 1, arm_num)#mu=0, sigma=1;the value for each arm
    Q = np.zeros(arm_num)#the estimate of value for each arm
    N = np.zeros(arm_num)#the numbers of exploitations for each arm
    reward = np.zeros(step_num//100 + 1)#the reward of each step
    if_optimal = np.zeros(step_num//100 +1)#1 for the optimal action;0 for other actions
    for i in range(step_num + 1):
        if i%100 == 0:
            if random.random() > eps:#random.random() returns a number in [0,1) randomly
                A = np.argmax(Q)#A is index of the first max element in Q
                if value[A] == np.max(value):
                    if_optimal[i//100] = 1
            else:
                A = random.randint(0, 9)#A is a random integer from 0 to 9
                if value[A] == np.max(value):
                    if_optimal[i//100] = 1
            R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
            reward[i//100] = R
            N[A] = N[A] + 1
            Q[A] = Q[A] + (R - Q[A])/N[A]
        else:
            if random.random() > eps:#random.random() returns a number in [0,1) randomly
                A = np.argmax(Q)#A is index of the first max element in Q
            else:
                A = random.randint(0, 9)#A is a random integer from 0 to 9
            R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
            N[A] = N[A] + 1
            Q[A] = Q[A] + (R - Q[A])/N[A]
    return (reward, if_optimal)

repeat_num = 1000
arm_num = 10
eps = np.array([0, 0.01, 0.1])
step_num = 100000
ave_reward = np.zeros((3,step_num//100 + 1))
rate_optimal = np.zeros((3,step_num//100 + 1))
for j in range(3):
    for i in range(repeat_num):
        [a, b] = OneRun(arm_num, eps[j], step_num)
        ave_reward[j,:] = ave_reward[j,:] + a
        rate_optimal[j,:] = rate_optimal[j,:] + b
    ave_reward[j,:] = ave_reward[j,:] / repeat_num
    rate_optimal[j,:] = rate_optimal[j,:] / repeat_num

step = range(step_num//100 + 1)

ax = plt.subplot(111)
plt.plot(step, ave_reward[0,:], label='$\epsilon = 0$', color='r')
plt.plot(step, ave_reward[1,:], label='$\epsilon = 0.01$', color='g')
plt.plot(step, ave_reward[2,:], label='$\epsilon = 0.1$', color='b')
ax.yaxis.set_major_locator(MultipleLocator(0.1))
plt.grid(True)
plt.xlabel('Steps(devided by 100)')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('Average_reward_100000.png')
plt.show()

ax = plt.subplot(111)
plt.plot(step, rate_optimal[0,:], label='$\epsilon = 0$', color='r')
plt.plot(step, rate_optimal[1,:], label='$\epsilon = 0.01$', color='g')
plt.plot(step, rate_optimal[2,:], label='$\epsilon = 0.1$', color='b')
ax.yaxis.set_major_locator(MultipleLocator(0.05))
plt.grid(True)
plt.xlabel('Steps(devided by 100)')
plt.ylabel('rate of optimal action')
plt.legend()
plt.savefig('Rate_optimal_100000.png')
#plt.show()