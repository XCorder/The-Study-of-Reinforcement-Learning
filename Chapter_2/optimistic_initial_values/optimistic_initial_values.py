#python3 script
#optimistic initial values method
#author: Xiang Chao

import numpy as np
import matplotlib.pyplot as plt
import random

def OneRun_1(arm_num, eps, step_num, alpha):
    value = np.random.normal(0, 1, arm_num)#mu=0, sigma=1;the value for each arm
    #Q = np.zeros(arm_num)#the estimate of value for each arm
    Q = np.full(arm_num, 5.)
    
    reward = np.zeros(step_num)#the reward of each step
    if_optimal = np.zeros(step_num)#1 for the optimal action;0 for other actions
    for i in range(step_num):
        if random.random() > eps:#random.random() returns a number in [0,1) randomly
            A = np.argmax(Q)#A is index of the first max element in Q
            if value[A] == np.max(value):
                if_optimal[i] = 1
        else:
            A = random.randint(0, 9)#A is a random integer from 0 to 9
            if value[A] == np.max(value):
                if_optimal[i] = 1
        R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
        reward[i] = R
        
        Q[A] = Q[A] + (R - Q[A]) * alpha
    return [reward, if_optimal]

def OneRun_2(arm_num, eps, step_num):
    value = np.random.normal(0, 1, arm_num)#mu=0, sigma=1;the value for each arm
    Q = np.zeros(arm_num)#the estimate of value for each arm
    N = np.zeros(arm_num)#the numbers of exploitations for each arm
    reward = np.zeros(step_num)#the reward of each step
    if_optimal = np.zeros(step_num)#1 for the optimal action;0 for other actions
    for i in range(step_num):
        if random.random() > eps:#random.random() returns a number in [0,1) randomly
            A = np.argmax(Q)#A is index of the first max element in Q
            if value[A] == np.max(value):
                if_optimal[i] = 1
        else:
            A = random.randint(0, 9)#A is a random integer from 0 to 9
            if value[A] == np.max(value):
                if_optimal[i] = 1
        R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
        reward[i] = R
        N[A] = N[A] + 1
        Q[A] = Q[A] + (R - Q[A])/N[A]
    return [reward, if_optimal]

repeat_num = 2000
arm_num = 10
eps = np.array([0, 0.1])
alpha = 0.1
step_num = 1000
ave_reward = np.zeros((2,step_num))
rate_optimal = np.zeros((2,step_num))

for i in range(repeat_num):
    [a, b] = OneRun_1(arm_num, eps[0], step_num, alpha)
    ave_reward[0,:] = ave_reward[0,:] + a
    rate_optimal[0,:] = rate_optimal[0,:] + b
ave_reward[0,:] = ave_reward[0,:] / repeat_num
rate_optimal[0,:] = rate_optimal[0,:] / repeat_num

for i in range(repeat_num):
    [a, b] = OneRun_2(arm_num, eps[1], step_num)
    ave_reward[1,:] = ave_reward[1,:] + a
    rate_optimal[1,:] = rate_optimal[1,:] + b
ave_reward[1,:] = ave_reward[1,:] / repeat_num
rate_optimal[1,:] = rate_optimal[1,:] / repeat_num

step = range(step_num)
plt.plot(step, ave_reward[0,:], label='$\\alpha = 0.1, Q_1 = 5, \epsilon = 0$', color='r')
plt.plot(step, ave_reward[1,:], label='$\\alpha = 1/n, Q_1 = 0, \epsilon = 0.1$', color='g')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('Average_reward.png')
plt.show()

plt.plot(step, rate_optimal[0,:], label='$\\alpha = 0.1, Q_1 = 5, \epsilon = 0$', color='r')
plt.plot(step, rate_optimal[1,:], label='$\\alpha = 1/n, Q_1 = 0, \epsilon = 0.1$', color='g')
plt.xlabel('Steps')
plt.ylabel('rate of optimal action')
plt.legend()
plt.savefig('Rate_optimal.png')
#plt.show()