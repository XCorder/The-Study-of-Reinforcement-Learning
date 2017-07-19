#python3 script
#Gradient Bandit method timestep 100000
#author: Xiang Chao

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import random
#import math

def OneRun_1(arm_num, alpha, step_num):#with baseline
    #value = np.random.normal(0, 1, arm_num)#mu=0, sigma=1;the value for each arm
    value = np.random.normal(4, 1, arm_num)
    H = np.zeros(arm_num)#the preference for each arm
    N = np.zeros(arm_num)#the numbers of exploitations for each arm
    ave_R = 0#the average of all reward
    reward = np.zeros(step_num//100 + 1)#the reward of each step
    if_optimal = np.zeros(step_num//100 + 1)#1 for the optimal action;0 for other actions
    for i in range(step_num + 1):
        if i%100 == 0:
            Pr = np.exp(H)
            Pr = Pr / np.sum(Pr)
            A = np.random.choice(arm_num, p=Pr)
            if value[A] == np.max(value):
                if_optimal[i//100] = 1
            R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
            ave_R = ave_R + (R - ave_R)/(i+1)
            reward[i//100] = R
            for j in range(arm_num):
                if j == A:
                    H[j] = H[j] + alpha*(R-ave_R)*(1-Pr[j])
                else:
                    H[j] = H[j] - alpha*(R-ave_R)*Pr[j]
        else:
            Pr = np.exp(H)
            Pr = Pr / np.sum(Pr)
            A = np.random.choice(arm_num, p=Pr)
            R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
            ave_R = ave_R + (R - ave_R)/(i+1)
            for j in range(arm_num):
                if j == A:
                    H[j] = H[j] + alpha*(R-ave_R)*(1-Pr[j])
                else:
                    H[j] = H[j] - alpha*(R-ave_R)*Pr[j]
    return (reward, if_optimal)

def OneRun_2(arm_num, alpha, step_num):#without baseline
    #value = np.random.normal(0, 1, arm_num)#mu=0, sigma=1;the value for each arm
    value = np.random.normal(4, 1, arm_num)
    H = np.zeros(arm_num)#the preference for each arm
    N = np.zeros(arm_num)#the numbers of exploitations for each arm
    ave_R = 0#the average of all reward
    reward = np.zeros(step_num//100 + 1)#the reward of each step
    if_optimal = np.zeros(step_num//100 + 1)#1 for the optimal action;0 for other actions
    for i in range(step_num + 1):
        if i%100 == 0:
            Pr = np.exp(H)
            Pr = Pr / np.sum(Pr)
            A = np.random.choice(arm_num, p=Pr)
            if value[A] == np.max(value):
                if_optimal[i//100] = 1
            R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
            #ave_R = ave_R + (R - ave_R)/(i+1)
            reward[i//100] = R
            for j in range(arm_num):
                if j == A:
                    H[j] = H[j] + alpha*(R-ave_R)*(1-Pr[j])
                else:
                    H[j] = H[j] - alpha*(R-ave_R)*Pr[j]
        else:
            Pr = np.exp(H)
            Pr = Pr / np.sum(Pr)
            A = np.random.choice(arm_num, p=Pr)
            R = random.gauss(value[A], 1)#mu=value[A], sigma=1;the reward for arm A
            #ave_R = ave_R + (R - ave_R)/(i+1)
            for j in range(arm_num):
                if j == A:
                    H[j] = H[j] + alpha*(R-ave_R)*(1-Pr[j])
                else:
                    H[j] = H[j] - alpha*(R-ave_R)*Pr[j]
    return (reward, if_optimal)

repeat_num = 1000
arm_num = 10
alpha = np.array([0.1, 0.4, 0.1, 0.4])
step_num = 100000
ave_reward = np.zeros((4,step_num//100 +1))
rate_optimal = np.zeros((4,step_num//100 +1))

for j in range(2):#with baseline
    for i in range(repeat_num):
        a, b = OneRun_1(arm_num, alpha[j], step_num)
        ave_reward[j,:] = ave_reward[j,:] + a
        rate_optimal[j,:] = rate_optimal[j,:] + b
    ave_reward[j,:] = ave_reward[j,:] / repeat_num
    rate_optimal[j,:] = rate_optimal[j,:] / repeat_num
for j in range(2, 4):#without baseline
    for i in range(repeat_num):
        a, b = OneRun_2(arm_num, alpha[j], step_num)
        ave_reward[j,:] = ave_reward[j,:] + a
        rate_optimal[j,:] = rate_optimal[j,:] + b
    ave_reward[j,:] = ave_reward[j,:] / repeat_num
    rate_optimal[j,:] = rate_optimal[j,:] / repeat_num

step = range(step_num//100 + 1)
plt.plot(step, ave_reward[0,:], label='with baseline, $\\alpha = 0.1$', color='r')
plt.plot(step, ave_reward[1,:], label='with baseline, $\\alpha = 0.4$', color='g')
plt.plot(step, ave_reward[2,:], label='without baseline, $\\alpha = 0.1$', color='b')
plt.plot(step, ave_reward[3,:], label='without baseline, $\\alpha = 0.4$', color='c')
plt.xlabel('Steps(devided by 100)')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('Average_reward_ValueNear4_100000.png')
plt.close()

step = range(step_num//100 + 1)
plt.plot(step, rate_optimal[0,:], label='with baseline, $\\alpha = 0.1$', color='r')
plt.plot(step, rate_optimal[1,:], label='with baseline, $\\alpha = 0.4$', color='g')
plt.plot(step, rate_optimal[2,:], label='without baseline, $\\alpha = 0.1$', color='b')
plt.plot(step, rate_optimal[3,:], label='without baseline, $\\alpha = 0.4$', color='c')
plt.xlabel('Steps(devided by 100)')
plt.ylabel('rate of optimal action')
plt.legend()
plt.savefig('Rate_optimal_ValueNear4_100000.png')