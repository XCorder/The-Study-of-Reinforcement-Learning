#python3 script
#Jack's Car Rental(Example 4.2)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math

start = time.time()

def Poisson(lamb, n):
    return (lamb**n)/(math.factorial(n)) * math.exp(-lamb)

def PolicyEvaluation(V, PI, theta, MaxIteration):#the V input is the initial value for this evaluation
    cnt = 0
    while True:
        cnt = cnt+1
        delta = 0
        for i in range(21):
            for j in range(21):
                v = V[i,j]
                s_begin_1 = i-PI[i,j]# the begining state of daytime in location 1
                s_begin_2 = j+PI[i,j]# the begining state of daytime in location 2
                r_move = -2 * abs(PI[i,j])# the reward of moving cars
                r_1 = np.zeros(21)# the expected rental reward given each possible eventual state of daytime in location 1
                r_2 = np.zeros(21)# the expected rental reward given each possible eventual state of daytime in location 2
                p_1 = np.zeros(21)# the probability for each possible eventual state of daytime in location 1
                p_2 = np.zeros(21)# the probability for each possible eventual state of daytime in location 2

                pp_requested = 0
                for requested_1 in range(s_begin_1+1):# requested_1 means the number of cars requested in location 1
                    if requested_1==s_begin_1:
                        p_requested = 1 - pp_requested
                    else:
                        p_requested = Poisson(3, requested_1)
                        pp_requested = pp_requested + p_requested
                    returned_max = 20 - (s_begin_1-requested_1)
                    pp_returned = 0
                    for returned_1 in range(returned_max+1):# returned_1 means the number of cars returned in location 1
                        if returned_1==returned_max:
                            p_returned = 1 - pp_returned
                        else:
                            p_returned = Poisson(3, returned_1)
                            pp_returned = pp_returned + p_returned
                        s_now = s_begin_1 - requested_1 + returned_1
                        p_1[s_now] = p_1[s_now] + p_requested*p_returned
                        r_1[s_now] = r_1[s_now] + p_requested*p_returned*(10*requested_1)
                r_1 = r_1 / p_1

                pp_requested = 0
                for requested_2 in range(s_begin_2+1):# requested_2 means the number of cars requested in location 2
                    if requested_2==s_begin_2:
                        p_requested = 1 - pp_requested
                    else:
                        p_requested = Poisson(4, requested_2)
                        pp_requested = pp_requested + p_requested
                    returned_max = 20 - (s_begin_2-requested_2)
                    pp_returned = 0
                    for returned_2 in range(returned_max+1):# returned_2 means the number of cars returned in location 2
                        if returned_2==returned_max:
                            p_returned = 1 - pp_returned
                        else:
                            p_returned = Poisson(2, returned_2)
                            pp_returned = pp_returned + p_returned
                        s_now = s_begin_2 - requested_2 + returned_2
                        p_2[s_now] = p_2[s_now] + p_requested*p_returned
                        r_2[s_now] = r_2[s_now] + p_requested*p_returned*(10*requested_2)
                r_2 = r_2 / p_2
                r = np.reshape(r_1, (21,1)) + np.reshape(r_2, (1,21)) 
                p = np.dot(np.reshape(p_1, (21,1)), np.reshape(p_2, (1,21)))
                V[i,j] = r_move + np.sum(p * (r + 0.9*V))#Gauss Seidel style
                delta = max(delta, abs(v - V[i,j]))
        if (cnt >= MaxIteration) or (delta < theta):
            break  
    return V

def PolicyImprovement(V):
    PI = np.zeros((21,21), dtype=np.int8)
    lamb = np.array([[3,3], [4,2]])
    for i in range(21):
        for j in range(21):
            a_min = max(-5, -j, -(20-i))
            a_max = min(5, i, 20-j)
            Q = np.zeros(a_max - a_min + 1)
            for a in range(a_min, a_max+1):# a means all possible actions
                s_begin = np.array([i-a, j+a])
                r_move = -2 * abs(a)
                r_rent = np.zeros((2,21))
                p_marginal = np.zeros((2,21))
                for k in range(2):# k means the 2 locations
                    pp_requested = 0
                    for requested in range(s_begin[k]+1):
                        if requested==s_begin[k]:
                            p_requested = 1 - pp_requested
                        else:
                            p_requested = Poisson(lamb[k, 0], requested)
                            pp_requested = pp_requested + p_requested
                        returned_max = 20 - (s_begin[k]-requested)
                        pp_returned = 0
                        for returned in range(returned_max+1):
                            if returned==returned_max:
                                p_returned = 1 - pp_returned
                            else:
                                p_returned = Poisson(lamb[k,1], returned)
                                pp_returned = pp_returned + p_returned
                            s_now = s_begin[k] - requested + returned
                            p_marginal[k,s_now] = p_marginal[k,s_now] + p_requested*p_returned
                            r_rent[k,s_now] = r_rent[k,s_now] + p_requested*p_returned*(10*requested)
                    r_rent[k,:] = r_rent[k,:] / p_marginal[k,:]
                r = np.reshape(r_rent[0,:], (21,1)) + np.reshape(r_rent[1,:], (1,21))
                p = np.dot(np.reshape(p_marginal[0,:], (21,1)), np.reshape(p_marginal[1,:], (1,21)))
                Q[a-a_min] = r_move + np.sum(p * (r + 0.9*V))
            PI[i,j] = a_min + np.argmax(Q)
    return PI

V = np.zeros((21,21))
PI = np.zeros((21,21), dtype=np.int8)
cnt = 0
MaxEvaluationIteration = 1000
theta = 0.001
MaxImproveIteration = 10
while True:
    cnt = cnt + 1
    V = PolicyEvaluation(V, PI, theta, MaxEvaluationIteration)
    PI_old = PI
    PI = PolicyImprovement(V)
    if (np.all(PI_old==PI)) or (cnt >= MaxImproveIteration):
        break
print (PI)
print ('times of improvement is', cnt)

end = time.time()
print('time cost: '+str(end-start)+'s')