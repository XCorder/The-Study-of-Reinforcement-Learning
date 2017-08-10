#python3 script
#Racetrack(Exercise 5.8): train and save an optimal policy
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

start = time.time()

def VelocityToActionList(v_x, v_y):
    if v_x < 0 or v_x > 4:
        print('v_x out 0~4')
        exit(1)
    if v_y < 0 or v_y > 4:
        print('v_y out 0~4')
        exit(1)
    if v_x == 0:
        if v_y == 0:
            action_list = np.array([[1,2], [2,1], [2,2]])
        elif v_y == 1:
            action_list = np.array([[1,1], [1,2], [2,0], [2,1], [2,2]])
        elif v_y == 2 or v_y == 3:
            action_list = np.array([[1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
        elif v_y == 4:
            action_list = np.array([[1,0], [1,1], [2,0], [2,1]])
    elif v_x == 1:
        if v_y == 0:
            action_list = np.array([[0,2], [1,1], [1,2], [2,1], [2,2]])
        elif v_y == 1:
            action_list = np.array([[0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
        elif v_y == 2 or v_y == 3:
            action_list = np.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
        elif v_y == 4:
            action_list = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]])
    elif v_x == 2 or v_x == 3:
        if v_y == 0:
            action_list = np.array([[0,1], [0,2], [1,1], [1,2], [2,1], [2,2]])
        elif v_y == 1 or v_y == 2 or v_y == 3:
            action_list = np.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
        elif v_y == 4:
            action_list = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]])
    elif v_x == 4:
        if v_y == 0:
            action_list = np.array([[0,1], [0,2], [1,1], [1,2]])
        elif v_y == 1 or v_y == 2 or v_y == 3:
            action_list = np.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]])
        elif v_y == 4:
            action_list = np.array([[0,0], [0,1], [1,0], [1,1]])
    return action_list

def OneStep(state, PI, eps_mu, p_v_dont_change, track, h, w, starting_line, fin_x, fin_y1, fin_y2):
    action_list = VelocityToActionList(state[2], state[3])
    if random.random() > eps_mu:
        action = PI[state[0], state[1], state[2], state[3], :]
    else:
        action = action_list[np.random.randint(action_list.shape[0]), :]

    if np.all(action == PI[state[0], state[1], state[2], state[3], :]):
        MU_action = (1 - eps_mu) + eps_mu/(action_list.shape[0])
    else:
        MU_action = eps_mu/(action_list.shape[0])

    next_state = np.zeros(4, dtype=np.int8)
    if random.random() > p_v_dont_change:
        next_state[2] = state[2] + action[0] - 1
        next_state[3] = state[3] + action[1] - 1
    else:
        next_state[2] = state[2]
        next_state[3] = state[3]
    next_state[1] = state[1] + next_state[2]# horizontal coordinate
    next_state[0] = state[0] - next_state[3]# vertical coordinate

    if next_state[0] < 0 or next_state[0] >= h or next_state[1] < 0 or next_state[1] >= w or track[next_state[0], next_state[1]] == -1:
        if next_state[1] > fin_x:
            intersection = (next_state[0] - state[0])/(next_state[1] - state[1]) * (fin_x - state[1]) + state[0]
            if intersection >= fin_y1 and intersection <= fin_y2:
                next_state = 'Terminal'
            else:
                next_state = tuple(starting_line[np.random.randint(starting_line.shape[0]), :]) + (0, 0)
        else:
            next_state = tuple(starting_line[np.random.randint(starting_line.shape[0]), :]) + (0, 0)
    else:
        next_state = tuple(next_state)

    return tuple(action), MU_action, next_state

track = np.load('track_1.npy')
h, w = track.shape
starting_line = np.argwhere(track == 1)#<class 'numpy.ndarray'>, each row is a starting position
finish_line = np.argwhere(track == 2)
fin_x = finish_line[0, 1]# the x coordinate of the finish line
fin_y1 = np.min(finish_line[:, 0]) # the topmost y coordinate of the finish line
fin_y2 = np.max(finish_line[:, 0]) # the undermost y coordinate of the finish line
Q = np.zeros(track.shape+(5,5)+(3,3))
#s: track.shape: position(vertical + horizontal)
#   (5,5): velocity(right+up), and can't move left or down
#a: (3,3): acceleration(horizontal+vertical): 0, minus 1; 1, do nothing; 2, plus 1
C = np.zeros(Q.shape)

PI = np.zeros(track.shape+(5,5)+(2,), dtype=np.int8)#target policy
PI = np.load('policy_1_eps_mu_0.3_episode_150000.npy')
# for i in range(h):# initial PI
#     for j in range(w):
#         for v_x in range(5):
#             for v_y in range(5):
#                 action_list = VelocityToActionList(v_x, v_y)
#                 PI[i, j, v_x, v_y, :] = action_list[np.random.randint(action_list.shape[0]), :]

eps_mu = 0.3# behavior policy is the 'eps_mu'-soft of the current target policy 
p_v_dont_change = 0
MaxEpisode = 300000#the max number of episodes
MaxTimestep = 100000# the max number of timesteps for each episodes
cnt_episode = 0
while True:
    cnt_episode = cnt_episode + 1
    episode = []
    state = tuple(starting_line[np.random.randint(starting_line.shape[0]), :]) + (0, 0)
    #choose a state from starting line randomly with 0 velocity
    timestep = 0
    while True:
        timestep = timestep + 1
        action, MU_action, next_state = OneStep(state, PI, eps_mu, p_v_dont_change, track, h, w, starting_line, fin_x, fin_y1, fin_y2)
        episode.append((state, action, MU_action))
        state = next_state
        if state == 'Terminal':
            episode.append(state)
            break
        if timestep > MaxTimestep:
            print('occur an episode of whitch length is over ' + str(MaxTimestep))
            break

    if episode.pop() == 'Terminal':
        G = 0
        W = 1.0
        while episode:
            state, action, MU_action = episode.pop()
            G = G - 1
            C[state+action] = C[state+action] + W
            Q[state+action] = Q[state+action] + (W / C[state+action]) * (G - Q[state+action])

            action_list = VelocityToActionList(state[2], state[3])
            A_greedy = action_list[0, :]
            q_max = Q[state[0],state[1],state[2],state[3],A_greedy[0],A_greedy[1]]
            for i in range(action_list.shape[0]):
                if Q[state[0],state[1],state[2],state[3],action_list[i, 0],action_list[i, 1]] > q_max:
                    A_greedy = action_list[i, :]
                    q_max = Q[state[0],state[1],state[2],state[3],action_list[i, 0],action_list[i, 1]]
            # A_greedy_position = np.argwhere(Q[state[0],state[1],state[2],state[3],:,:] == np.max(Q[state[0],state[1],state[2],state[3],:,:]))
            # A_greedy = tuple(A_greedy_position[np.random.randint(A_greedy_position.shape[0]), :])

            PI[state[0],state[1],state[2],state[3], :] = A_greedy
            if tuple(A_greedy) != action:
                break
            W = W / MU_action
    else:
        cnt_episode = cnt_episode - 1

    if cnt_episode >= MaxEpisode:
        break



np.save('policy_1_eps_mu_'+str(eps_mu)+'_pvdontch_'+str(p_v_dont_change)+'_episode_'+str(MaxEpisode)+'.npy', PI)
np.save('Q_of_policy_1_eps_mu_'+str(eps_mu)+'_pvdontch_'+str(p_v_dont_change)+'_episode_'+str(MaxEpisode)+'.npy', Q)
np.save('C_of_policy_1_eps_mu_'+str(eps_mu)+'_pvdontch_'+str(p_v_dont_change)+'_episode_'+str(MaxEpisode)+'.npy', C)

end = time.time()
print('time cost: '+str((end-start)//3600)+'h'+str(((end-start)%3600)//60)+'m'+str(((end-start)%3600)%60)+'s')