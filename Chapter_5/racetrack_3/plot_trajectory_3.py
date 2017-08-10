#python3 script
#Racetrack(Exercise 5.9): plot the trajectory
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

pure_track = np.load('track_1.npy')
track = copy.deepcopy(pure_track)
h, w = track.shape
starting_line = np.argwhere(track == 1)#<class 'numpy.ndarray'>, each row is a starting position
finish_line = np.argwhere(track == 2)
fin_x = finish_line[0, 1]# the x coordinate of the finish line
fin_y1 = np.min(finish_line[:, 0]) # the topmost y coordinate of the finish line
fin_y2 = np.max(finish_line[:, 0]) # the undermost y coordinate of the finish line

# PI = np.load('policy_3_track_1_eps_mu_0.3_pvdontch_0_episode_1000_gamma_0.9.npy')
# PI = np.load('policy_3_track_1_eps_mu_0.3_pvdontch_0_episode_10000_gamma_0.9.npy')
# PI = np.load('policy_3_track_1_eps_mu_0.3_pvdontch_0_episode_600000_gamma_0.9.npy')
# PI = np.load('policy_3_track_1_eps_mu_0.3_pvdontch_0_episode_100000_gamma_0.9.npy')
PI = np.load('policy_3_track_1_eps_mu_0.2_pvdontch_0_episode_100000_gamma_0.9.npy')
# PI = np.load('policy_3_track_1_eps_mu_0.2_pvdontch_0_episode_10000_gamma_0.9.npy')
# PI = np.load('policy_3_track_1_eps_mu_0.2_pvdontch_0_episode_1000_gamma_0.9.npy')
# PI = np.load('policy_3_track_1_eps_mu_0.2_pvdontch_0_episode_10000000_gamma_0.9.npy')

numCols = 3
numRows = math.ceil(starting_line.shape[0] / numCols)
for i in range(starting_line.shape[0]):
    track = copy.deepcopy(pure_track)
    next_state = np.concatenate((starting_line[i, :], np.array([0,0])))
    while True:
        state = copy.deepcopy(next_state)
        track[state[0], state[1]] = 3
        action = PI[state[0], state[1], state[2], state[3], :]
        next_state[2] = state[2] + action[0] - 1
        next_state[3] = state[3] + action[1] - 1
        next_state[1] = state[1] + next_state[2]# horizontal coordinate
        next_state[0] = state[0] - next_state[3]# vertical coordinate
        if next_state[0] < 0 or next_state[0] >= h or next_state[1] < 0 or next_state[1] >= w or track[next_state[0], next_state[1]] == -1:
            if next_state[1] > fin_x:
                intersection = (next_state[0] - state[0])/(next_state[1] - state[1]) * (fin_x - state[1]) + state[0]
                if intersection >= fin_y1 and intersection <= fin_y2:
                    result = 'crossed finish line'
                    break
                else:
                    result = 'hit the track boundary'
                    break
            else:
                result = 'hit the track boundary'
                break

    grid = np.zeros((h, w))
    for ii in range(h):
        for j in range(w):
            if track[ii,j]==0:
                grid[ii,j] = 0.14
            elif track[ii,j]==-1:
                grid[ii,j] = 0.99
            elif track[ii,j]==1:
                grid[ii,j] = 0.54
            elif track[ii,j]==2:
                grid[ii,j] = 0.34
            elif track[ii,j]==3:
                grid[ii,j] = 0

    plt.subplot(numRows, numCols, i+1)
    plt.imshow(grid, cmap='tab20c')
    plt.axis('off')
    plt.title(result)

# plt.savefig('policy_3_track_1_eps_mu_0.2_pvdontch_0_episode_10000000_gamma_0.9.png', dpi=200)
plt.show()