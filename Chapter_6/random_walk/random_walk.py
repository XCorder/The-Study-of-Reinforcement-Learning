#python3 script
#Random Walk(Example 6.2): I do this programme in order to find the reason why the RMS error
#of the TD method seems to go down and then up again.
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
#import copy

start = time.time()

V = np.zeros(7)# V[0] is of the left terminal state, V[6] is of the right terminal state, 
# V[1]~V[5] if of the state A~E
V[6] = 1
V_true = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])
MaxEpisode = 200
alpha = 0.15
Repeat = 1000
RMS_error = np.zeros(MaxEpisode)
for r in range(Repeat):
    #V = np.array([0.,0.,0.,0.,0.,0.,1.]) #down
    #V = np.array([0.,0.5,0.5,0.5,0.5,0.5,1.]) #down and up
    #V = np.array([0, 1., 2/6, 3/6, 4/6, 0., 1]) #down and up
    #V = np.array([0.,0.25,0.25,0.25,0.25,0.25,1.]) #down
    #V = np.array([0.,0.4,0.4,0.4,0.4,0.4,1.]) #down
    #V = np.array([0.,0.4,0.4,0.5,0.4,0.4,1.]) #down and up(slightly)
    V = np.array([0.,0.6,0.6,0.5,0.4,0.4,1.]) #down and up
    #V = np.array([0.,0.4,0.4,0.5,0.6,0.6,1.]) #down and up
    #V = np.array([0.,1/6,1/6,1/6,1/6,1/6,1.]) #down
    #V = np.array([0.,2/6,2/6,2/6,2/6,2/6,1.]) #down
    #V = np.array([0.,1.,1.,1.,1.,1.,1.]) #down
    #V = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]) #up
    #V = np.array([0., 5/6, 4/6, 3/6, 2/6, 1/6, 1.]) #down and up
    #V = np.array([0., 1., 1., 0.5, 0., 0., 1.]) #down and up
    #V = np.array([0., 0., 0., 0.5, 1., 1., 1.]) #down 
    #V = np.array([0.,0.3,0.4,0.5,0.6,0.7,1.]) #down and up
    #V = np.array([0.,0.2,0.35,0.5,0.65,0.8,1.]) #up
    for i in range(MaxEpisode):
        state = 3
        while state != 0 and state != 6:
            if random.random() < 0.5:
                state_1 = state - 1
            else:
                state_1 = state + 1
            V[state] = V[state] + alpha*(V[state_1]-V[state])
            state = state_1
        RMS_error[i] = RMS_error[i] + np.linalg.norm(V - V_true)/math.sqrt(5)
RMS_error = RMS_error/Repeat
print('RMS_error[min] = '+str(np.min(RMS_error)))

episode = range(1,MaxEpisode+1)
plt.ylim(0, 0.2)
plt.plot(episode, RMS_error, label='$\\alpha$ = '+str(alpha))
plt.xlabel('Episodes')
plt.ylabel('Empirical RMS error')
plt.legend()
plt.show()

end = time.time()
print('time cost: '+str((end-start)//3600)+'h'+str(((end-start)%3600)//60)+'m'+str(((end-start)%3600)%60)+'s')