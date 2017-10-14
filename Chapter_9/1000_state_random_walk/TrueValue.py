#python3 script
#the true state value function for 1000-state Random Walk(Example 9.1)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy

start_time = time.time()

S = np.zeros(1200)
S[:100] = -1
S[1100:] = 1
theta = 1e-5
while True:
    diff = 0
    for i in range(100, 1100):
        tem = (np.sum(S[i-100:i])+np.sum(S[i+1:i+101]))/200
        if abs(S[i]-tem)>diff:
            diff = abs(S[i]-tem)
        S[i] = tem
    if diff<theta:
        break
np.save('TrueValue.npy', S[100:1100])

plt.plot(range(1,1001), S[100:1100])
plt.show()

end_time = time.time()
print('time cost: '+str((end_time-start_time)//3600)+'h' \
      +str(((end_time-start_time)%3600)//60)+'m' \
      +str(((end_time-start_time)%3600)%60)+'s')