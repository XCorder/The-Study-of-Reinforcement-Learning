#python3 script
#Gambler's Problem(Example 4.3 & Exercise 4.9)
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math

start = time.time()

def ValueIteration(V, theta, MaxIter, p_h):
    cnt = 0
    while True:
        cnt = cnt + 1
        V_old = V
        V = np.zeros(101)
        V[100] = 1
        for i in range(1,100):
            Q = p_h*V_old[i : i+min(i, 100-i)+1] + (1-p_h)*V_old[i-101: i-min(i, 100-i)-1-101:-1]
            V[i] = np.max(Q)
        delta = np.max(np.abs(V-V_old))
        if (delta < theta) or (cnt >= MaxIter):
            PI_1 = np.zeros(99, dtype=np.int8)
            PI_2 = np.zeros(99, dtype=np.int8)
            for i in range(1,100):
                Q = p_h*V[i : i+min(i, 100-i)+1] + (1-p_h)*V[i-101: i-min(i, 100-i)-1-101:-1]
                PI_1[i-1] = np.argmax(Q)
                PI_2[i-1] = Q.size-1 - np.argmax(Q[::-1])
            break     
    return (V, PI_1, PI_2, cnt)

V = np.zeros(101)
V[100] = 1
theta = 1e-10
MaxIter = 1e5
p_h = 0.55
(V, PI_1, PI_2, cnt) = ValueIteration(V, theta, MaxIter, p_h)
print ('times of iteration is', cnt)

plt.plot(range(1,V.size-1), V[1:V.size-1])
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.savefig('VE_ph_'+str(p_h)+'.png')
plt.show()

plt.bar(range(1,PI_2.size+1), PI_2, label='most aggressive optimal policy', color='c', width=1)
plt.bar(range(1,PI_1.size+1), PI_1, label='most careful optimal policy', color='m', width=1)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.legend()
plt.savefig('FP_ph_'+str(p_h)+'.png')
plt.show()

end = time.time()
print('time cost: '+str(end-start)+'s')