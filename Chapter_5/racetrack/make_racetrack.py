#python3 script
#Racetrack(Exercise 5.8): produce and save a racetrack
#author: Xiang Chao

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math

track = np.zeros((32, 17), dtype=np.int8)
track[31, 3:9] = 1#starting line
track[:6, 16] = 2#finish line
track[0, :3] = -1
track[1, :2] = -1
track[2, :2] = -1
track[3, 0] = -1
track[14:, 0] = -1
track[22:, 1] = -1
track[29:, 2] = -1
track[6:, 10:] = -1
track[7:, 9] = -1

np.save('track_1.npy', track)