The algorithm used here is partially from page 121 (5.9) with my incremental implementation as said in Exercise 5.9.
Different from the algorithm in folder "racetrack_2", I append the discounting-aware weighted importance-sampling here.
The result is good overall, 
    1, It can learn a policy, under which it can cross the finish line in some position on the starting line, even only 
    experienced 1000 episodes.
    2, It learns more fast than former algorithms, in 7h37m, it can run through 10000000 episodes whereas the former 
    algorithms can only go through 300000 episodes.
to se the result, you need to run the script "plot_trajectory_3.py", you can also choose the policy by selecting the PI 
in that script.