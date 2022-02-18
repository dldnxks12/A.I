import numpy as np
import gym
import random
from time import sleep
import matplotlib.pyplot as plt
from IPython.display import clear_output

data  = np.load("./ddpg_con_save1.npy")
data2 = np.load("./ddpg_dis_save1.npy")

#print(data.shape)
#print(data2.shape)

length = np.arange(len(data))
plt.figure()
plt.title("DDPG")
plt.plot(length, data , label = 'Continuous')
plt.plot(length, data2, label = 'Discrete')
plt.xlabel("Episode")
plt.ylabel("10 Avg Reward")
plt.legend()
plt.show()
