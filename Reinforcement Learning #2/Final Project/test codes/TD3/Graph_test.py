import numpy as np
import gym
import sys
import random
from time import sleep
import matplotlib.pyplot as plt

# Baseline DDPG Model Output
#data1 = np.load("./Final Result.npy")
data2 = np.load("./Final Result Softmax.npy")
data2 = data2.transpose()
#data3 = np.array([0, -114.1 ,-110.1 ,-102.2 ,-84.3 ,-55.4 ,-38.1 ,-33.6 ,-21.8 ,-14.4 ,-59.3 ,-45.6 ,-35.6 ,25.3 ,25.5 ,250.3 ,281.5 ,244.2 ,246.4 ,289.4 ,248.0 ,294.1 ,295.4 ,296.1 ,251.7 ,301.2 ,299.7 ,298.0 ,296.0 ,254.4 ,293.0 ,296.2 ,295.5 ,266.9 ,294.9 ,281.2 ,173.8 ,-114.4 ,270.6 ,288.5 ,288.5 ,293.3 ,295.7 ,254.1 ,297.2 ,298.5 ,294.1 ,293.2 ,288.3 ,295.7 ])
#data1[:10] = 0
print(data2.shape)
#length = np.arange(len(data1))
#length2 = np.arange(len(data3))
length3 = np.arange(len(data2[0]))*10
plt.figure()
plt.title("Soft Max Value")

plt.plot(length3, data2[0], label = 'Action1 SoftMax Value')
plt.plot(length3, data2[1], label = 'Action2 SoftMax Value')
plt.plot(length3, data2[2], label = 'Action3 SoftMax Value')
plt.plot(length3, data2[3], label = 'Action4 SoftMax Value')
plt.plot(length3, data2[4], label = 'Action5 SoftMax Value')
plt.plot(length3, data2[5], label = 'Action6 SoftMax Value')
plt.plot(length3, data2[6], label = 'Action7 SoftMax Value')
plt.plot(length3, data2[7], label = 'Action8 SoftMax Value')
plt.xlabel("Time Steps")
plt.ylabel("Soft Max Value")
plt.legend()
plt.show()

'''
plt.figure()
plt.title("Bi Pedal Walker")
plt.plot(length2, data3, label = 'TD3 Ensemble', color = 'r')
plt.xlabel("Episode")
plt.ylabel("10 Average Reward")
plt.legend()

plt.show()


plt.figure()
plt.title("Soft Max Value")
plt.plot(length, data1, label = 'Base Model')
plt.xlabel("Episode")
plt.ylabel("10 Average Reward")
plt.legend()
plt.show()
'''