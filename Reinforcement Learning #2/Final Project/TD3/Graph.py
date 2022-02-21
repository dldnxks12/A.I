import numpy as np
import gym
import sys
import random
from time import sleep
import matplotlib.pyplot as plt
from IPython.display import clear_output

data1 = np.load("./TD3_ensemble_model v4_Softmax_Converge.npy")
data1 = np.transpose(data1, (1,0))
data2 = np.load("./TD3_ensemble_model v5_Softmax_Converge.npy")
data2 = np.transpose(data2, (1,0))
length = np.arange(len(data2[0]))

plt.figure()
plt.title("Soft Max Value")
plt.plot(length, data2[0], label = 'Action1 Soft Max Value')
plt.plot(length, data2[1], label = 'Action2 Soft Max Value')
plt.plot(length, data2[2], label = 'Action3 Soft Max Value')
plt.xlabel("Episode")
plt.ylabel("Soft Max Value")
plt.legend()
plt.show()

'''
data = np.load("./Softmax_result.npy")
data = np.transpose(data, (1,0))
#print(data)
#sys.exit()
length = np.arange(len(data[0]))
#print(length)
#sys.exit()
plt.figure()
plt.title("Soft Max Value")
plt.plot(length, data[0], label = 'Action1 Soft Max Value')
plt.plot(length, data[1], label = 'Action2 Soft Max Value')
plt.plot(length, data[2], label = 'Action3 Soft Max Value')
plt.xlabel("Episode")
plt.ylabel("Soft Max Value")
plt.legend()
plt.show()
'''
'''

data  = np.load("./Pendulum single.npy")
data2 = np.load("./Pendulum.npy")
length = np.arange(len(data[0]))
plt.figure()
plt.title("TD3")
plt.plot(length, data , label = 'Single Model')
plt.plot(length, data2, label = 'Ensemble Model')
plt.xlabel("Episode")
plt.ylabel("10 Avg Reward")
plt.legend()

plt.show()
'''