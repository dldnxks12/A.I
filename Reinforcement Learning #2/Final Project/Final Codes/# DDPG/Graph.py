import numpy as np
import sys
import matplotlib.pyplot as plt

# Baseline DDPG Model Output
data1 = np.load("./single type1.npy")
data2 = np.load("./single type3.npy")
data3 = np.load("./single type4.npy")
data4 = np.load("./single type5.npy")
data5 = np.load("./single type6.npy")
data6 = np.load("./single type7.npy")
data7 = np.load("./single type8.npy")
data8 = np.load("./single type9.npy")

length = np.arange(len(data1))

plt.figure()
plt.title("DDPG Single Model")

plt.plot(length, data1, label = 'Test 1')
plt.plot(length, data2, label = 'Test 2')
plt.plot(length, data3, label = 'Test 3')
plt.plot(length, data4, label = 'Test 4')
plt.plot(length, data5, label = 'Test 5')
plt.plot(length, data6, label = 'Test 6')
plt.plot(length, data7, label = 'Test 7')
plt.plot(length, data8, label = 'Test 8')

plt.xlabel("Episode")
plt.ylabel("10 MVA Rewards")
plt.legend()
plt.show()
