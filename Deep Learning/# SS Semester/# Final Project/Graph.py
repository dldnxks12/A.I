import numpy as np
import sys
import matplotlib.pyplot as plt

# Baseline Model Output
data1 = np.load("./Baseline.npy")
data2 = np.load("./Baseline2.npy")
data3 = np.load("./Baseline3.npy")

length = np.arange(len(data1))

plt.figure()
plt.title("Reward Graph")
plt.plot(length, data1, label = 'Baseline1')
plt.plot(length, data2, label = 'Baseline2')
plt.plot(length, data3, label = 'Baseline3')
plt.xlabel("Episode")
plt.ylabel("10 Average Reward")
plt.legend()
plt.show()
