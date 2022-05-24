import numpy as np
import sys
import matplotlib.pyplot as plt

# Baseline DDPG Model Output
data1 = np.load("./BaseModel.npy")
data2 = np.load("./Rewards_base.npy")
length = np.arange(len(data1))

print("Total Reward : ", sum(data2))
plt.figure()
plt.title("Soft Max Value")
plt.plot(length, data1, label = 'Base Model')
plt.xlabel("Episode")
plt.ylabel("10 Average Reward")
plt.legend()
plt.show()
