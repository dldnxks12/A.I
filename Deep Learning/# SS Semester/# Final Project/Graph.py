import numpy as np
import sys
import matplotlib.pyplot as plt

# Baseline Model Output
Baseline_data1 = np.load("./Baseline.npy")
Baseline_data2 = np.load("./Baseline2.npy")
Baseline_data3 = np.load("./Baseline3.npy")

Modified_model_data1 = np.load("./type22.npy")
Modified_model_data2 = np.load("./type3.npy")
Modified_model_data3 = np.load("./type4.npy")
#Modified_model_data4 = np.load("./type22.npy")

length = np.arange(len(Baseline_data1))

plt.figure()
plt.title("Reward Graph")
plt.plot(length, Baseline_data1, label = '#DDPG_Baseline1')
plt.plot(length, Baseline_data2, label = '#DDPG_Baseline2')
plt.plot(length, Baseline_data3, label = '#DDPG_Baseline3')
plt.plot(length, Modified_model_data1, label = '#Modified_Model1')
plt.plot(length, Modified_model_data2, label = '#Modified_Model2')
plt.plot(length, Modified_model_data3, label = '#Modified_Model3')

plt.xlabel("Episode")
plt.ylabel("10 Average Reward")
plt.legend()
plt.show()
