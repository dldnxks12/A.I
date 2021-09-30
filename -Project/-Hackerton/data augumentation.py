import random
import pandas as pd
import numpy as np

train_data = pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_features.csv")
train_label = pd.read_csv("C:/Users/USER/Desktop/Hackerton/train_labels.csv")

# data split
act_list=train.iloc[:,2:].columns
acc_list=['acc_x','acc_y','acc_z']
gy_list=['gy_x','gy_y','gy_z']
print(act_list)