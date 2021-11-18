import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

data = np.loadtxt("./csv files/data_linear_regression.csv", delimiter = ',', dtype = np.float32)
np.random.shuffle(data)

X = torch.FloatTensor(data[:, :-1])
y = torch.FloatTensor(data[:,-1])

model = nn.Linear(3, 1)  # Feature 3개 output 1개
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(100):
    hypothesis = model(X)
    cost = F.mse_loss(hypothesis, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    if epoch % 10 == 0:
        print(f" cost : {cost.item()}")







