import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

data = np.loadtxt("data_linear_regression.csv", delimeter=',', dtype = np.float32)

X = torch.FloatTensor(data[:, :-1])
y = torch.FloatTensor(data[:,[-1]])

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(15):

    hypothesis = model(X)
    cost = F.mse_loss(X, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()






