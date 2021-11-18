    import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

dataset = np.loadtxt("csvfile.csv", delimiter = ',', dtype = np.float32)

X = torch.FloatTensor(dataset[:, :-1])
y = torch.FloatTensor(dataset[:, [-1]])

model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr  = 0.01)

params = list(model.parameters())

for epoch in range(10):

    hypothesis = model(X)
    cost = F.mse_loss(hypothesis, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Weight : {params[0].item()} , Bias : {params[1].item()}")


