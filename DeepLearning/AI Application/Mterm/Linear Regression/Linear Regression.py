import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load Data
dataset = np.loadtxt('data_linear_regression.csv', delimiter=',', dtype = np.float32)

print("dataset Shape", dataset.shape)  # 25 x 4

np.random.shuffle(dataset)

X_train = torch.FloatTensor(dataset[:,:-1]) # 25 x 3
Y_train = torch.FloatTensor(dataset[:,[-1]])  # 25 x 1

# Make model
model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr = 0.00001)

for epoch in range(5000):

    hypothesis = model(X_train)
    cost = F.mse_loss(hypothesis, Y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        params = list(model.parameters())
        weight = params[0]
        bias = params[1]

        print(f"Cost : {cost.item()} , Weight : {weight}, Bias : {bias}")



