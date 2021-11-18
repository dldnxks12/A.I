import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


dataset = np.loadtxt("csv.file", delimiter = ',', dtype = np.float32)

X = torch.FloatTensor(dataset[:, :-1])
y = torch.LongTensor(dataset[:, -1])

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(16, 7)

    def forward(self, X):
        out = self.layer(X)
        return out


model = Logistic()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(10):

    hypothesis = model(X)
    cost = nn.CrossEntropyLoss(hypothesis, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 1 == 0:

        prediction = torch.argmax(hypothesis, axis = 1)
        corr = (prediction == y).sum() / len(prediction)
        print(f"Correct {corr} ")

