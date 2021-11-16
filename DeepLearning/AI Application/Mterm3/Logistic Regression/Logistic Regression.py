import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

dataset = np.loadtxt("", delimiter =',', dtype = np.float32)
X = torch.FloatTensor(dataset[:, :-1])
y = torch.LongTensor(dataset[:, -1])  # 이후 One hot Encoding 수행

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(16, 7)

    def forward(self, X):
        out = self.layer(X)

        return out

model = Logistic()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(15):

    hypothesis = model(X)
    cost = nn.CrossEntropyLoss(hypothesis, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:

        prediction = torch.argmax(hypothesis, axis = 1)
        correction = (prediction == y).sum() / len(prediction)

        print("Correction" , correction)


