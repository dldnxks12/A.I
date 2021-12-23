# Multi-nomial 이기 때문에 Sigmoid 대신 Softmax 사용할 것
# but softmax - one hot encoding이 cross entroy 함수에 모두 포함되어 있다.

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data = np.loadtxt("data_multinomial_classification.csv", delimeter=',', dtype = np.flaot32)

X = torch.FloatTensor(data[:, :-1])
y = torch.LongTensor(data[:, -1])

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.Linear = nn.Linear(16, 7)

    def forward(self, X):
        return self.Linear(X)


model = Logistic()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(15):

    hypothesis = model(X)
    cost = F.CrossEntropyLoss(hypothesis, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    