import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(777)

data = np.loadtxt("C:/Users/USER/PycharmProjects/A.I/Deep Learning/# AI Application/csv files/data_logistic_regression.csv", delimiter=',')

print(data.shape)
x_train = torch.FloatTensor(data[:,:-1])
y_train = torch.FloatTensor(data[:,[-1]])

class Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = Logistic(8, 1)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(3000):

    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:

        prediction = hypothesis >= torch.FloatTensor([0.5])
        Acc = (prediction.float() == y_train).sum().item() / len(prediction)

        print(f"Epoch {epoch}, Cost {cost.item()}, ACC {Acc*100}")


