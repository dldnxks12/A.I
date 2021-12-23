import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(777)

data = np.loadtxt("C:/Users/USER/PycharmProjects/A.I/DeepLearning/# AI Application/csv files/data_multinomial_classification.csv", delimiter=',')

x_train = torch.FloatTensor(data[:,:-1])
y_train = torch.LongTensor(data[:,-1])

class Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = Logistic(16, 7)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(3000):

    hypothesis = model(x_train)
    cost = F.cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:

        prediction = torch.argmax(hypothesis, axis = 1)
        Acc = (prediction == y_train).sum() / len(prediction)

        print(f"Epoch {epoch}, Cost {cost.item()}, ACC {Acc*100}")

