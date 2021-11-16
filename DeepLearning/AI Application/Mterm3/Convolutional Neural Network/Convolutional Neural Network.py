import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.dataset as dsets
import torchvision.transforms as transform


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print("Device", device)

mnist_train = dsets.MNIST(root = "MNIST_DATA/", train = True, transform = transform.ToTensor(), download = False)
mnist_test = dsets.MNIST(root = "MNIST_DATA/", train = False, transform = transform.ToTensor(), download = False)

batch_size = 100
dataloader = DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =2, stride = 2)

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2)
        )

        self.fc = nn.Linear(3*3*64, 10)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


learning_rate = 0.01
training_epochs = 15

model = CNN.to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(training_epochs):
    avg_cost = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        hypothesis = model(X)
        Cost = nn.CrossEntropyLoss(hypothesis, y)

        optimizer.zero_grad()
        Cost.backward()
        optimizer.step()

        avg_cost += Cost.item() / len(dataloader)

    print(f"Epoch {epoch} | Cost {Cost.item()}")

print("Learning Finished")

with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    prediction = torch.argmax(prediction, axis = 1)

    correct = (prediction == Y_test).sum() / len(prediction)

    print("Correct : ", correct)


with torch.no_grad():
    r = random.randint(0, len(mnist_test) -1)

    X_sample = mnist_test.data[r].view(1, 1, 28, 28).float().to(device)
    Y_sample = mnist_test.targets[r].to(device)

    hypothesis = model(X_sample)

    corr = torch.argmax(hypothesis, axis = 1)

    print("Correction", corr)

    