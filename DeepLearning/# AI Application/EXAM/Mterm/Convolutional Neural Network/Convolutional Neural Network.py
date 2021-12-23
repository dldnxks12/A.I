# MNIST Data
# DataLoader

import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transform
import torchvision.datasets as dsets

from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print("Device", device)

mnist_train = dsets.MNIST(root = "MNIST_DATA/", train=True, transform = transform.ToTensor(), download=True)
mnist_test = dsets.MNIST(root = "MNIST_DATA/" ,train=False, transform = transform.ToTensor(), download=True)

batch_size = 100
dataloader = DataLoader(dataset = mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(3*3*64, 10)
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

learning_rate = 0.001
training_epochs = 15

model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

total_batch = len(dataloader)
for epoch in range(training_epochs):
    avg_cost = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        hypothesis = model(x)
        cost = criterion(hypothesis, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch

    print(f"Epoch {epoch} | Cost {avg_cost} ")

print("Learning Finished")

with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, axis = 1)

    correct = (correct_prediction == Y_test).float().sum() / len(prediction)

    print("Correct", correct)

with torch.no_grad():
    r = random.randint(0, len(mnist_test)-1)
    X = mnist_test.data[r].view(1, 1, 28, 28).float().to(device)
    Y = mnist_test.targets[r].to(device)

    pred = model(X)
    corr = torch.argmax(pred, axis = 1)

    print(f"Prediction", corr.item())

    plt.imshow(mnist_test.data[r].view(28, 28), cmap = 'Greys', interpolation='nearest')
    plt.show()





