import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim

import torchvision.transforms as transform
import torchvision.datasets as dsets

from torch.utils.data import DataLoader


import numpy as np
import random

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(10000, 625),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.layer4 = nn.Linear(625, 10)
        nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)

batch_size = 100

trans = transform.Compose([transform.ToTensor()])
dataset = dsets.ImageFolder(root = "files", transform = trans)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True)

for epoch in range(10):
    avg_cost = 0
    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        hypothesis = model(X)
        cost = nn.CrossEntropyLoss(hypothesis, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost.item() / len(dataloader)


# Load Test
trans = transform.Compose([transform.Resize((64, 128), transform.ToTensor())])
dataset = dsets.ImageFolder(root = "test file", transform = trans)
dataloader = DataLoader(dataset = dataset, batch_size = len(dataset), shuffle = True, drop_last = True)

with torch.no_grad():
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        hypothesis = model(X)

        prediction = torch.argmax(hypothesis, axis = 1)
        corr = (prediction == y).sum() / len(dataloader)


with torch.no_grad():
    r = random.randint(0, len(dataloader) - 1)

    sample = dataset.__getitem__(r)[0].unsqueeze(0).to(device)
    sample_label = dataset.__getitem__(r)[1].to(device)

    hypothesis = model(sample)

    prediction = torch.argmax( hypothesis, axis = 1)

    corr = prediction == sample_label
