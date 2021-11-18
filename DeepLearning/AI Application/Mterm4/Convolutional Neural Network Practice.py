import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transform

from torch.utils.data as DataLoader

import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.manual_seed_all(777)

class CNN(nn.Mudule):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # image size = 3 x 64 x 128
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)  ## 32 x 32 x 64
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3 , stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride  =2) # 128 x 8 x 16
        )

        self.layer4 = nn.Sequential(
            nn.Linear(128*8*16, 625),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer5 = nn.Linear(625, 10)
        nn.init.xavier_uniform_.(self.layer5.weight)

    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)

        return out

learning_rate = 0.01
batch_size = 100

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr =learning_rate)
criterion = nn.CrossEntropyLoss()

trans = transform.Compose([transform.ToTensor()])
dataset = dsets.ImageFolder(root = "file", transform = trans)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True)

total_batch = len(dataloader)

for epoch in range(10):
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        hypothesis = model(X)
        cost = criterion(hypothesis, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

# test

trans = transform.Compose([transform.Resize(64, 128), transform.ToTensor()])
dataset = dsets.ImageFolder(root = "testfile", transform = trans)
dataloader = DataLoader(dataset = dataset, batch_size = len(dataset), shuffle = True, drop_last = True)

with torch.no_grad():

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        hypothesis = model(X)

        prediction = torch.argmax(hypothesis, axis = 1)
        correction = (prediction == y).sum() /  total_batch

with torch.no_grad():
    r = random.randint(0, len(dataset)-1)

    sample = dataloader.__getitem__(r)[0].unsqueeze(0).to(device)
    sample_label = dataloader.__getitem__(r)[1].to(device)

    hypothesis = model(sample)
    prediction = torch.argmax(sample, axis = 1)
    correction = prediction == y







