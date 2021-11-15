import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim

import torchvision.dataset as dsets
import torchvision.transforms as transforms
from torch.utils.data as DataLoader

import numpy as np
import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_availiable() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print("Device", device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(16384, 625, bias = True),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Linear(625, 10, bias = True)

        nn.init.xavier_uniform_(self.layer5)


    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)

        return out

training_epochs = 10

model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.01).to(device)
criterion = nn.CrossEntropyLoss()

trans = transform.Compose([transform.ToTensor()])
train_data = dsets.ImageFolder(root = "~")
dataloader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, drop_last = True)

total_batch = len(dataloader)

for epoch in range(training_epochs):
    avg_cost = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        hypothesis = model(x)
        cost = criterion(hypothesis, y)

        avg_cost += cost.item() / total_batch

trans = transform.Compose([transform.Resize((64, 128), transform.ToTensor())])
dataset = dsets.ImageFolder(root = "~")
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True)

with torch.no_grad():
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        hypothesis = model(x)

        prediction = torch.argmax(hypothesis, axis = 1)
        correction = (prediction == y).sum() / len(dataloader)

with torch.no_grad():

    r = random.randint(0, len(dataloader) - 1)
    sample = dataset.__getitem__(r)[0].unsqueeze(0).to(device)
    sample_label = dataset.__getitem__(r)[1].to(device)

    prediction = model(sample)

    corr = torch.argmax(prediction, axis = 1)
