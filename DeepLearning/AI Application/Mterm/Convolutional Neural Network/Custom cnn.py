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

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Liear(16384, 625, bias = True),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Linear(625, 10, bias = True)
        nn.init.xavier_uniform_(self.layer5.weight)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)

        return out

# params
learning_rate = 0.01
training_epochs = 10

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001).to(device)
criterion = nn.CrossEntropyLoss()

trans = transform.Compose([transform.ToTensor()])
train_data = dsets.ImageFolder(root = "/content/drive/MyDrive/Colab Notebooks/인공지능 응용/train_data", transform=trans)
dataloader = DataLoader(dataset = train_data, batch_size = 100, shuffle = True, drop_last=True)

total_batch = len(dataloader)

for epoch in range(15)
    avg_cost = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        hypothesis = model(x)
        cost = criterion(hypothesis, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost.item() / total_batch

    print(f"Cost : {avg_cost}")


trans = transform.Compose([transform.Resize((64, 128)), transform.ToTensor()])
dataset = dsets.ImageFolder(root = "/content/drive/MyDrive/Colab Notebooks/인공지능 응용/test_data", transform=trans)
dataloader = DataLoader(dataset = dataset, batch_size = len(dataset))

# test
with torch.no_grad():
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        hypothesis = model(x)

        prediction = torch.argmax(hypothesis, axis = 1)
        correction = (prediction == y).sum() / len(dataloader)

        print(f" Correction {correction}")

# See Sample test
with torch.no_grad():

    r = random.randint(0, len(dataset)-1)
    x = dataset.__getitem__(r)[0].unsqueeze(0).to(device)
    y = dataset.__getitem__(r)[1].to(device)

    predict = model(x)

    corr = torch.argmax(predict, axis = 1)
    print(f"prediction : {corr} , Label : {y}")







