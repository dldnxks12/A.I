'''

Dataset : data를 담아두고 관리하는 Class
DataLoader : Dataset에서 data를 꺼내서 학습하는 과정을 지원해주는 Class

'''

import numpy as np
import random 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.nn.init # for xavier initializer
import torchvision.transforms as transforms # data attribute transformation
import torchvision.datasets as dsets # for dataset given by pytorch

device = 'cude' if torch.cuda.is_available() else 'cpu'
print("Device : " , device)

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

class CNN(nn.Module):
  def __init__(self, In = 3, Out = 2):
    super(CNN,self).__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(In, 32, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size = 2, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )
    self.layer4 = nn.Sequential(        
        nn.Linear(16384, 625, bias = True),
        nn.ReLU(),
        nn.Dropout(0.5)
    )
    self.layer5 = nn.Linear(625, 10, bias = True)
    nn.init.xavier_uniform_(self.layer5.weight)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.view(out.size(0), -1) # Flatten for linear layer 
    out = self.layer4(out)
    out = self.layer5(out)

    return out

# Create Instance
model = CNN().to(device)

# Parameter Setting 
learning_rate = 0.01
training_epochs = 10

criterion = torch.nn.CrossEntropyLoss() # .to(device)    # Softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataloader에 들어가는 dataset의 속성 변경하는 방법
trans = transforms.Compose([transforms.ToTensor()])
train_data = dsets.ImageFolder(root = "/content/drive/MyDrive/Colab Notebooks/인공지능 응용/train_data", transform = trans)
data_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True)

total_batch = len(data_loader)

for epoch in range(training_epochs):
  avg_cost = 0 

  for x, y in data_loader:
    x = x.to(device)
    y = y.to(device)

    prediction = model(x)
    cost = criterion(prediction, y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch

  print(f"[Epoch : {epoch+1:>4}]  Cost : {avg_cost :>.9}")  
  
trans = transforms.Compose([transforms.Resize((64,128)),transforms.ToTensor()])
test_data = dsets.ImageFolder(root = "/content/drive/MyDrive/Colab Notebooks/인공지능 응용/test_data", transform  = trans)
test_set = DataLoader(dataset = test_data, batch_size = len(test_data))

with torch.no_grad():
  for x, y in test_set:
    x = x.to(device)
    y = y.to(device)

    prediction = model(x)
    
    correct = torch.argmax(prediction, axis = 1) == y
    Acc = correct.float().mean()

    print(f"ACC : {Acc} Cost : {cost}")

with torch.no_grad():
  r = random.randint(0, len(test_data)-1)
  
  sample = test_data.__getitem__(r)[0].unsqueeze(0)
  label_sample = test_data.__getitem__(r)[1]  
  print("Label" , label_sample)
  sample_prediction = model(sample)
  print("Prediction", torch.argmax(sample_prediction, axis = 1).item())

  plt.imshow(np.transpose(test_data.__getitem__(r)[0],(1,2,0)), cmap = 'Greys', interpolation = 'nearest')
  plt.show()
    
