import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.nn.init # for xavier initializer
import torchvision.transforms as transforms # data attribute transformation
import torchvision.datasets as dsets # for dataset given by pytorch


device = 'cude' if torch.cuda.is_available() else 'cpu'
print("Device : " , device)

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

class CNN(nn.Module):
  def __init__(self, In = 1, Out = 10):
    super(CNN,self).__init__()

    # Input image size : batch x 1 x 28 x 28 
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
        nn.Linear(128*4*4, 625, bias = True),
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
  
x = np.ones((1,1,28,28))
x = torch.from_numpy(x).float()

model = CNN()
out = model(x)
print("Input shape", x.shape)
print("output shape", out.shape)
