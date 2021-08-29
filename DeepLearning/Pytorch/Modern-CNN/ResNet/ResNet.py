# 출처 및 참고 https://deep-learning-study.tistory.com/534

# MNIST 대신 CIFAR100 Dataset으로 바꾸기

# model 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR # 

# dataset Loading - STL10 dataset from torchvision
import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# visualization
%matplotlib inline
import matplotlib.pyplot as plt
from torchvision import utils 


# utils
import time
import copy
import numpy as np

#### dataset load

train_data = dsets.MNIST('/MNIST_DATA', train = True, transform = transforms.ToTensor(), download = True)
test_data = dsets.MNIST('/MNIST_DATA', train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_data, batch_size = 100, shuffle = True, drop_last = True)
test_loader = DataLoader(dataset = test_data, batch_size = 100)

# Model Architecture

class Block(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride = 1):
        super().__init__()
        
        #batch norm에 bias가 포함 -- conv2d의 bias를 false로 설정
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True), # Input 자체를 수정 - 메모리 효율 상승 but, 원래 데이터 자체를 변경
            nn.Conv2d(out_channel, out_channel * Block.expansion, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel * Block.expansion)
            
        )
        
        self.shortcut = nn.Sequential() # Identity Mapping에 사용 --- Input과 Output의 channel x height x width가 같을 때 사용 
        self.relu     = nn.ReLU(inplace = True)
        
        # Projection Mapping with 1 x 1 Conv Layer
        
        if stride != 1 or in_channel != Block.expansion*out_channel: # Input Channel과 Output Channel이 다를 때!
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*Block.expansion, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(out_channel*Block.expansion)
            )
        
    def forward(self, x):
        
        # self.shortcut --- in out channel 안맞으면 1 x 1 conv으로 projection해서 channel 크기 맞추기, 아니면 그대로 x 출력
        # self.residual 이나 self.shortcut이나 사이즈가 줄어드는 부분은 없으므로, channel 만 맞추어주면 더하기 (skip connection)가능
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
# for decrease complexity of Network -- layer가 50개가 넘어가는 ResNet에서 사용
class BottleNeck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channel, out_channel, stride = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
        
            nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride , bias = False),
            nn.BatchNorm2d(out_channe),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding = 1,  bais = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel*BottleNeck.expansion, kernel_size = 1, stride = stride , bias = False),
            nn.BatchNorm2d(out_channe*BottleNeck.expansion)            
            
        )
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channel != out_channel*BottleNeck.expansion:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*BottleNeck.expansion, stride =stride, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channe*BottleNeck.expansion)                        
            )
            
    def forward(self,x):
        x = nn.ReLU(inplace = True )(self.residual_function(x) + self.shortcut(x))
        return x
        
# main architecture
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes = 10):
        super().__init__()
        
        self.in_channel = 32
        
        self.conv1 = nn.Sequential(
        
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1, bias = False ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)            
        )
        
        self.conv2 = self.make_layer(block, 32, num_block[0], 1)
        self.conv3 = self.make_layer(block, 64, num_block[1], 2)
        self.conv4 = self.make_layer(block, 128, num_block[2], 2)
        self.conv5 = self.make_layer(block, 256, num_block[3], 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256*block.expansion, num_classes)
        
    def make_layer(self, block, out_channel, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel*block.expansion
            
        return nn.Sequential(*layers) # 배열 전달 
    
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        
        return output # 10 channel 짜리 
        
def resnet18():        
    return ResNet(Block, [2,2,2,2])
        
res = resnet18()

optimizer = optim.SGD(res.parameters(), lr = 0.1)
avg_cost = 0
for epoch in range(1):    
    batch_length = len(train_loader)
    for x, y in train_loader:        
        optimizer.zero_grad()        
        hypothesis = res(x)        
        cost = F.cross_entropy(hypothesis, y)
        
        cost.backward()
        optimizer.step()
        
        print(f"cost : {cost}")
        avg_cost += cost / batch_length
            
    print(f"Epoch : {epoch} avg_cost : {avg_cost}")
        
