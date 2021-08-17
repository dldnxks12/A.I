  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.models.vgg import VGG # Pretrained VGG Model
from torchvision import models
import numpy as np

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output
      
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)           
  

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)  
      
ranges = {'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}
cfg = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

vgg_model = VGGNet(requires_grad = True)
fcn = FCNs(pretrained_net = vgg_model, n_class = 2)

optimizer = optim.SGD(fcn.parameters(), lr = 0.01, momentum = 0.7)
criterion = nn.BCELoss()



x_train = np.load('./saved_x.npy')
y_train = np.load('./saved_y.npy')

x_train = np.transpose(x_train, (0, 3, 1, 2))
y_train = np.transpose(y_train, (0, 3, 1, 2))

# 2 channel image로 바꾸기 
y_train = y_train[:, :2, : ,:]

thresh_np1 = np.zeros_like(y_train[:, 0, : ,:])
thresh_np2 = np.zeros_like(y_train[:, 1, : ,:])

thresh_np1[ y_train[:, 0, : ,:] < 10] = 1
thresh_np2[ y_train[:, 1, : ,:] > 10] = 1

y_train[:, 0, : ,:] = thresh_np1
y_train[:, 1, : ,:] = thresh_np2

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

train_data  = x_train[:436]
train_label = y_train[:436]

test_data  = x_train[-1]
test_label = y_train[-1]

test_data2  = x_train[-2]
test_label2 = y_train[-2]

test_data =test_data.unsqueeze(0)
test_label =test_label.unsqueeze(0)

test_data2 =test_data2.unsqueeze(0)
test_label2 =test_label2.unsqueeze(0)

# DataLoader에 넣어줄 dataset type 생성
train_dataset = TensorDataset(train_data, train_label)

# DataLoader 
train_loader = DataLoader( dataset = train_dataset, batch_size = 40, shuffle = True, drop_last = True )

# train
avg_cost = 0

for epoch in range(1):    
    batch_length = len(train_loader)
    for x, y in train_loader:
        
        optimizer.zero_grad()
        
        output = fcn(x)
        output = F.sigmoid(output)
        
        cost = criterion(output , y)
        
        cost.backward()        
        optimizer.step()
        
        avg_cost += cost / batch_length        
        print("Loop cost : ", cost)
        
    print("Avg_cost: ", avg_cost)        
