# MSE Loss  , Softmax 사용 

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
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

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
                
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) # batch x 2 x 224 x 224 

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  
        x4 = output['x4'] 
        x3 = output['x3']  
        x2 = output['x2'] 
        x1 = output['x1']  

        score = self.bn1(self.relu(self.deconv1(x5)))    
        score = score + x4                                
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = score + x3                                
        score = self.bn3(self.relu(self.deconv3(score))) 
        score = score + x2                                
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score + x1                                
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  # size=(N, n_class, x.H/1, x.W/1)  
      
ranges = {'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}
cfg = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

vgg_model = VGGNet(requires_grad = True)
fcn = FCNs(pretrained_net = vgg_model, n_class = 2)

optimizer = optim.SGD(fcn.parameters(), lr = 0.01, momentum = 0.7)
criterion = nn.MSELoss() 

# data load
train_x = np.load("numpy_x.npy")
train_y = np.load("numpy_y.npy")

# BG와 Object 2개의 Class
train_x = np.transpose(train_x, (0,3,1,2))
train_y = np.transpose(train_y, (0,3,1,2))

train_y = train_y[:, :2, :, :]

# BUSI label threshold 
thresh_np1 = np.zeros_like(train_y[:, 0, : ,:])
thresh_np2 = np.zeros_like(train_y[:, 1, : ,:])

thresh_np1[ train_y[:, 0, : ,:] < 10] = 1
thresh_np2[ train_y[:, 1, : ,:] > 10] = 1

train_y[:, 0, : ,:] = thresh_np1
train_y[:, 1, : ,:] = thresh_np2

# numpy to Tensor 
train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)

# test data select
test_x = train_x[0]
test_y = train_y[0]
test_x2 = train_x[1]
test_y2 = train_y[1]
test_x = test_x.unsqueeze(0)
test_y = test_y.unsqueeze(0)
test_x2 = test_x2.unsqueeze(0)
test_y2 = test_y2.unsqueeze(0)

train_x = train_x[1:]
train_y = train_y[1:]

# DataLoader 
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader( dataset = train_dataset, batch_size = 50, shuffle = True, drop_last = True )

# train
for epoch in range(3):    
    avg_cost = 0
    batch_length = len(train_loader)
    for x, y in train_loader:
        
        optimizer.zero_grad()
        
        output = fcn(x)       
        output = F.softmax(output, 1)       # ---------------------- channel에 대해 Softmax
        cost = criterion(output , y)        
        cost.backward()        
        optimizer.step()
        avg_cost += cost / batch_length        
        print("epoch & Loop cost : ", epoch, cost)
        
    print("Avg_cost: ", avg_cost)        
    
    
# color map
def decode_segmap(image, nc = 1):
  
    label_colors = np.array([(255, 255, 255), (0, 0, 0)]) 

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb    
  
with torch.no_grad():
    prediction = fcn(test_x2)
    prediction = F.softmax(prediction, 1)  # Channel에 대해서 softmax 
    prediction = prediction.squeeze()

    pred = torch.argmax(prediction, dim = 0).detach().numpy() # max value index return 
    
       
rgb_pred = decode_segmap(pred)
print(rgb_pred.shape)     # 224 x 224 x 3

# See Segmented Map && Label Map 
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(rgb_pred[:,:,:])
plt.subplot(1,2,2)
plt.imshow(test_y2[0][0])
plt.show()
  
