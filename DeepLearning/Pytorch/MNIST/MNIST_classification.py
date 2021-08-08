import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# torchvision - mnist dataset, data preprocessing 
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# dataloader 
from torch.utils.data import DataLoader

import random
import matplotlib.pyplot as plt

# GPU 사용 가능하면 사용 - Corab 또는 개인 PC에서 GPU와 CPU로 학습할 시 다음과 같은 방법을 사용하니 잘 기억하기

USE_CUDA = torch.cuda.is_available() # GPU 사용가능하면 True , else False return
device = torch.device("cuda" if USE_CUDA else "cpu")
print("학습 장비 : ", device)

# seed 고정
random.seed(777) 
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 하이퍼 파라미터
training_epochs = 15
batch_size = 100

# MNIST 데이터셋 불러오기

# root : data 다운로드 경로 , train = True : train dataset, transform : 데이터전처리 (Tensor로 가져올 것)
mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform=transforms.ToTensor(), download = True)
mnist_test = dsets.MNIST(root  = 'MNIST_data/', train = False, transform=transforms.ToTensor(), download = True )

dataloader = DataLoader(dataset = mnist_train ,batch_size = batch_size, shuffle = True, drop_last = True)

# Model Design

linear = nn.Linear(784 , 10, bias = True).to(device) # bias 기본값이 True지만 걍 해줌  device는 어디에서 연산을 수행할 것인지!
optimizer = optim.SGD(linear.parameters(), lr = 0.1)

total_batch = len(dataloader) # dataloader의 길이는 1 epoch을 돌게 하는 batch 의 개수 

for epoch in range(training_epochs):
    
    avg_cost = 0
    total_batch = len(dataloader) # dataloader의 길이는 1 epoch을 돌게 하는 batch 의 개수  -- 600 (100 x 600 = 60000)
    
    for x, y in dataloader:                
        # print(x.shape) : torch.Size([100, 1, 28, 28]) --- dataloader를 통해서 batch_size 만큼의 데이터를 뽑아온다
        # print(y.shape) : torch.Size([100])
        # Batch 크기가 100이므로 x의 크기는 100 x 784 !
        x = x.view(-1, 28*28).to(device)
        y = y.to(device)
        
        prediction = linear(x).to(device)
        cost = F.cross_entropy(prediction, y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        avg_cost += cost/total_batch 

    print(epoch, cost, avg_cost)

print("Finished")

# ---- test ---- #

with torch.no_grad(): # 이렇게 하면 gradient 계산을 수행하지 않는다.
    
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device) # torch.Size([10000, 784])
    Y_test = mnist_test.test_labels.to(device) # torch.Size([10000])
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test # 각 data가 맞으면 True, 틀리면 False
    # correct_prediction = tensor([ True,  True,  True,  ...,  True, False,  True])
    
    accuracy = correct_prediction.float().mean() # 다 합해서 평균 
    # print(accuracy.item()) : 0.8883000016212463
    
    # MNIST 테스트 데이터에서 하나 뽑아서 예측 해보자
    r = random.randint(0, len(mnist_test)-1)
    X_single_data = mnist_test.test_data[r : r+1].view(-1,28*28).float().to(device)
    Y_single_data = mnist_test.test_labels[r : r+1].to(device)
    
    single_prediction = linear(X_single_data)
    
    print(torch.argmax(single_prediction, 1))
        
    plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap = 'Greys', interpolation = 'nearest')
    plt.show()
    
