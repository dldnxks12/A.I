# 기존에 사용하던 모델을 학습하다 저장하고, 다시 학습하던 곳 부터 시작할 때 사용하는 방법
# MNIST의 CNN Model을 예로 사용하였음

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torch.nn.init  # weight initialization을 위해 import (Xavier or HE  )

# Modern-CNN Model Class 선언
class CNN(nn.Module):

    def __init__(self):
        super().__init__()  # nn.Module 초기값 설정

        # 1 Layer Conv + Pooling
        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 2 Layer Conv + Pooling
        self.layer2 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 3 Layer FC
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)

        # fc layer에 대해서는 Weight initialization
        nn.init.xavier_uniform_(self.fc.weight)  # Linear 객체에 대한 Weight 호출 ( linear.weight , linear.bias로 가능! )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # FC layer를 위해 flatten
        out = self.fc(out)

        # out은 feature 개수 x 10 -- 이후 softmax 함수 거친 뒤 cost function 계산 해야함
        return out

# GPU Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("사용 device", device)

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Hyperparameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_DATA/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_DATA/', train=False, transform=transforms.ToTensor(), download=True)

# dataset to DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# model 객체 생성
model = CNN()

# Optimizer, cost function

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(train_loader)
total_batch2 = len(train_loader.dataset)

print("total_batch : ", total_batch)
print("total_batch.dataset : ", total_batch2)

# -------------------------------------------------------------------------------- Save

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, filename)

# -------------------------------------------------------------------------------- Load
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # acc 등 다른 것도 다 가져올 수 있다.

# ---- load model
load_model = True
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar")) # model에 저장해둔 가중치 넣어서 그 상태에서 시작

# method 2 : load_model = (torch.load("my_checkpoint.pth.tar"))
# ---- load model

for epoch in range(training_epochs):
    avg_cost = 0

    if epoch % 5:
        checkpoint = {"state_dict" : model.state_dict(), "optimizer" : optimizer.state_dict()} # model과 optimizer를 저장
        save_checkpoint(checkpoint)  # 위 코드에서 가져온 model과 optimizer를 저장할 함수 Call


    for x, y in train_loader:
        x = x.to(device)  # MLP로 할 때 처럼 28 x 28 이미지를 784 크기의 1차원 tensor로 바꾸는 과정 불필요
        y = y.to(device)

        prediction = model(x)
        cost = F.cross_entropy(prediction, y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print(epoch + 1, avg_cost)

# ---- test ---- #

with torch.no_grad():  # Gradient 학습 x

    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)  # 차원 채우기 - 10000 , 1, 28, 28
    Y_test = mnist_test.test_labels.to(device)

    # print(mnist_test) # dataset 객체
    # print(mnist_test.test_data.shape) # 객체에서 data 가져오기   # 10000 , 28, 28
    # print(mnist_test.test_labels.shape) # 객체에서 data 가져오기 # 10000,
    # print(X_test.shape) # 10000 , 1, 28, 28

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
