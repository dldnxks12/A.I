import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms

### Tensor board !
from tensorboardX import SummaryWriter # To print Tensorboard !

from torch.utils.data import DataLoader
import torch.nn.init  # weight initialization을 위해 import (Xavier or HE  )

# 5. Modern CNN Model Class 선언
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
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # FC layer를 위해 flatten
        out = self.fc(out)
        return out

# GPU Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("사용 device", device)

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Hyperparameter
learning_rate = 0.001
training_epochs = 5
batch_size = 100

# 4. MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_DATA/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_DATA/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Tensor board Writer
writer = SummaryWriter(logdir="scalar/4. MNIST")

total_batch = len(train_loader)
step = 0
for epoch in range(training_epochs):

    losses = []
    accuracies = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        prediction = model(x)
        loss = criterion(prediction, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running Training ACC
        _, predictions = prediction.max(1) # Value랑 Index return 할 것
        num_correct = (predictions == y).sum() # Index끼리 비교해서 맞는 것 Summation

        running_train_acc = float(num_correct) / float(x.shape[0]) # 맞춘 개수 / batch size

        writer.add_scalar('Training_loss', loss, global_step =step)
        writer.add_scalar('Training_ACC', running_train_acc, global_step = step)
        step += 1

        print(epoch, loss.item())
        #accuracies.append(running_train_acc)




