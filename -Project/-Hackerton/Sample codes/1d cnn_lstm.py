import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print("Device", device)

train_data = pd.read_csv("C:/Users/USER/PycharmProjects/A.I/-Project/-Hackerton/csv files/js_train_data.csv")


out_list = []
for i in range(10):
    id = train_data.loc[(train_data['id'] == i)].values[:, 2:]
    out_list.append(id)

out_list = np.array(out_list).transpose((0, 2, 1))
data = torch.from_numpy(out_list)

print("Data ok")
print("data shape check", data.shape)  # [15624, 6, 600]

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Conv1d(in_channels=6, out_channels=20, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm1d(20)
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=100, kernel_size=2, stride=2)
        self.bn2 = torch.nn.BatchNorm1d(100)
        self.layer3 = torch.nn.Conv1d(in_channels=100, out_channels=30, kernel_size=2, stride=2)
        self.bn3 = torch.nn.BatchNorm1d(30)

        self.activation = torch.nn.ReLU()

        self.fc = nn.Linear(30 * 75, 100)
        self.bn4 = torch.nn.BatchNorm1d(100)
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.layer3(x)
        x = self.activation(x)
        x = self.bn3(x)

        x = x.view(-1, 30 * 75)
        x = self.fc(x)
        x = self.bn4(x)
        x = self.activation(x)
        print(x.shape)

        x = torch.nn.LSTM(100, 61, 10)(x)

        print("lstm",x.shape)

        x = self.dropout(x)
        x = self.activation(x)

        print(x.shape)

        return x


# Create model
model = SimpleCNN().to(device)

output = model(data.float().to(device))
