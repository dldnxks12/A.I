# pytorch 에서 제공하는 Weight initialize 방법

import torch.nn as nn
import torch.nn.functional as F

# 초기화를 연습해볼 간단한 CNN 모델
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=6, kernel_size= 3, stride = 1, padding = 1)
        self.pool  = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

        self.initialize_weights() # 이렇게 하면 객체를 실행할 때 이 함수를 자동으로 부르면서 초기화 함

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def initialize_weights(self):
        for m in self.modules():  # self.modules는 list형태로 우리가 init에서 정의한 layer들을 가지고 있다다.
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # bias를 사용한다면 0으로 초기화

            elif isinstance(m, nn.BatchNorm2d): # layer가 batch norm이라면
                nn.init.constant_(m.weight, 1)  # batch norm을 사용할 때 weight를 1로 초기화하는 것이 좋다고 한다.
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = CNN(in_channels=3, num_classes=10)

    for param in model.parameters():
        print(param)
