# CNN으로 MNIST 분류 
# 모델 구조 Conv Layer1 + Pooling 1 -> Conv_layer2 + Pooling2 -> FullyConnected layer -> Softmax -> output 

import torch 
import torch.nn as nn

inputs = torch.Tensor(1,1,28,28) # batch_size x channel x height x width

# Layer 정의
conv1 = nn.Conv2d(1, 32, (3,3), padding = 1)
conv2 = nn.Conv2d(32, 64, (3,3), padding = 1)
pool  = nn.MaxPool2d(2)
fc    = nn.Linear(3136, 10)

# 실제로 값 넣어서 출력 보기

out = conv1(inputs) # torch.Size([1, 32, 28, 28])
out = pool(out) # torch.Size([1, 32, 14, 14])
out = conv2(out) # torch.Size([1, 64, 14, 14])
out = pool(out) # torch.Size([1, 64, 7, 7])
# .size(n) : tensor의 n번째 차원에 접근하게 해준다. ex) out.shape = tensor(1 x 1 x 32 x 64) ---  out.size(2) = 32 , out.size(3) = 64
# 첫 번째 차원인 Batch 차원은 그대로 놔두고, 나머지는 쭉 펼쳐라 (즉, batch , channel , height , width --- > batch , channel x height x width )
out = out.view(out.size(0),-1) # torch.Size([1, 3136])
out = fc(out) #torch.Size([1, 10])

# 이 후 softmax를 통해 분류 한다.

'''
input : Tensor Size torch.Size([1, 1, 28, 28])
Conv1 : Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Conv2 : Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Pool  : MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
'''
