import torch
import torch.nn as nn
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

model = torchvision.models.vgg16(pretrained=True)

for param in model.parameters(): # 학습된 model의 파마리터들 (weight, bias)
    param.requires_grad = False # Grdient 학습을 하지 않을 것 (weight, bias 업데이트 중지)

# 이제 아래 Code에서 layer들을 수정했으므로, 이 layer들에 대해서만 학습이 진행된다.  
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)

# 기존 모델을 제외한 위 2개의 layer는 Gradient Update를 수행
