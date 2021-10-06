import torch
import torch.nn as nn
import torchvision

model = torchvision.models.vgg16(pretrained=False)  # 미리 정의된 모델을 가져옴

'''
print(model)
     ...     
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): 1. Linear Regression(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): 1. Linear Regression(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): 1. Linear Regression(in_features=4096, out_features=1000, bias=True)
  )
'''

# 위 모델 구조에서 avgpool을 없애고, classifier를 수정하는 방법

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)

'''
print(model)
  (avgpool): Identity()
  (classifier): 1. Linear Regression(in_features=512, out_features=10, bias=True)
'''

# 주의할 점은 방금 우리가 가져온 layer와 tuning한 layer들은 pretrain되어 있지 않다는 것

# 미리 학습된 Weight들과 학습되어 있지 않은 layer들을 가지고 fine tuning하는 것은 fine-tuning.py code 참고
