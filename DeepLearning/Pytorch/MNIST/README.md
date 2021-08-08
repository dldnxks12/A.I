#### Multi-Class Classification을 이용한 MNIST dataset 분류

    MNIST Dataset : 총 60,000개의 train data와 train label data, 10,000개의 test data와 test label data로 구성

    label은 0 ~ 9까지 총 10개

    data 1개는 각각 28x28 크기의 이미지 

---

#### dataloader

    dataset의 크기가 방대하므로, dataloader를 통해 mini_batch 학습을 진행한다.
  
---  
   
#### data 전처리
 
    Softmax 함수에 넣어서 총 10개의 Class로 분류하는 과정을 거칠 것 
  
    1. 28x28 pixel의 이미지를 1x784 vector로 만든 다음 행렬 곱을 통해 1x10 vector로 만들어주어야한다.
  
    2. 1 x 10 vector를 만들었으면 label data와의 loss를 계산해서 cost를 구한다 (F.cross_entropy)
    

#### torchvision module 

    torchvision은 유명한 데이터셋(dataset), 이미 구현되있는 유명한 모델(models), 그리고 일반적인 이미지 전처리 도구들(transforms)을 포함하고 있는 유용한 패키지이다.

    링크 : https://pytorch.org/docs/stable/torchvision/index.html  
  
  
