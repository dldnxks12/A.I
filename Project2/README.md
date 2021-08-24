### 2021 Segmentation Project in NI²L LAB at KW Univ

#### Purpose of this Project  

  - Building Segmentation Model for Ultrasonic-Wave Image Processing for medical purposes

        what to do? 
        
        가공된 데이터를 넣어 질병의 유무를 Classification 해줄 Segmentation 모델을 만들 것
        
---               
                
### To do list

#### 1주차

  - Pytorch remind

        1. Linear Regression   - ok
        2. Logistic Regression - ok
        3. Softmax Regression  - ok
        4. MLP/ANN             - ok
        5. CNN                 - ok

  - Segmentation ?

        1. Segmentation 이란?

  - Code Review

        * code review ( segtrain.py ) - ok
        
        1. argparse - ok 
        2. torch - dataloader - ok
        3. torchvision - ok 

  - 추가 개념 

        1. SOTA Browser 
        
        2. Kaggle 

            Dataset Archive 
          
        3. torchvision - deeplabv3

            Semantic Segmentation model provided by Pytorch
            
              - 현재 UltraSound에서 사용하는 모델 : deeplabv-resnet50 , deeplabv-resnet101         

        4. contigous - ok
        
            데이터 읽어 들이는 순서에 관한 함수
          
        5. permute VS view - ok 
          
            view    : 데이터 읽어들이는 순서 변경 (shape는 사실 변하지 않는다.)
            permute : Dimension Index 순서 변경
              
               permute를 사용할 경우 contigous와 같이 사용하는 경우가 많다.
          
        6. SubsetRandomSampler - ok

            전체 dataset에서 Train dataset과 Test dataset을 각각 부분 집합으로 만들어 dataloader에 넣어주는 방법 

---

#### 2주차

  - Paper & Notation 

        1. FCN - ok
        2. SegNet - ok
        3. Deconvolution - ok
        4. AE  (Auto Encoder) - ok
        5. CAE (Convolutional Auto Encoder) - ok
        6. deeplabv3
        7. 평가 지표 - ok
        8. ResNet - ok
  
  [Deconvolution](https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/)

  - 기본적인 Segmentation 구현 with Kaggle BUSI Dataset  - 80 %

        1. Segmentation 구현 with FCN, BUSI 

            BUSI Dataset 중 feature가 확실한 benign dataset을 이용
            
            Lower layer : VGG-16
            Upper layer : FCN-8s 
            
            * 문제 *
            
              데이터 전처리 부분에서 model을 모두 같은 사이즈로 만드는 과정에서 불필요한 여백이 생김
              
              이 부분을 감안하고 학습을 시켰고, 결과적으로 성능이 그다지 좋지 않았다.
              
              해당 여백이 문제인 것인지는 아직 잘 모르지만, 우선 해결하는 것이 좋아보임 
              
              (사이즈 상관없이 넣어주어도 되는 것으로 아는데, 적어도 데이터셋의 크기를 통일시켜야하나?)
                

<div align="center">

preprocessed test data
  
![image](https://user-images.githubusercontent.com/59076451/129725196-72cc0b4d-50bb-4f8e-8dbd-c18cfd8e7c93.png)
  
test label 
  
![image](https://user-images.githubusercontent.com/59076451/129725093-f61ebf10-a38d-4cd2-815c-53e6548d4575.png)
  
test result

![image](https://user-images.githubusercontent.com/59076451/129725036-cdc0b1ee-f10d-4abb-a55b-aafcbcecd1fe.png)

test result 2
  
![image](https://user-images.githubusercontent.com/59076451/129725644-3292973f-c2aa-4d7e-ab21-1a52326e3a5b.png)


with human parsiong dataset 
  
result 
  
![image](https://user-images.githubusercontent.com/59076451/130016299-604180d0-9926-4f7a-9e82-65d6dd49225d.png)  
  
</div>

  - pytorch deeplabv3 
   
        1. Segmentation 구현 with deeplabv3, BUSI



- 링크 

[Deconvolution-CAE](https://wjddyd66.github.io/pytorch/Pytorch-AutoEncoder/)<br>
[deeplabv3](https://shangom-developer.tistory.com/4)<br>
[deeplabv3](https://github.com/jfzhang95/pytorch-deeplab-xception)

[Reference Code](https://github.com/spmallick/learnopencv/tree/master/PyTorch-Segmentation-torchvision)

---

#### 3주차

- Segmenatation Customization 

    - Model 
        
          1. SegNet Review - ok
          2. SegNet 구현 - 80%
          3. U-Net Review
          4. U-Net 구현 

    - 추가 구현 

          VGG  - ok
          GoogLeNet - ok
          ResNet - ok

    - Data Preprocessing
    
          1. torchvision.datasets.ImageFolder 이용 - ok
          2. dataloader class __init__에서 전처리 
          3. data augmentation - ok 

    - 기존 모델 향상   

          1. FCN -> SegNet + BUSI + Softmax
          2. FCN -> SegNet + BUSI + Sigmoid  
          3. FCN -> U-Net + BUSI + Softmax
          4. FCN -> U-Net + BUSI + Sigmoid
          5. Weight Initialize 
          6. BCE + sigmoid / MSE + softmax - ok
                
  
  
#### 진행 결과 

<div align=center>

FCN + BUSI + BCELoss + 2ch-Sigmoid  
  
![image](https://user-images.githubusercontent.com/59076451/130317090-6d769014-2c5a-413b-9f7e-06fe4929a766.png) 
  
FCN - BUSI + MSELoss + 2ch-Softmax  
  
![image](https://user-images.githubusercontent.com/59076451/130358077-dcd75094-4ef1-46b6-b32e-da79b28e380e.png)
  
FCN - BUSI + MSELoss + 1ch-Sigmoid  
  
![image](https://user-images.githubusercontent.com/59076451/130567067-2b951db7-d418-4dec-b2f8-2b3e06ecb536.png)
  
<div>  
  
  
  
  
  
  
